use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Device},
};

use crate::spherical_harmonics;
use crate::{camera::Camera, utils};
use burn::tensor::Tensor;

use burn::tensor::activation::sigmoid;

use crate::renderer::RenderPackage;

// A Gaussian splat model.
// This implementation wraps CUDA kernels from (Kerbel and Kopanas et al, 2023).
#[derive(Module, Debug)]
pub(crate) struct GaussianSplats<B: Backend> {
    // Current and maximum spherical harmonic degree. This is increased over
    // training.
    active_sh_degree: i32,
    max_sh_degree: i32,

    // f32[n,3]. Position.
    xyz: Param<Tensor<B, 2>>,

    // f32[n,1,3]. SH coefficients for diffuse color.
    sh_dc: Param<Tensor<B, 3>>,

    // f32[n,sh-1,3]. Remaining SH coefficients.
    sh_rest: Param<Tensor<B, 3>>,

    // f32[n,4]. Rotation as quaternion matrices.
    rotation: Param<Tensor<B, 2>>,

    // f32[n]. Opacity parameters.
    opacity: Param<Tensor<B, 1>>,

    // f32[n,3]. Scale matrix coefficients.
    scale: Param<Tensor<B, 2>>,

    // Non trainable params.

    // f32[n]. Maximum projected radius of each Gaussian in pixel-units. It is
    // later used during culling.
    max_radii_2d: Tensor<B, 1>,

    // Helper tensors for accumulating the viewspace_xyz gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    //
    // Sum of gradient norms for each Gaussian in pixel-units. This accumulator
    // is incremented when a Gaussian is visible in a training batch.
    xyz_gradient_accum: Tensor<B, 1>,

    // Number of times a Gaussian is visible in a training batch.
    denom: Tensor<B, 1>,
}

struct Config {
    position_lr_scale: f32,
}

impl<B: Backend> GaussianSplats<B> {
    pub(crate) fn new(
        xyz: Tensor<B, 2>,
        rgb: Tensor<B, 2>,
        scale: Option<Tensor<B, 2>>,
        rotation: Option<Tensor<B, 2>>,
        opacity: Option<Tensor<B, 1>>,
        max_sh_degree: i32,
        active_sh_degree: i32,
        device: &Device<B>,
    ) -> GaussianSplats<B> {
        // Activation function for scale. Since scale values must be >0, use an
        // activation function to go between (-inf, inf) to (0, inf).
        let inverse_scale_activation = "log";
        let scale_activation = "exp";

        // Activation function for opacity. Opacity values must be in (0, 1).
        let inverse_opacity_activation = "inverse_sigmoid";
        let opacity_activation = "sigmoid";

        let num_points = xyz.dims()[0];

        // f32[n, 1, 3]. Diffuse color, aka the first spherical harmonic coefficient
        // for each color channel.
        let sh_dc = spherical_harmonics::rgb_to_sh_dc(rgb);
        let sh_dc = sh_dc.unsqueeze_dim(1);

        // f32[n,sh-1,3]. All other spherical harmonic coefficients.
        let sh_rest = Tensor::zeros(
            [
                num_points as usize,
                ((max_sh_degree + 1).pow(2) - 1) as usize,
                3,
            ],
            device,
        );

        let init_rotation = if let Some(rotation) = rotation {
            rotation
        } else {
            Tensor::from_floats([1.0, 0.0, 0.0, 0.0], device)
                .unsqueeze::<2>()
                .repeat(0, num_points)
        };

        let init_opacity = if let Some(opacity) = opacity {
            utils::inverse_sigmoid(opacity)
        } else {
            utils::inverse_sigmoid(Tensor::from_floats([0.1], device)).repeat(0, num_points)
        };

        let init_scale = if let Some(scale) = scale {
            Tensor::log(scale)
        } else {
            // If scale is not given, initialize each point's scale to the average
            // squared distance between each the point and its k=4 nearest neighbors.
            // Initial scale will be equal in all directions.
            //   let dist = Tensor::sqrt(
            //       Tensor::max(gs_utils.average_sqd_knn(xyz), 0.0000001)
            //   );

            // TODO: Fancy KNN init.
            let dist = Tensor::random(
                [num_points, 3],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                device,
            );
            Tensor::log(dist)
        };

        // Model parameters.
        GaussianSplats {
            active_sh_degree: active_sh_degree,
            max_sh_degree: max_sh_degree,
            xyz: xyz.into(),
            sh_dc: sh_dc.into(),
            sh_rest: sh_rest.into(),
            rotation: init_rotation.into(),
            opacity: init_opacity.into(),
            scale: init_scale.into(),
            max_radii_2d: Tensor::zeros([num_points], device),
            xyz_gradient_accum: Tensor::zeros([num_points], device),
            denom: Tensor::zeros([num_points], device),
        }
    }

    // Sets up this class for the training loop.

    // Args:
    //   cfg: ...
    //   position_lr_scale: Multiplier for learning rate for positions.  Larger
    //     values mean higher learning rates.

    pub(crate) fn training_setup(&mut self, cfg: Config, position_lr_scale: f32) {
        // Initialize accumulators to empty.
        self.xyz_gradient_accum = Tensor::zeros_like(&self.xyz_gradient_accum);
        self.denom = Tensor::zeros_like(&self.xyz_gradient_accum);

        // TODO: All this custom ADAM bullshit.
        //     // Setup different optimizer arguments for different parameters.
        //   param_dict = [
        //       {
        //           'params': [self.xyz],
        //           'lr': cfg.position_lr_init * position_lr_scale,
        //           'name': 'xyz',
        //       },
        //       {'params': [self.sh_dc], 'lr': cfg.feature_lr, 'name': 'sh_dc'},
        //       {
        //           'params': [self.sh_rest],
        //           # Low learning rate for SH coefficients.
        //           'lr': cfg.feature_lr / 20.0,
        //           'name': 'sh_rest',
        //       },
        //       {'params': [self.opacity], 'lr': cfg.opacity_lr, 'name': 'opacity'},
        //       {'params': [self.scale], 'lr': cfg.scale_lr, 'name': 'scale'},
        //       {'params': [self.rotation], 'lr': cfg.rotation_lr, 'name': 'rotation'},
        //   ]

        //   self.optimizer = torch.optim.Adam(param_dict, eps=1e-15)
        //   self.xyz_lr_function = gs_utils.ExponentialDecayLr(
        //       lr_init=cfg.position_lr_init * position_lr_scale,
        //       lr_final=cfg.position_lr_final * position_lr_scale,
        //       lr_delay_mult=cfg.position_lr_delay_mult,
        //       max_steps=cfg.position_lr_max_steps,
        //   )
    }

    // Learning rate scheduling per step."""
    pub(crate) fn update_learning_rate(&mut self, iteration: i32) {
        // TODO: More custom LR bullshit :/
        // Merge with setting up a custom adam?
        // for param_group in self.optimizer.param_groups {
        //     if param_group['name'] == 'xyz' {
        //     lr = self.xyz_lr_function(iteration)
        //     param_group['lr'] = lr
        //     }
        // }
    }

    // Returns SH coefficients.
    pub(crate) fn get_total_sh(self) -> Tensor<B, 3> {
        Tensor::cat(vec![*self.sh_dc, *self.sh_rest], 1)
        // torch.cat((self.sh_dc, self.sh_rest), dim=1)
    }

    // One-up sh degree.
    pub(crate) fn oneup_sh_degree(&mut self) {
        if self.active_sh_degree < self.max_sh_degree {
            self.active_sh_degree += 1
        }
    }

    // Updates rolling statistics that we capture during rendering.
    pub(crate) fn update_rolling_statistics(&mut self, render_pkg: RenderPackage<B>) {
        let radii = render_pkg.radii;
        let screenspace_points = render_pkg.screenspace_points;

        let visible_mask = render_pkg.radii.greater_elem(0.0);

        // TODO: This is not as efficient as could be...
        // Want these operations to be sparse.
        self.max_radii_2d = radii.mask_where(
            visible_mask,
            Tensor::cat(vec![radii.unsqueeze(), self.max_radii_2d.unsqueeze()], 0).max_dim(0),
        );

        // TODO: How do we get grads here? Would need to be sure B: AutoDiffBackend.
        // let grad = screenspace_points.
        // self.xyz_gradient_accum[visibility_filter] += torch.norm(
        //     screenspace_points.grad[visibility_filter, :2], dim=-1, keepdim=True
        // );

        self.denom = self.denom + visible_mask.float();
    }

    /// Resets all the opacities to 0.01.
    pub(crate) fn reset_opacity(self) {
        let opacities_new = utils::inverse_sigmoid(sigmoid(self.opacity.val()).clamp_max(0.01));

        // TODO: Wtf.
        // Update optimizer with the new tensor
        //   optimizable_tensors = gs_adam_helpers.replace_tensor_to_optimizer(
        //       self.optimizer, opacities_new, 'opacity'
        //   );

        //   // Make sure that the tensor we are storing is the same tensor the
        //   // optimizer is optimizing
        //   self.opacity = optimizable_tensors['opacity'];
    }

    // // Densifies and prunes the Gaussians.

    // // Args:
    // //   max_grad: See densify_by_clone() and densify_by_split().
    // //   min_opacity_threshold: Gaussians with an opacity lower than this will be
    // //     deleted.
    // //   max_pixel_threshold: Optional. If specified, prune Gaussians whose radius
    // //     is larger than this in pixel-units.
    // //   max_world_size_threshold: Optional. If specified, prune Gaussians whose
    // //     radius is larger than this in world coordinates.
    // //   clone_vs_split_size_threshold: See densify_by_clone() and
    // //     densify_by_split().
    // fn densify_and_prune(
    //     self,
    //     max_grad: f32,
    //     min_opacity_threshold: f32,
    //     max_pixel_threshold: f32,
    //     max_world_size_threshold: f32,
    //     clone_vs_split_size_threshold: f32,
    //     device: &Device<B>,
    // ) {

    //   // f32[n,1]. Compute average magnitude of the gradient for each Gaussian in
    //   // pixel-units while accounting for the number of times each Gaussian was
    //   // seen during training.
    //   let grads = self.xyz_gradient_accum / self.denom;
    //   grads[grads.isnan()] = 0.0;

    //   self.densify_by_clone(grads, max_grad, clone_vs_split_size_threshold, device);
    //   self.densify_by_split(grads, max_grad, clone_vs_split_size_threshold, 2, device);

    //   // bool[n]. If True, delete these Gaussians.
    //   let prune_mask = (
    //       self.opacity_activation(self.opacity) < min_opacity_threshold
    //   ).squeeze();

    //   if let Some(threshold) = max_pixel_threshold {
    //     // Delete Gaussians with too large of a radius in pixel-units.
    //     let big_points_vs = self.max_radii_2d > max_pixel_threshold;

    //     // Delete Gaussians with too large of a radius in world-units.
    //     let big_points_ws =
    //         self.scale_activation(self.scale).max_dim(1).values
    //         > max_world_size_threshold;

    //     let prune_mask = Tensor::logical_or(
    //         Tensor::logical_or(prune_mask, big_points_vs), big_points_ws
    //     );
    // }

    //   self.prune_points(prune_mask);
    // }

    // // Prunes points based on the given mask.
    // //
    // // Args:
    // //   mask: bool[n]. If True, prune this Gaussian.
    // fn prune_points(&mut self, mask: Tensor<B, 2>) {
    //     // TODO: Ehh not sure how/what.
    // //   let valid_points_mask = 1.0 - mask;

    // //   let optimizable_tensors = gs_adam_helpers.prune_optimizer(
    // //       self.optimizer, valid_points_mask
    // //   );

    // //   self.xyz = optimizable_tensors['xyz'];
    // //   self.sh_dc = optimizable_tensors['sh_dc'];
    // //   self.sh_rest = optimizable_tensors['sh_rest'];
    // //   self.opacity = optimizable_tensors['opacity'];
    // //   self.scale = optimizable_tensors['scale'];
    // //   self.rotation = optimizable_tensors['rotation'];

    // //   self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask];
    // //   self.denom = self.denom[valid_points_mask];
    // //   self.max_radii_2d = self.max_radii_2d[valid_points_mask];
    // }

    // // Densifies Gaussians by splitting.

    // // Args:
    // //   grads: f32[n,1]. Average squared magnitude of the gradient for each
    // //     Gaussian in pixel-units.
    // //   grad_threshold: Minimum gradient magnitude for
    // //   clone_vs_split_size_threshold: Threshold on scale in world units.
    // //     Gaussians which meet the gradient condition and have a scale larger than
    // //     this are split into `n_splits` new Gaussians.
    // //   n_splits: Number of new Gaussians to create for each split Gaussian.
    // fn densify_by_split(
    //     &mut self,
    //     grads: Tensor<B, 2>,
    //     grad_threshold: f32,
    //     clone_vs_split_size_threshold: f32,
    //     n_splits: i32,
    //     device: &Device<B>
    // ) {

    //   let n_init_points = self.xyz.dims()[0];
    //   // f32[n]. Extract points that satisfy the gradient condition.
    //   let padded_grad = Tensor::zeros([n_init_points], device);
    //   padded_grad.slice_assign([0..grads.dims()[0]], grads);

    //   // Decide which Gaussians are eligible for splitting or cloning based on
    //   // their gradient magnitude.
    //   let selected_pts_mask = padded_grad >= grad_threshold;

    //   // Gaussians are split if their radius in world-units exceeds a threshold.
    //   selected_pts_mask = Tensor::logical_and(
    //       selected_pts_mask,
    //       Tensor::max_dim(self.scale_activation(self.scale), 1).values
    //       > clone_vs_split_size_threshold,
    //   );

    //   // Sample position of each new Gaussian.
    //   let stds = self.scale_activation(self.scale[selected_pts_mask]).repeat(
    //       n_splits, 1
    //   );
    //   let means = torch.zeros((stds.size(0), 3), device);
    //   let samples = torch.normal(mean=means, std=stds);
    //   let rots = gs_utils.qvec2rotmat(self.rotation[selected_pts_mask]).repeat(
    //       n_splits, 1, 1
    //   );
    //   let new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.xyz[
    //       selected_pts_mask
    //   ].repeat(n_splits, 1);

    //   // Set the scale of each new Gaussian to approximately 1/k of its parent.
    //   let new_scale = self.inverse_scale_activation(
    //       self.scale_activation(self.scale[selected_pts_mask]).repeat(n_splits, 1)
    //       / (0.8 * n_splits)
    //   );

    //   // Split Gaussians inherit remaining properties from their parent.
    //   let new_rotation = self.rotation[selected_pts_mask].repeat(n_splits, 1);
    //   let new_sh_dc = self.sh_dc[selected_pts_mask].repeat(n_splits, 1, 1);
    //   let new_sh_rest = self.sh_rest[selected_pts_mask].repeat(n_splits, 1, 1);
    //   let new_opacity = self.opacity[selected_pts_mask].repeat(n_splits, 1);

    //   self.densification_postfix(
    //       new_xyz,
    //       new_sh_dc,
    //       new_sh_rest,
    //       new_opacity,
    //       new_scale,
    //       new_rotation,
    //   );

    //   let prune_filter = torch.cat((
    //       selected_pts_mask,
    //       torch.zeros(
    //           n_splits * selected_pts_mask.sum()
    //       ),
    //   ));

    //   self.prune_points(prune_filter);
    // }

    // // Densifies Gaussians by cloning.
    // //
    // // Args:
    // //   grads: f32[n,1]. Average squared magnitude of the gradient for each
    // //     Gaussian in pixel-units.
    // //   grad_threshold: Minimum gradient magnitude for
    // //   clone_vs_split_size_threshold: Threshold on scale in world units.
    // //     Gaussians which meet the gradient condition and have a scale smaller
    // //     than this are cloned with the exact same parameters.
    // fn densify_by_clone(
    //     &mut self,
    //     grads: Tensor<B, 2>,
    //     grad_threshold: f32,
    //     clone_vs_split_size_threshold: f32,
    //     device: &Device<B>,
    // ) {

    //   // Extract points that satisfy the gradient condition
    //   let selected_pts_mask = Tensor::where(
    //       torch.norm(grads, dim=-1) >= grad_threshold, true, false
    //   );

    //   // From those choose only the ones that are small enough to be cloned
    //   selected_pts_mask = Tensor::logical_and(
    //       selected_pts_mask,
    //       Tensor::max_dim(self.scale_activation(self.scale), 1).values
    //       <= clone_vs_split_size_threshold,
    //   );

    //   let new_xyz = self.xyz[selected_pts_mask];
    //   let new_sh_dc = self.sh_dc[selected_pts_mask];
    //   let new_sh_rest = self.sh_rest[selected_pts_mask];
    //   let new_opacities = self.opacity[selected_pts_mask];
    //   let new_scale = self.scale[selected_pts_mask];
    //   let new_rotation = self.rotation[selected_pts_mask];

    //   self.densification_postfix(
    //       new_xyz,
    //       new_sh_dc,
    //       new_sh_rest,
    //       new_opacities,
    //       new_scale,
    //       new_rotation,
    //   );
    // }

    // // Updates the optimizer by appending the new tensors.
    // fn densification_postfix(
    //     self,
    //     new_xyz: Tensor<B, 2>,
    //     new_features_dc: Tensor<B, 3>,
    //     new_features_rest: Tensor<B, 3>,
    //     new_opacities: Tensor<B, 2>,
    //     new_scale: Tensor<B, 2>,
    //     new_rotation: Tensor<B, 2>,
    // ) {
    //   tensors_dict = {
    //       'xyz': new_xyz,
    //       'sh_dc': new_features_dc,
    //       'sh_rest': new_features_rest,
    //       'opacity': new_opacities,
    //       'scale': new_scale,
    //       'rotation': new_rotation,
    //   };

    //   optimizable_tensors = gs_adam_helpers.cat_tensors_to_optimizer(
    //       self.optimizer, tensors_dict
    //   );

    //   self.xyz = optimizable_tensors['xyz'];
    //   self.sh_dc = optimizable_tensors['sh_dc'];
    //   self.sh_rest = optimizable_tensors['sh_rest'];
    //   self.opacity = optimizable_tensors['opacity'];
    //   self.scale = optimizable_tensors['scale'];
    //   self.rotation = optimizable_tensors['rotation'];

    //   self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device='cuda');
    //   self.denom = torch.zeros((self.xyz.shape[0], 1), device='cuda');
    //   self.max_radii_2d = torch.zeros((self.xyz.shape[0]), device='cuda');
    // }

    // Renders an image by splatting the gaussians.
    // Args:
    //   camera: Camera to render.
    //   bg_color: Background color.
    // Returns:
    //   A tuple of which the first element is the rendered image and the second
    //   elements is a dictionary consisting of statistics that we need to keep
    //   track
    //   during training. More specifically:
    //   * screenspace_points: a tensor that "holds" the viewspace positional
    //     gradients
    //   * visibility_filter: a boolean tensor that indicates which gaussians
    //     participated in the rendering.
    //   * radii: the maximum screenspace radius of each gaussian
    pub(crate) fn render_engine(
        self,
        camera: Camera,
        bg_color: Tensor<B, 1>,
        device: &Device<B>,
    ) -> RenderPackage<B> {
        let shs = self.get_total_sh();
        let opacity = burn::tensor::activation::sigmoid(*self.opacity);
        let scale = (*self.scale).exp();
        let rotation = *self.rotation; // TODO: torch.nn.functional.normalize?

        let (rendered_image, screenspace_points, radii) = crate::renderer::render(
            camera,
            *self.xyz,
            shs,
            self.active_sh_degree,
            opacity,
            scale,
            rotation,
            bg_color,
            device,
        );

        RenderPackage {
            image: rendered_image,
            screenspace_points: screenspace_points,
            radii: radii,
        }
    }
}
