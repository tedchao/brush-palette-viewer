use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    tensor::{activation::sigmoid, Device},
};
use tracing::info_span;

use crate::{
    camera::Camera,
    splat_render::{self, Aux, Backend},
};
use burn::tensor::Distribution;
use burn::tensor::Tensor;

use anyhow::Result;

#[cfg(feature = "rerun")]
use rerun::{Color, RecordingStream};

#[derive(Config)]
pub(crate) struct SplatsConfig {
    num_points: usize,
    aabb_scale: f32,
    max_sh_degree: u32,
    position_lr_scale: f32,
}

impl SplatsConfig {
    pub(crate) fn build<B: Backend>(&self, device: &Device<B>) -> Splats<B> {
        Splats::new(self.num_points, self.aabb_scale, device)
    }
}

// A Gaussian splat model.
// This implementation wraps CUDA kernels from (Kerbel and Kopanas et al, 2023).
#[derive(Module, Debug)]
pub struct Splats<B: Backend> {
    // f32[n, 3]. Position.
    pub(crate) means: Param<Tensor<B, 2>>,
    // f32[n, sh]. SH coefficients for diffuse color.
    pub(crate) sh_coeffs: Param<Tensor<B, 2>>,
    // f32[n, 4]. Rotation as quaternion matrices.
    pub(crate) rotation: Param<Tensor<B, 2>>,
    // f32[n]. Opacity parameters.
    pub(crate) raw_opacity: Param<Tensor<B, 1>>,
    // f32[n, 3]. Scale matrix coefficients.
    pub(crate) log_scales: Param<Tensor<B, 2>>,

    pub(crate) xys_dummy: Tensor<B, 2>,
}

pub struct SplatsTrainState<B: Backend> {
    // Non trainable params.
    // Maximum projected radius of each Gaussian in pixel-units. It is
    // later used during culling.
    max_radii_2d: Tensor<B, 1>,

    // Helper tensors for accumulating the viewspace_xyz gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    //
    // Sum of gradient norms for each Gaussian in pixel-units. This accumulator
    // is incremented when a Gaussian is visible in a training batch.
    xy_grad_norm_accum: Tensor<B, 1>,

    // Number of times a Gaussian was visible so far.
    denom: Tensor<B, 1>,
}

impl<B: Backend> SplatsTrainState<B> {
    pub fn new(num_points: usize, device: &Device<B>) -> Self {
        Self {
            max_radii_2d: Tensor::zeros([num_points], device),
            xy_grad_norm_accum: Tensor::zeros([num_points], device),
            denom: Tensor::zeros([num_points], device),
        }
    }
}

pub fn num_sh_coeffs(degree: usize) -> usize {
    (degree + 1).pow(2)
}

pub fn sh_basis_from_coeffs(degree: usize) -> usize {
    match degree {
        1 => 0,
        4 => 1,
        9 => 2,
        16 => 3,
        25 => 4,
        _ => panic!("Invalid nr. of sh bases {degree}"),
    }
}

impl<B: Backend> Splats<B> {
    pub(crate) fn new(num_points: usize, aabb_scale: f32, device: &Device<B>) -> Splats<B> {
        let extent = (aabb_scale as f64) / 2.0;
        let means = Tensor::random(
            [num_points, 3],
            Distribution::Uniform(-extent, extent),
            device,
        );

        let num_coeffs = num_sh_coeffs(0);

        let sh_coeffs = Tensor::random(
            [num_points, num_coeffs * 3],
            Distribution::Uniform(-0.5, 0.5),
            device,
        );
        let init_rotation = Tensor::from_floats([1.0, 0.0, 0.0, 0.0], device)
            .unsqueeze::<2>()
            .repeat(0, num_points);

        let init_raw_opacity =
            Tensor::random([num_points], Distribution::Uniform(-4.0, 4.0), device);

        // TODO: Fancy KNN init.
        let init_scale = Tensor::random([num_points, 3], Distribution::Uniform(-3.0, -2.0), device);

        // TODO: Support lazy loading.
        // Model parameters.
        Splats {
            means: Param::initialized(ParamId::new(), means.require_grad()),
            sh_coeffs: Param::initialized(ParamId::new(), sh_coeffs.require_grad()),
            rotation: Param::initialized(ParamId::new(), init_rotation.require_grad()),
            raw_opacity: Param::initialized(ParamId::new(), init_raw_opacity.require_grad()),
            log_scales: Param::initialized(ParamId::new(), init_scale.require_grad()),
            xys_dummy: Tensor::zeros([num_points, 2], device).require_grad(),
        }
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
    //     &mut self,
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
    //   let grads = self.mean_grads / self.denom;
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

    // // // Densifies Gaussians by splitting.
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
    //   let n_init_points = self.num_splats();

    //   // f32[n]. Extract points that satisfy the gradient condition.
    //   let padded_grad = Tensor::zeros([n_init_points], device);
    //   padded_grad.slice_assign([0..grads.dims()[0]], grads);

    //   // Decide which Gaussians are eligible for splitting or cloning based on
    //   // their gradient magnitude.
    //   // Gaussians are split if their radius in world-units exceeds a threshold.
    //   selected_pts_mask = Tensor::logical_and(
    //     padded_grad >= grad_threshold,
    //       Tensor::max_dim(self.scale_activation(self.scale), 1).values  > clone_vs_split_size_threshold,
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
    //         new_xyz,
    //         new_sh_dc,
    //         new_sh_rest,
    //         new_opacity,
    //         new_scale,
    //         new_rotation,
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
    //     xy_grads: Tensor<B, 2>,
    //     grad_threshold: f32,
    //     clone_vs_split_size_threshold: f32,
    //     device: &Device<B>,
    // ) {
    //   // Extract points that satisfy the gradient condition
    //   let selected_pts_mask = Tensor::where(
    //       torch.norm(xy_grads, dim=-1) >= grad_threshold, true, false
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
    pub(crate) fn render(
        &self,
        camera: &Camera,
        img_size: glam::UVec2,
        bg_color: glam::Vec3,
    ) -> (Tensor<B, 3>, crate::splat_render::Aux<B>) {
        let _span = info_span!("Splats render").entered();
        let cur_rot = self.rotation.val();

        // TODO: Norm after grad, not on render.
        let norms = Tensor::sum_dim(cur_rot.clone() * cur_rot.clone(), 1).sqrt();
        let norm_rotation = cur_rot / Tensor::clamp_min(norms, 1e-6);

        splat_render::render::render(
            camera,
            img_size,
            self.means.val(),
            self.xys_dummy.clone(),
            self.log_scales.val(),
            norm_rotation,
            self.sh_coeffs.val(),
            self.raw_opacity.val(),
            bg_color,
        )
    }

    #[cfg(feature = "rerun")]
    pub(crate) fn visualize(&self, rec: &RecordingStream) -> Result<()> {
        use crate::utils;
        use ndarray::Axis;
        let means_data = utils::burn_to_ndarray(self.means.val());
        let means = means_data
            .axis_iter(Axis(0))
            .map(|c| glam::vec3(c[0], c[1], c[2]));

        let num_points = self.sh_coeffs.shape().dims[0];
        let base_rgb = self.sh_coeffs.val().slice([0..num_points, 0..3]) + 0.5;

        // TODO: Fix for SH.
        let colors_data = utils::burn_to_ndarray(base_rgb);
        let colors = colors_data.axis_iter(Axis(0)).map(|c| {
            Color::from_rgb(
                (c[[0]] * 255.0) as u8,
                (c[[1]] * 255.0) as u8,
                (c[[2]] * 255.0) as u8,
            )
        });

        let scales_data = utils::burn_to_ndarray(self.log_scales.val().exp());
        let radii = scales_data
            .axis_iter(Axis(0))
            .map(|c| 0.5 * 0.33 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt());

        rec.log(
            "world/splat/points",
            &rerun::Points3D::new(means)
                .with_colors(colors)
                .with_radii(radii),
        )?;
        Ok(())
    }

    pub(crate) fn num_splats(&self) -> usize {
        self.means.dims()[0]
    }
}

impl<B: Backend> SplatsTrainState<B> {
    // Args:
    //   cfg: ...
    //   position_lr_scale: Multiplier for learning rate for positions.  Larger
    //     values mean higher learning rates.

    // Updates rolling statistics that we capture during rendering.
    pub(crate) fn update_stats(&mut self, aux_data: &Aux<B>, xys_grad: Tensor<B, 2>) {
        let visible = aux_data.radii.clone().greater_elem(0.0);
        self.max_radii_2d = Tensor::max_pair(self.max_radii_2d.clone(), aux_data.radii.clone());

        self.xy_grad_norm_accum = self.xy_grad_norm_accum.clone()
            + xys_grad
                .clone()
                .mul(xys_grad.clone())
                .sum_dim(1)
                .squeeze(1)
                .sqrt();

        self.denom = self.denom.clone() + visible.float();
    }

    fn reset(&mut self) {
        self.xy_grad_norm_accum = Tensor::zeros_like(&self.xy_grad_norm_accum);
        self.denom = Tensor::zeros_like(&self.denom);
        self.max_radii_2d = Tensor::zeros_like(&self.max_radii_2d);
    }
}

// Prunes points based on the given mask.
//
// Args:
//   mask: bool[n]. If True, prune this Gaussian.
pub fn prune_invisible_point<B: splat_render::AutodiffBackend>(
    splats: Splats<B>,
    train_state: &mut SplatsTrainState<B>,
    min_opacity_threshold: f32,
) -> Splats<B> {
    println!("{:?}", splats.means.dims());

    // bool[n]. If True, delete these Gaussians.
    let valid_inds = sigmoid(splats.raw_opacity.clone().val())
        .greater_elem(min_opacity_threshold)
        .argwhere()
        .squeeze(1);

    let start_splats = splats.num_splats();
    let new_points = valid_inds.dims()[0];

    if new_points < start_splats {
        println!("Pruned {:?} splats", start_splats - new_points);

        train_state.max_radii_2d = train_state
            .max_radii_2d
            .clone()
            .select(0, valid_inds.clone());
        train_state.xy_grad_norm_accum = train_state
            .xy_grad_norm_accum
            .clone()
            .select(0, valid_inds.clone());
        train_state.denom = train_state.denom.clone().select(0, valid_inds.clone());

        return Splats {
            means: splats.means.map(|x| {
                Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad()
            }),
            sh_coeffs: splats.sh_coeffs.map(|x| {
                Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad()
            }),
            rotation: splats.rotation.map(|x| {
                Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad()
            }),
            raw_opacity: splats.raw_opacity.map(|x| {
                Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad()
            }),
            log_scales: splats.log_scales.map(|x| {
                Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad()
            }),
            xys_dummy: splats.xys_dummy,
        };
    }
    splats
}
