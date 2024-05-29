use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    tensor::{activation::sigmoid, Bool, Device},
};
use tracing::info_span;

use crate::{
    camera::Camera,
    splat_render::{self, AutodiffBackend, Aux, Backend},
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
}

impl<B: Backend> SplatsTrainState<B> {
    pub fn new(num_points: usize, device: &Device<B>) -> Self {
        Self {
            max_radii_2d: Tensor::zeros([num_points], device),
            xy_grad_norm_accum: Tensor::zeros([num_points], device),
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
            Tensor::random([num_points], Distribution::Uniform(-2.0, -1.0), device);

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
    pub(crate) fn update_stats(&mut self, aux: &Aux<B>, xys_grad: Tensor<B, 2>) {
        let radii = Tensor::zeros_like(&aux.radii_compact).select_assign(
            0,
            aux.global_from_compact_gid.clone(),
            aux.radii_compact.clone(),
        );

        self.max_radii_2d = Tensor::max_pair(self.max_radii_2d.clone(), radii.clone());

        self.xy_grad_norm_accum = Tensor::max_pair(
            self.xy_grad_norm_accum.clone(),
            xys_grad
                .clone()
                .mul(xys_grad.clone())
                .sum_dim(1)
                .squeeze(1)
                .sqrt(),
        );
    }
}

// Densifies and prunes the Gaussians.
// Args:
//   max_grad: See densify_by_clone() and densify_by_split().
//   max_pixel_threshold: Optional. If specified, prune Gaussians whose radius
//     is larger than this in pixel-units.
//   max_world_size_threshold: Optional. If specified, prune Gaussians whose
//     radius is larger than this in world coordinates.
//   clone_vs_split_size_threshold: See densify_by_clone() and
//     densify_by_split().
pub fn densify_and_prune<B: AutodiffBackend>(
    splats: &mut Splats<B>,
    state: &mut SplatsTrainState<B>,
    grad_threshold: f32,
    max_pixel_threshold: Option<f32>,
    max_world_size_threshold: Option<f32>,
    clone_vs_split_size_threshold: f32,
    device: &Device<B>,
) {
    if let Some(threshold) = max_pixel_threshold {
        // Delete Gaussians with too large of a radius in pixel-units.
        let big_splats_mask = state.max_radii_2d.clone().greater_elem(threshold);
        prune_points(splats, state, big_splats_mask)
    }

    if let Some(threshold) = max_world_size_threshold {
        // Delete Gaussians with too large of a radius in world-units.
        let prune_mask = splats
            .log_scales
            .val()
            .exp()
            .max_dim(1)
            .squeeze(1)
            .greater_elem(threshold);
        prune_points(splats, state, prune_mask);
    }

    // Compute average magnitude of the gradient for each Gaussian in
    // pixel-units while accounting for the number of times each Gaussian was
    // seen during training.
    let grads = state.xy_grad_norm_accum.clone();

    // self.densify_by_clone(grads, max_grad, clone_vs_split_size_threshold, device);

    let big_grad_mask = grads.greater_equal_elem(grad_threshold);
    let split_clone_size_mask = splats
        .log_scales
        .val()
        .exp()
        .max_dim(1)
        .squeeze(1)
        .lower_elem(clone_vs_split_size_threshold);

    let clone_mask = Tensor::stack::<2>(
        vec![split_clone_size_mask.clone(), big_grad_mask.clone()],
        1,
    )
    .all_dim(1)
    .squeeze::<1>(1);
    let split_mask = Tensor::stack::<2>(vec![split_clone_size_mask.bool_not(), big_grad_mask], 1)
        .all_dim(1)
        .squeeze::<1>(1);

    // Need to be very careful not to do any operations with this tensor, as it might be
    // less than the minimum size wgpu can support :/
    let clone_where = clone_mask.clone().argwhere();

    if clone_where.dims()[0] >= 4 {
        let clone_inds = clone_where.squeeze(1);

        // Extract points that satisfy the gradient condition
        // let selected_pts_mask = Tensor::where(
        //     torch.norm(xy_grads, dim=-1) >= grad_threshold, true, false
        // );

        //  Clone inds.
        // let new_xyz = self.xyz[selected_pts_mask];
        // let new_sh_dc = self.sh_dc[selected_pts_mask];
        // let new_sh_rest = self.sh_rest[selected_pts_mask];
        // let new_opacities = self.opacity[selected_pts_mask];
        // let new_scale = self.scale[selected_pts_mask];
        // let new_rotation = self.rotation[selected_pts_mask];

        let new_means = splats.means.val().select(0, clone_inds.clone());
        let new_rots = splats.rotation.val().select(0, clone_inds.clone());
        let new_coeffs = splats.sh_coeffs.val().select(0, clone_inds.clone());
        let new_opac = splats.raw_opacity.val().select(0, clone_inds.clone());
        let new_scales = splats.log_scales.val().select(0, clone_inds.clone());

        merge_splats(
            splats, new_means, new_rots, new_coeffs, new_opac, new_scales,
        );
    }

    let split_where = split_mask.clone().argwhere();
    if split_where.dims()[0] >= 4 {
        // self.densify_by_split(grads, max_grad, clone_vs_split_size_threshold, 2, device);

        // f32[n]. Extract points that satisfy the gradient condition.
        // let padded_grad = Tensor::zeros([n_init_points], device);
        // padded_grad.slice_assign([0..grads.dims()[0]], grads);
        let split_inds = split_where.squeeze(1);
        let samps = split_inds.dims()[0];

        let scaled_samples = splats
            .log_scales
            .val()
            .select(0, split_inds.clone())
            .repeat(0, 2)
            .exp()
            * Tensor::random([samps * 2, 3], Distribution::Normal(0.0, 1.0), device);

        // Remove original points we're splitting.
        // TODO: Could just replace them? Maybe?

        let repeats = 2;

        // TODO: Rotate samples
        let new_means = scaled_samples
            + splats
                .means
                .val()
                .select(0, split_inds.clone())
                .repeat(0, repeats);
        let new_rots = splats
            .rotation
            .val()
            .select(0, split_inds.clone())
            .repeat(0, repeats);
        let new_coeffs = splats
            .sh_coeffs
            .val()
            .select(0, split_inds.clone())
            .repeat(0, repeats);
        let new_opac = splats
            .raw_opacity
            .val()
            .select(0, split_inds.clone())
            .repeat(0, repeats);
        let new_scales = (splats.log_scales.val().select(0, split_inds.clone()).exp() / 1.6)
            .log()
            .repeat(0, repeats);
        prune_points(splats, state, split_mask.clone());

        merge_splats(
            splats, new_means, new_rots, new_coeffs, new_opac, new_scales,
        );
    }
}

pub(crate) fn reset_opacity<B: AutodiffBackend>(splats: &mut Splats<B>) {
    splats.raw_opacity = splats
        .raw_opacity
        .clone()
        .map(|x| Tensor::from_inner((x - 2.0).inner()).require_grad());
}

pub fn merge_splats<B: AutodiffBackend>(
    splats: &mut Splats<B>,
    new_means: Tensor<B, 2>,
    new_rots: Tensor<B, 2>,
    sh_coeffs: Tensor<B, 2>,
    raw_opacity: Tensor<B, 1>,
    log_scales: Tensor<B, 2>,
) {
    // Concat new params.
    splats.means = splats.means.clone().map(|x| {
        Tensor::from_inner(Tensor::cat(vec![x, new_means.clone()], 0).inner()).require_grad()
    });
    splats.rotation = splats.rotation.clone().map(|x| {
        Tensor::from_inner(Tensor::cat(vec![x, new_rots.clone()], 0).inner()).require_grad()
    });
    splats.sh_coeffs = splats.sh_coeffs.clone().map(|x| {
        Tensor::from_inner(Tensor::cat(vec![x, sh_coeffs.clone()], 0).inner()).require_grad()
    });
    splats.raw_opacity = splats.raw_opacity.clone().map(|x| {
        Tensor::from_inner(Tensor::cat(vec![x, raw_opacity.clone()], 0).inner()).require_grad()
    });
    splats.log_scales = splats.log_scales.clone().map(|x| {
        Tensor::from_inner(Tensor::cat(vec![x, log_scales.clone()], 0).inner()).require_grad()
    });
}

pub fn prune_invisible_points<B: AutodiffBackend>(
    splats: &mut Splats<B>,
    state: &mut SplatsTrainState<B>,
    cull_alpha_thresh: f32,
) {
    let alpha_mask = sigmoid(splats.raw_opacity.val()).lower_elem(cull_alpha_thresh);
    prune_points(splats, state, alpha_mask);
}

// Prunes points based on the given mask.
//
// Args:
//   mask: bool[n]. If True, prune this Gaussian.
pub fn prune_points<B: AutodiffBackend>(
    splats: &mut Splats<B>,
    train_state: &mut SplatsTrainState<B>,
    prune: Tensor<B, 1, Bool>,
) {
    // bool[n]. If True, delete these Gaussians.
    let valid_inds = prune.bool_not().argwhere().squeeze(1);

    let start_splats = splats.num_splats();
    let new_points = valid_inds.dims()[0];

    if new_points < start_splats {
        train_state.max_radii_2d = train_state
            .max_radii_2d
            .clone()
            .select(0, valid_inds.clone());
        train_state.xy_grad_norm_accum = train_state
            .xy_grad_norm_accum
            .clone()
            .select(0, valid_inds.clone());

        splats.means = splats
            .means
            .clone()
            .map(|x| Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad());
        splats.sh_coeffs = splats
            .sh_coeffs
            .clone()
            .map(|x| Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad());
        splats.rotation = splats
            .rotation
            .clone()
            .map(|x| Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad());
        splats.raw_opacity = splats
            .raw_opacity
            .clone()
            .map(|x| Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad());
        splats.log_scales = splats
            .log_scales
            .clone()
            .map(|x| Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad());
    }
}
