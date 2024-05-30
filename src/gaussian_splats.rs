use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    tensor::Device,
};
use tracing::info_span;

use crate::{
    camera::Camera,
    splat_render::{self, AutodiffBackend, Backend},
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

    pub(crate) fn render(
        &self,
        camera: &Camera,
        img_size: glam::UVec2,
        bg_color: glam::Vec3,
        render_u32_buffer: bool,
    ) -> (Tensor<B, 3>, crate::splat_render::RenderAux<B>) {
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
            render_u32_buffer,
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

// TODO: This really shouldn't need autodiff. Burn is very confused if you try to
// create new tensors from a param. The autodiff graph can get all messed up, doing this
// weird from_inner/inner dance seems to fix it, but can only be done on an autodiff backend.
impl<B: AutodiffBackend> Splats<B> {
    pub fn concat_splats(
        &mut self,
        new_means: Tensor<B, 2>,
        new_rots: Tensor<B, 2>,
        sh_coeffs: Tensor<B, 2>,
        raw_opacity: Tensor<B, 1>,
        log_scales: Tensor<B, 2>,
    ) {
        // Concat new params.
        self.means = self.means.clone().map(|x| {
            Tensor::from_inner(Tensor::cat(vec![x, new_means.clone()], 0).inner()).require_grad()
        });
        self.rotation = self.rotation.clone().map(|x| {
            Tensor::from_inner(Tensor::cat(vec![x, new_rots.clone()], 0).inner()).require_grad()
        });
        self.sh_coeffs = self.sh_coeffs.clone().map(|x| {
            Tensor::from_inner(Tensor::cat(vec![x, sh_coeffs.clone()], 0).inner()).require_grad()
        });
        self.raw_opacity = self.raw_opacity.clone().map(|x| {
            Tensor::from_inner(Tensor::cat(vec![x, raw_opacity.clone()], 0).inner()).require_grad()
        });
        self.log_scales = self.log_scales.clone().map(|x| {
            Tensor::from_inner(Tensor::cat(vec![x, log_scales.clone()], 0).inner()).require_grad()
        });
    }
}
