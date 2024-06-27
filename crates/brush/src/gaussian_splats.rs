use brush_render::camera::Camera;
use brush_render::{render::num_sh_coeffs, AutodiffBackend, Backend};
use burn::tensor::Distribution;
use burn::tensor::Tensor;
use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    tensor::Device,
};

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
        Splats::init_random(self.num_points, self.aabb_scale, device)
    }
}

#[derive(Module, Debug)]
pub struct Splats<B: Backend> {
    pub(crate) means: Param<Tensor<B, 2>>,
    pub(crate) sh_coeffs: Param<Tensor<B, 2>>,
    pub(crate) rotation: Param<Tensor<B, 2>>,
    pub(crate) raw_opacity: Param<Tensor<B, 1>>,
    pub(crate) log_scales: Param<Tensor<B, 2>>,

    // Dummy input to track screenspace gradient.
    pub(crate) xys_dummy: Tensor<B, 2>,
}

impl<B: Backend> Splats<B> {
    pub(crate) fn init_random(num_points: usize, aabb_scale: f32, device: &Device<B>) -> Splats<B> {
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
        let init_rotation = Tensor::<_, 1>::from_floats([1.0, 0.0, 0.0, 0.0], device)
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
    ) -> (Tensor<B, 3>, brush_render::RenderAux<B>) {
        brush_render::render::render(
            camera,
            img_size,
            self.means.val(),
            self.xys_dummy.clone(),
            self.log_scales.val(),
            self.rotation.val(),
            self.sh_coeffs.val(),
            self.raw_opacity.val(),
            bg_color,
            render_u32_buffer,
        )
    }

    #[cfg(feature = "rerun")]
    pub(crate) fn visualize(&self, rec: &RecordingStream) -> Result<()> {
        let means = self.means.val().into_data().convert().value;
        let means = means.chunks(3).map(|c| glam::vec3(c[0], c[1], c[2]));

        let sh_c0 = 0.2820947917738781;
        let base_rgb = self.sh_coeffs.val().slice([0..self.num_splats(), 0..3]) * sh_c0 + 0.5;
        let colors = base_rgb.into_data().convert::<f32>().value;
        let colors = colors.chunks(3).map(|c| {
            Color::from_rgb(
                (c[0] * 255.0) as u8,
                (c[1] * 255.0) as u8,
                (c[2] * 255.0) as u8,
            )
        });

        let radii = self
            .log_scales
            .val()
            .exp()
            .into_data()
            .convert::<f32>()
            .value;
        let radii = radii
            .chunks(3)
            .map(|c| 0.5 * glam::vec3(c[0], c[1], c[2]).length());

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

    pub fn norm_rotations(&mut self) {
        self.rotation = self.rotation.clone().map(|x| {
            let norms = Tensor::sum_dim(x.clone().powf_scalar(2.0), 1).sqrt();
            let norm_rotation = x / Tensor::clamp_min(norms, 1e-6);
            Tensor::from_inner(norm_rotation.inner()).require_grad()
        });
    }
}
