use brush_render::camera::Camera;
use brush_render::{render::num_sh_coeffs, Backend};
use burn::tensor::Distribution;
use burn::tensor::Tensor;
use burn::{
    module::{Module, Param, ParamId},
    tensor::Device,
};

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

fn map_param<B: Backend, const D: usize>(
    tensor: &mut Param<Tensor<B, D>>,
    f: impl Fn(Tensor<B, D>) -> Tensor<B, D>,
) {
    *tensor = tensor.clone().map(|x| f(x).detach().require_grad());
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
    ) -> (Tensor<B, 3>, brush_render::RenderAux) {
        B::render_splats(
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

    pub(crate) fn num_splats(&self) -> usize {
        self.means.dims()[0]
    }

    pub fn concat_splats(
        &mut self,
        means: Tensor<B, 2>,
        rotations: Tensor<B, 2>,
        sh_coeffs: Tensor<B, 2>,
        raw_opacities: Tensor<B, 1>,
        log_scales: Tensor<B, 2>,
    ) {
        map_param(&mut self.means, |x| Tensor::cat(vec![x, means.clone()], 0));
        map_param(&mut self.rotation, |x| {
            Tensor::cat(vec![x, rotations.clone()], 0)
        });
        map_param(&mut self.sh_coeffs, |x| {
            Tensor::cat(vec![x, sh_coeffs.clone()], 0)
        });
        map_param(&mut self.raw_opacity, |x| {
            Tensor::cat(vec![x, raw_opacities.clone()], 0)
        });
        map_param(&mut self.log_scales, |x| {
            Tensor::cat(vec![x, log_scales.clone()], 0)
        });
    }

    pub fn norm_rotations(&mut self) {
        map_param(&mut self.rotation, |x| {
            x.clone() / Tensor::clamp_min(Tensor::sum_dim(x.powf_scalar(2.0), 1).sqrt(), 1e-6)
        });
    }
}
