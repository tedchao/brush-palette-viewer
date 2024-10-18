use crate::bounding_box::BoundingBox;
use crate::camera::Camera;
use crate::safetensor_utils::safetensor_to_burn;
use crate::shaders;
use crate::{render::num_sh_coeffs, Backend};
use burn::config::Config;
use burn::tensor::activation::sigmoid;
use burn::tensor::Tensor;
use burn::tensor::{Distribution, Shape};
use burn::{
    module::{Module, Param, ParamId},
    tensor::Device,
};
use glam::Vec3;
use kiddo::{KdTree, SquaredEuclidean};
use safetensors::SafeTensors;

#[derive(Config)]
pub struct RandomSplatsConfig {
    // period of steps where refinement is turned off
    #[config(default = 50000)]
    init_count: usize,
    #[config(default = -2.0)]
    opacity_min: f64,
    #[config(default = -1.0)]
    opacity_max: f64,
    #[config(default = 0.05)]
    scale_min: f64,
    #[config(default = 0.15)]
    scale_max: f64,
}

#[derive(Module, Debug)]
pub struct Splats<B: Backend> {
    pub means: Param<Tensor<B, 2>>,
    pub sh_coeffs: Param<Tensor<B, 2>>,
    pub rotation: Param<Tensor<B, 2>>,
    pub raw_opacity: Param<Tensor<B, 1>>,
    pub log_scales: Param<Tensor<B, 2>>,

    // Dummy input to track screenspace gradient.
    pub xys_dummy: Tensor<B, 2>,
}

pub fn inverse_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

impl<B: Backend> Splats<B> {
    pub fn from_random_config(
        config: RandomSplatsConfig,
        bounds: BoundingBox,
        device: &B::Device,
    ) -> Self {
        let num_points = config.init_count;

        let min = bounds.min();
        let max = bounds.max();

        let xx: Tensor<_, 1> = Tensor::random(
            [num_points],
            Distribution::Uniform(min.x as f64, max.x as f64),
            device,
        );
        let yy: Tensor<_, 1> = Tensor::random(
            [num_points],
            Distribution::Uniform(min.y as f64, max.y as f64),
            device,
        );
        let zz: Tensor<_, 1> = Tensor::random(
            [num_points],
            Distribution::Uniform(min.z as f64, max.z as f64),
            device,
        );
        let means = Tensor::stack(vec![xx, yy, zz], 1);
        let num_coeffs = num_sh_coeffs(0);

        let sh_coeffs = Tensor::random(
            [num_points, num_coeffs * 3],
            Distribution::Uniform(-0.5, 0.5),
            device,
        );
        let init_rotation = Tensor::<_, 1>::from_floats([1.0, 0.0, 0.0, 0.0], device)
            .unsqueeze::<2>()
            .repeat_dim(0, num_points);

        let init_raw_opacity = Tensor::random(
            [num_points],
            Distribution::Uniform(config.opacity_min, config.opacity_max),
            device,
        );

        // TODO: Fancy KNN init.
        let scale_min = config.scale_min.ln();
        let scale_max = config.scale_max.ln();

        let init_scale = Tensor::random(
            [num_points, 3],
            Distribution::Uniform(scale_min, scale_max),
            device,
        );

        Splats::from_data(
            means,
            sh_coeffs,
            init_rotation,
            init_raw_opacity,
            init_scale,
            device,
        )
    }

    pub fn from_point_cloud(
        positions: Vec<Vec3>,
        colors: Vec<Vec3>,
        device: &B::Device,
    ) -> Splats<B> {
        let num_points = positions.len();
        let positions_arr: Vec<_> = positions.into_iter().map(|v| [v.x, v.y, v.z]).collect();
        let means: Vec<f32> = positions_arr.iter().copied().flatten().collect();
        let means = Tensor::<B, 1>::from_floats(means.as_slice(), device).reshape([num_points, 3]);

        let colors: Vec<f32> = colors.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
        let colors =
            Tensor::<B, 1>::from_floats(colors.as_slice(), device).reshape([num_points, 3]);

        // TODO: TO SH
        let sh_coeffs = (colors - 0.5) / shaders::gather_grads::SH_C0;

        // let dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001);
        // let scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3);

        let init_rotation = Tensor::<_, 1>::from_floats([1.0, 0.0, 0.0, 0.0], device)
            .unsqueeze::<2>()
            .repeat_dim(0, num_points);

        let raw_opacities = Tensor::ones(Shape::new([num_points]), device) * inverse_sigmoid(0.1);

        // use the kiddo::KdTree type to get up and running quickly with default settings
        let tree: KdTree<_, 3> = (&positions_arr).into();
        let extents: Vec<_> = positions_arr
            .iter()
            .map(|p| {
                // Get average of 3 nearest squared distances.
                tree.nearest_n::<SquaredEuclidean>(p, 3)
                    .iter()
                    .map(|x| x.distance * x.distance)
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();

        let scales = Tensor::<B, 1>::from_floats(extents.as_slice(), device)
            .reshape([num_points, 1])
            .repeat_dim(1, 3);

        let log_scales = scales.clamp(0.0000001, f32::MAX).log();

        Self::from_data(
            means,
            sh_coeffs,
            init_rotation,
            raw_opacities,
            log_scales,
            device,
        )
    }

    pub fn from_data(
        means: Tensor<B, 2>,
        sh_coeffs: Tensor<B, 2>,
        rotation: Tensor<B, 2>,
        raw_opacity: Tensor<B, 1>,
        log_scales: Tensor<B, 2>,
        device: &Device<B>,
    ) -> Self {
        let num_points = means.shape().dims[0];
        Splats {
            means: Param::initialized(ParamId::new(), means.detach().require_grad()),
            sh_coeffs: Param::initialized(ParamId::new(), sh_coeffs.detach().require_grad()),
            rotation: Param::initialized(ParamId::new(), rotation.detach().require_grad()),
            raw_opacity: Param::initialized(ParamId::new(), raw_opacity.detach().require_grad()),
            log_scales: Param::initialized(ParamId::new(), log_scales.detach().require_grad()),
            xys_dummy: Tensor::zeros([num_points, 2], device).require_grad(),
        }
    }

    pub fn map_param<const D: usize>(
        tensor: &mut Param<Tensor<B, D>>,
        f: impl Fn(Tensor<B, D>) -> Tensor<B, D>,
    ) {
        *tensor = tensor.clone().map(|x| f(x).detach().require_grad());
    }

    pub fn render(
        &self,
        camera: &Camera,
        img_size: glam::UVec2,
        bg_color: glam::Vec3,
        render_u32_buffer: bool,
    ) -> (Tensor<B, 3>, crate::RenderAux) {
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

    pub fn opacity(&self) -> Tensor<B, 1> {
        sigmoid(self.raw_opacity.val())
    }

    pub fn scale(&self) -> Tensor<B, 2> {
        self.log_scales.val().exp()
    }

    pub fn num_splats(&self) -> usize {
        self.means.dims()[0]
    }

    pub fn norm_rotations(&mut self) {
        Self::map_param(&mut self.rotation, |x| {
            x.clone() / Tensor::clamp_min(Tensor::sum_dim(x.powf_scalar(2.0), 1).sqrt(), 1e-6)
        });
    }

    pub fn from_safetensors(tensors: &SafeTensors, device: &B::Device) -> anyhow::Result<Self> {
        let means = safetensor_to_burn::<B, 2>(tensors.tensor("means")?, device);
        let num_points = means.dims()[0];
        let log_scales = safetensor_to_burn::<B, 2>(tensors.tensor("scales")?, device);

        // TODO: This doesn't really handle SH properly. Probably should serialize this in the format
        // we expect and save this reshape hassle.
        let sh_coeffs =
            safetensor_to_burn::<B, 3>(tensors.tensor("coeffs")?, device).reshape([num_points, 3]);
        let quats = safetensor_to_burn::<B, 2>(tensors.tensor("quats")?, device);
        let raw_opacity = safetensor_to_burn::<B, 1>(tensors.tensor("opacities")?, device);

        Ok(Self::from_data(
            means,
            sh_coeffs,
            quats,
            raw_opacity,
            log_scales,
            device,
        ))
    }
}
