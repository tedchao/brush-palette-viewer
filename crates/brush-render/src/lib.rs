#![allow(clippy::too_many_arguments)]
#![allow(clippy::single_range_in_vec_init)]
use burn::backend::Autodiff;
use burn::prelude::Tensor;
use burn::tensor::{ElementConversion, Int};
use burn_jit::JitBackend;
use burn_wgpu::WgpuRuntime;
use camera::Camera;

mod dim_check;
mod kernels;
mod safetensor_utils;
mod shaders;

pub mod bounding_box;
pub mod camera;
pub mod gaussian_splats;
pub mod render;

#[derive(Debug, Clone)]
pub struct RenderAux<B: Backend> {
    /// The packed projected splat information, see ProjectedSplat in helpers.wgsl
    pub projected_splats: B::FloatTensorPrimitive,

    pub uniforms_buffer: B::IntTensorPrimitive,
    pub num_intersections: B::IntTensorPrimitive,
    pub num_visible: B::IntTensorPrimitive,
    pub final_index: B::IntTensorPrimitive,
    pub cum_tiles_hit: B::IntTensorPrimitive,
    pub tile_bins: B::IntTensorPrimitive,
    pub compact_gid_from_isect: B::IntTensorPrimitive,
    pub global_from_compact_gid: B::IntTensorPrimitive,
}

#[derive(Debug, Clone)]
pub struct RenderStats {
    pub num_visible: u32,
    pub num_intersections: u32,
}

impl<B: Backend> RenderAux<B> {
    pub async fn read_num_visible(&self) -> u32 {
        Tensor::<B, 1, Int>::from_primitive(self.num_visible.clone())
            .into_scalar_async()
            .await
            .elem()
    }

    pub async fn read_num_intersections(&self) -> u32 {
        Tensor::<B, 1, Int>::from_primitive(self.num_intersections.clone())
            .into_scalar_async()
            .await
            .elem()
    }

    pub fn read_tile_depth(&self) -> Tensor<B, 2, Int> {
        let bins = Tensor::from_primitive(self.tile_bins.clone());
        let [ty, tx, _] = bins.dims();
        let max = bins.clone().slice([0..ty, 0..tx, 1..2]).squeeze(2);
        let min = bins.clone().slice([0..ty, 0..tx, 0..1]).squeeze(2);
        max - min
    }
}

// Custom operations in Burn work by extending the backend with an extra func.
pub trait Backend: burn::tensor::backend::Backend {
    /// Render splats to a buffer.
    ///
    /// This projects the gaussians, sorts them, and rasterizes them to a buffer, in a\
    /// differentiable way.
    /// The arguments are all passed as raw tensors. See [`Splats`] for a convenient Module that wraps this fun
    /// The ['xy_dummy'] variable is only used to carry screenspace xy gradients.
    /// This function can optionally render a "u32" buffer, which is a packed RGBA (8 bits per channel)
    /// buffer. This is useful when the results need to be displayed immediatly.
    fn render_splats(
        cam: &Camera,
        img_size: glam::UVec2,
        means: Tensor<Self, 2>,
        xy_grad_dummy: Tensor<Self, 2>,
        log_scales: Tensor<Self, 2>,
        quats: Tensor<Self, 2>,
        sh_coeffs: Tensor<Self, 3>,
        raw_opacity: Tensor<Self, 1>,
        background: glam::Vec3,
        render_u32_buffer: bool,
    ) -> (Tensor<Self, 3>, RenderAux<Self>);
}

pub trait AutodiffBackend: burn::tensor::backend::AutodiffBackend + Backend {}
impl<B: Backend> AutodiffBackend for Autodiff<B> where burn::backend::Autodiff<B>: Backend {}

pub type PrimaryBackend = JitBackend<WgpuRuntime, f32, i32>;
