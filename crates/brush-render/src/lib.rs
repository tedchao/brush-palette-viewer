#![allow(clippy::too_many_arguments)]
#![allow(clippy::single_range_in_vec_init)]
use brush_kernel::bitcast_tensor;
use burn::backend::Autodiff;
use burn::tensor::{ElementConversion, Int, Tensor};
use burn_jit::JitBackend;
use burn_wgpu::{JitTensor, WgpuRuntime};
use camera::Camera;

mod dim_check;
mod kernels;
mod shaders;

pub mod camera;
pub mod render;

#[derive(Debug, Clone)]
pub struct RenderAux {
    pub uniforms_buffer: JitTensor<WgpuRuntime, u32, 1>,
    pub projected_splats: JitTensor<WgpuRuntime, f32, 2>,
    pub num_intersections: JitTensor<WgpuRuntime, u32, 1>,
    pub num_visible: JitTensor<WgpuRuntime, u32, 1>,
    pub final_index: JitTensor<WgpuRuntime, u32, 2>,
    pub cum_tiles_hit: JitTensor<WgpuRuntime, u32, 1>,
    pub tile_bins: JitTensor<WgpuRuntime, u32, 3>,
    pub compact_gid_from_isect: JitTensor<WgpuRuntime, u32, 1>,
    pub global_from_compact_gid: JitTensor<WgpuRuntime, u32, 1>,
}

#[derive(Debug, Clone)]
pub struct RenderStats {
    pub num_visible: u32,
    pub num_intersections: u32,
}

impl RenderAux {
    pub fn read_num_visible(&self) -> u32 {
        Tensor::<JitBackend<WgpuRuntime, f32, i32>, 1, Int>::from_primitive(bitcast_tensor(
            self.num_visible.clone(),
        ))
        .into_scalar()
        .elem()
    }

    pub fn read_num_intersections(&self) -> u32 {
        Tensor::<JitBackend<WgpuRuntime, f32, i32>, 1, Int>::from_primitive(bitcast_tensor(
            self.num_intersections.clone(),
        ))
        .into_scalar()
        .elem()
    }

    pub fn read_tile_depth(&self) -> Tensor<JitBackend<WgpuRuntime, f32, i32>, 2, Int> {
        let bins = Tensor::from_primitive(bitcast_tensor(self.tile_bins.clone()));
        let [ty, tx, _] = bins.dims();
        let max = bins.clone().slice([0..ty, 0..tx, 1..2]).squeeze(2);
        let min = bins.clone().slice([0..ty, 0..tx, 0..1]).squeeze(2);
        max - min
    }
}

/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend {
    // Render splats
    // Project splats processing step. This produces
    // a whole bunch of gradients that we store.
    // The return just happens to be the xy screenspace points
    // which we use to 'carry' the gradients'.
    fn render_splats(
        cam: &Camera,
        img_size: glam::UVec2,
        means: Tensor<Self, 2>,
        xy_dummy: Tensor<Self, 2>,
        log_scales: Tensor<Self, 2>,
        quats: Tensor<Self, 2>,
        colors: Tensor<Self, 2>,
        raw_opacity: Tensor<Self, 1>,
        background: glam::Vec3,
        render_u32_buffer: bool,
    ) -> (Tensor<Self, 3>, RenderAux);
}

pub trait AutodiffBackend: burn::tensor::backend::AutodiffBackend + Backend {}
impl<B: Backend> AutodiffBackend for Autodiff<B> where burn::backend::Autodiff<B>: Backend {}
