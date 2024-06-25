#![allow(clippy::too_many_arguments)]
#![allow(clippy::single_range_in_vec_init)]
use burn::backend::Autodiff;
use burn::prelude::Int;
use burn::tensor::Tensor;
use camera::Camera;
mod dim_check;
mod kernels;
mod shaders;

pub mod camera;
pub mod render;
pub mod sync_span;

#[derive(Debug, Clone)]
pub struct RenderAux<B: Backend> {
    pub uniforms_buffer: Tensor<B, 1, Int>,
    pub projected_splats: Tensor<B, 2>,

    pub num_intersects: Tensor<B, 1, Int>,
    pub num_visible: Tensor<B, 1, Int>,

    pub final_index: Tensor<B, 2, Int>,
    pub cum_tiles_hit: Tensor<B, 1, Int>,

    pub tile_bins: Tensor<B, 3, Int>,
    pub compact_gid_from_isect: Tensor<B, 1, Int>,
    pub global_from_compact_gid: Tensor<B, 1, Int>,
}

impl<B: Backend> RenderAux<B> {
    pub fn calc_tile_depth(&self) -> Tensor<B, 2, Int> {
        let bins = self.tile_bins.clone();
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
    fn render_gaussians(
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
    ) -> (Tensor<Self, 3>, RenderAux<Self>);
}

pub trait AutodiffBackend: burn::tensor::backend::AutodiffBackend + Backend {}
impl<B: Backend> AutodiffBackend for Autodiff<B> where burn::backend::Autodiff<B>: Backend {}
