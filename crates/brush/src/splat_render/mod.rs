use crate::camera::Camera;
use brush_kernel::BurnBack;
use burn::backend::Autodiff;
use burn::prelude::Int;
use burn::tensor::Tensor;
mod dim_check;
mod kernels;
mod shaders;

pub mod render;
pub mod sync_span;

#[derive(Debug, Clone)]
pub(crate) struct RenderAux<B: Backend> {
    pub num_visible: Tensor<B, 1, Int>,
    pub num_intersects: Tensor<B, 1, Int>,
    pub tile_bins: Tensor<B, 3, Int>,
    pub radii_compact: Tensor<B, 1>,
    pub depthsort_gid_from_isect: Tensor<B, 1, Int>,
    pub compact_from_depthsort_gid: Tensor<B, 1, Int>,
    pub depths: Tensor<B, 1>,
    pub cum_tiles_hit: Tensor<B, 1, Int>,
    pub conic_comps: Tensor<B, 2>,
    pub colors: Tensor<B, 2>,
    pub final_index: Tensor<B, 2, Int>,
    pub global_from_compact_gid: Tensor<B, 1, Int>,
    pub xys: Tensor<B, 2>,
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

// TODO: In rust 1.80 having a trait bound here on the inner backend would be great.
// For now all code using it will need to specify this bound itself.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}
impl AutodiffBackend for Autodiff<BurnBack> {}
