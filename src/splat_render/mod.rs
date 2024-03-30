use crate::camera::Camera;

pub mod render;
mod gen;
mod kernels;

pub type FloatTensor<B, const D: usize> =
    <B as burn::tensor::backend::Backend>::FloatTensorPrimitive<D>;

pub type IntTensor<B, const D: usize> =
    <B as burn::tensor::backend::Backend>::IntTensorPrimitive<D>;

struct ProjectionOutput<B: Backend> {
    covs3d: FloatTensor<B, 2>,
    xys: FloatTensor<B, 2>,
    depths: FloatTensor<B, 2>,
    radii: FloatTensor<B, 2>,
    conics: FloatTensor<B, 2>,
    compensation: FloatTensor<B, 2>,
    num_tiles_hit: FloatTensor<B, 2>,
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
        means: FloatTensor<Self, 2>,
        scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
        colors: FloatTensor<Self, 2>,
        opacity: FloatTensor<Self, 1>,
        background: glam::Vec3,
    ) -> FloatTensor<Self, 3>;
}

// TODO: In rust 1.80 having a trait bound here on the inner backend would be great.
// For now all code using it will need to specify this bound itself.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}

// /// Implement our custom backend trait for the existing backend `WgpuBackend`.
