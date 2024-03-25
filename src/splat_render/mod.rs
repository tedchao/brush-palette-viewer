use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::Autodiff;

use crate::camera::Camera;

mod project_gaussians;
pub mod render;
mod render_2d_gaussians;

mod gen;

pub type FloatTensor<B, const D: usize> =
    <B as burn::tensor::backend::Backend>::FloatTensorPrimitive<D>;

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
    fn project_splats(
        cam: &Camera,
        xys: FloatTensor<Self, 2>,
        scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
    ) -> FloatTensor<Self, 2>;
}

// TODO: This kinda needs trait bounds to really work? Oh boy.
pub trait AutodiffBackend:
    Backend + burn::tensor::backend::AutodiffBackend<InnerBackend: Backend>
{
}

impl<B: Backend, C: CheckpointStrategy> AutodiffBackend for Autodiff<B, C> {}

// /// Implement our custom backend trait for the existing backend `WgpuBackend`.

// TODO: backwards.
