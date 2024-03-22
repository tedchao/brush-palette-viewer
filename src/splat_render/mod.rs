mod project_gaussians;
mod render;
mod render_2d_gaussians;
use burn::{
    backend::wgpu::{FloatElement, GraphicsApi, IntElement, JitBackend, WgpuRuntime},
    tensor::Tensor,
};

/// We use a type alias for better readability.
pub type FloatTensor<B, const D: usize> =
    <B as burn::tensor::backend::Backend>::FloatTensorPrimitive<D>;

/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend {
    // Project 3D gaussians to screenspace in 2D.
    fn project_gaussians();

    // Render splats
    fn render_splats_2d(
        resolution: (u32, u32),
        means: FloatTensor<Self, 2>,
        shs: FloatTensor<Self, 3>,
        opacity: FloatTensor<Self, 1>,
    ) -> FloatTensor<Self, 3>;
}

// TODO: This kinda needs trait bounds to really work? Oh boy.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}

// /// Implement our custom backend trait for the existing backend `WgpuBackend`.
impl<G: GraphicsApi, F: FloatElement, I: IntElement> Backend for JitBackend<WgpuRuntime<G, F, I>> {
    fn project_gaussians() {
        todo!();
    }

    fn render_splats_2d(
        resolution: (u32, u32),
        means: FloatTensor<Self, 2>,
        shs: FloatTensor<Self, 3>,
        opacity: FloatTensor<Self, 1>,
    ) -> FloatTensor<Self, 3> {
        render_2d_gaussians::forward(resolution, means, shs, opacity)
    }
}
