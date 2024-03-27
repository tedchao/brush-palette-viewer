use crate::camera::Camera;
use burn::tensor::Tensor;

use super::Backend;

pub(crate) struct RenderPackage<B: Backend> {
    // The final rendered image [h, w, 3].
    pub image: Tensor<B, 3>,
    // The radii of each gaussian point.
    // TODO: Do we need this?
    pub radii: Tensor<B, 1>,
    // The sceenspace points.
    // TODO: What even is this?
    pub screenspace_points: Tensor<B, 2>,
}

pub fn render<B: Backend>(
    camera: &Camera,
    means: Tensor<B, 2>,
    scales: Tensor<B, 2>,
    quats: Tensor<B, 2>,
    colors: Tensor<B, 2>,
    opacity: Tensor<B, 1>,
    background: glam::Vec3,
) -> Tensor<B, 3> {
    let img = B::render_gaussians(
        camera,
        means.clone().into_primitive(),
        scales.clone().into_primitive(),
        quats.clone().into_primitive(),
        colors.clone().into_primitive(),
        opacity.clone().into_primitive(),
        background,
    );
    Tensor::from_primitive(img)
}
