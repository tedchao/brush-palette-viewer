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
    _shs: Tensor<B, 3>,
    _active_sh_degree: u32,
    _opacity: Tensor<B, 1>,
    _bg_color: glam::Vec3,
    _device: &B::Device,
) -> RenderPackage<B> {
    // screnspace_points is used as a vessel to carry the viewpsace gradients

    // let tanfovx = (camera.fovx * 0.5).tan();
    // let tanfovy = (camera.fovy * 0.5).tan();
    // let active_sh_degree = 1;

    let screenspace_points = B::project_splats(
        camera,
        means.clone().into_primitive(),
        scales.clone().into_primitive(),
        quats.clone().into_primitive(),
    );

    let image =
        means
            .clone()
            .unsqueeze::<3>()
            .reshape([camera.height as usize, camera.width as usize, 3]);

    let radii = Tensor::zeros_like(&means).unsqueeze();
    RenderPackage {
        image,
        radii,
        screenspace_points: Tensor::from_primitive(screenspace_points),
    }
}
