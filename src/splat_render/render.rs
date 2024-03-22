use crate::camera::Camera;
use burn::tensor::Tensor;
use rerun::external::glam;

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
    shs: Tensor<B, 3>,
    _active_sh_degree: u32,
    opacity: Tensor<B, 1>,
    _scale: Tensor<B, 2>,
    _rotation: Tensor<B, 2>,
    _bg_color: glam::Vec3,
    _device: &B::Device,
) -> RenderPackage<B> {
    // screnspace_points is used as a vessel to carry the viewpsace gradients
    let screenspace_points = Tensor::zeros_like(&means);
    let radii = Tensor::zeros_like(&opacity);

    // let tanfovx = (camera.fovx * 0.5).tan();
    // let tanfovy = (camera.fovy * 0.5).tan();
    // let active_sh_degree = 1;

    // let rasterizer = dgr.GaussianRasterizer(raster_settings=raster_settings);
    let output = B::render_splats_2d(
        (camera.height, camera.width),
        means.into_primitive(),
        shs.into_primitive(),
        opacity.into_primitive(),
    );
    let image = Tensor::from_primitive(output);

    RenderPackage {
        image,
        radii,
        screenspace_points,
    }
}
