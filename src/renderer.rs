use burn::tensor::{backend::Backend, Device, Tensor};

use crate::camera::Camera;

pub(crate) struct RenderPackage<B: Backend> {
    pub image: Tensor<B, 3>,
    pub radii: Tensor<B, 1>,
    pub screenspace_points: Tensor<B, 2>,
}

struct GaussianRasterizationSettings {}

// Renders a 3D image from a set of Gaussian Splats.

// Args:
//   camera: Camera to render.
//   xyz: The means of the Gaussian Splats.
//   shs: The spherical harmonics of the Gaussian Splats or None, if None the
//     rgb_precomputed needs to have values.
//   active_sh_degree: Number of active sh bands.
//   rgb_precomputed: The rgb colors of the splats or None, if None shs needs to
//     be set.
//   opacity: The opacity of the splats.
//   scale: The scales of the splats.
//   rotation: The rotations of the splats as quaternions.
//   cov3d_precomputed: Precomputed values of the covariance matrix or None.
//     Should be defined only if scale and rotation is None.
//   bg_color: The background color.

// Returns:
// A tuple that consists of:
//  * The rendered image.
//  * A tensor that holds the gradients of the Loss wrt the screenspace xyz.
//  * The maximum screenspace radius of each gaussian.
// TODO: Not ideal to depend on device.
pub fn render<B: Backend>(
    camera: Camera,
    xyz: Tensor<B, 2>,
    shs: Tensor<B, 3>,
    active_sh_degree: i32,
    opacity: Tensor<B, 1>,
    scale: Tensor<B, 2>,
    rotation: Tensor<B, 2>,
    bg_color: Tensor<B, 1>,
    device: &Device<B>,
) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 1>) {
    // screnspace_points is used as a vessel to carry the viewpsace gradients
    let screenspace_points = Tensor::zeros_like(&xyz);

    let tanfovx = (camera.fovx * 0.5).tan();
    let tanfovy = (camera.fovy * 0.5).tan();

    // let raster_settings = GaussianRasterizationSettings {
    //     camera.height,
    //     camera.width,
    //     tanfovx,
    //     tanfovy,
    //     bg_color,
    //     active_sh_degree,
    //     viewmatrix=camera.world_view_transform.transpose(0, 1),
    //     projmatrix=camera.full_proj_transform.transpose(0, 1),
    //     camera.camera_center,
    //     False,
    // };

    // let rasterizer = dgr.GaussianRasterizer(raster_settings=raster_settings);

    // let rendered_image, radii = rasterizer(
    //     means3D=xyz,
    //     means2D=screenspace_points,
    //     shs=shs,
    //     opacities=opacity,
    //     scales=scale,
    //     rotations=rotation,
    // );

    let rendered_image = Tensor::zeros([camera.height as usize, camera.width as usize, 3], device);
    let radii = Tensor::zeros([xyz.dims()[0]], device);
    (rendered_image, screenspace_points, radii)
}
