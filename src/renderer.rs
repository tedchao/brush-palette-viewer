use crate::camera::Camera;
use burn::tensor::{backend::Backend, Device, Tensor};
use rerun::external::glam;

pub(crate) struct RenderPackage<B: Backend> {
    pub image: Tensor<B, 3>,
    pub radii: Tensor<B, 1>,
    pub screenspace_points: Tensor<B, 2>,
}

struct GaussianRasterizationSettings {
    image_height: u32,
    image_width: u32,
    tanfovx: f32,
    tanfovy: f32,
    bg: glam::Vec3,
    scale_modifier: f32,
    sh_degree: i32,
    prefiltered: bool,
    viewmatrix: glam::Mat4,
    projmatrix: glam::Mat4,
}

fn forward<B: Backend>(
    means_3d: Tensor<B, 2>,
    _means_2d: Tensor<B, 2>,
    sh: Tensor<B, 3>,
    _opacities: Tensor<B, 1>,
    _scales: Tensor<B, 2>,
    _rotations: Tensor<B, 2>,
    raster_settings: GaussianRasterizationSettings,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 1>) {
    // TODO: Custom cuda rendering :)
    // Restructure arguments the way that the C++ lib expects them
    // let args = (
    //     raster_settings.bg,
    //     means_3d.clone(),
    //     opacities,
    //     scales,
    //     rotations,
    //     raster_settings.scale_modifier,
    //     raster_settings.viewmatrix,
    //     raster_settings.projmatrix,
    //     raster_settings.tanfovx,
    //     raster_settings.tanfovy,
    //     raster_settings.image_height,
    //     raster_settings.image_width,
    //     sh,
    //     raster_settings.sh_degree,
    //     raster_settings.prefiltered,
    // );
    // Invoke C++/CUDA rasterizer
    // num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
    // Keep relevant tensors for backward
    // ctx.raster_settings = raster_settings;
    // ctx.num_rendered = 0;
    // ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer);
    let dims = means_3d.dims();
    let color = sh.reshape([
        raster_settings.image_height as usize,
        raster_settings.image_width as usize,
        3,
    ]) + means_3d.reshape([
        raster_settings.image_height as usize,
        raster_settings.image_width as usize,
        3,
    ]) * 0.1;
    let radii = Tensor::zeros([dims[0]], device);
    (color, radii)
}

// fn backward() {
// let num_rendered = ctx.num_rendered;
// let raster_settings = ctx.raster_settings;
// let (colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer) = ctx.saved_tensors
// Restructure args as C++ method expects them
// let args = (raster_settings.bg,
//         means3D,
//         radii,
//         colors_precomp,
//         scales,
//         rotations,
//         raster_settings.scale_modifier,
//         cov3Ds_precomp,
//         raster_settings.viewmatrix,
//         raster_settings.projmatrix,
//         raster_settings.tanfovx,
//         raster_settings.tanfovy,
//         grad_out_color,
//         sh,
//         raster_settings.sh_degree,
//         raster_settings.campos,
//         geomBuffer,
//         num_rendered,
//         binningBuffer,
//         imgBuffer);
// // Compute gradients for relevant tensors by invoking backward method
// // let grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
// (
//     grad_means3D,
//     grad_means2D,
//     grad_sh,
//     grad_colors_precomp,
//     grad_opacities,
//     grad_scales,
//     grad_rotations,
//     grad_cov3Ds_precomp,
//     None,
// )
// }

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
    camera: &Camera,
    xyz: Tensor<B, 2>,
    shs: Tensor<B, 3>,
    _active_sh_degree: u32,
    opacity: Tensor<B, 1>,
    scale: Tensor<B, 2>,
    rotation: Tensor<B, 2>,
    bg_color: glam::Vec3,
    device: &Device<B>,
) -> RenderPackage<B> {
    // screnspace_points is used as a vessel to carry the viewpsace gradients
    let screenspace_points = Tensor::zeros_like(&xyz);

    let tanfovx = (camera.fovx * 0.5).tan();
    let tanfovy = (camera.fovy * 0.5).tan();
    let active_sh_degree = 1;

    let raster_settings = GaussianRasterizationSettings {
        image_height: camera.height,
        image_width: camera.width,
        tanfovx,
        tanfovy,
        bg: bg_color,
        sh_degree: active_sh_degree,
        viewmatrix: camera.transform,
        projmatrix: camera.proj_mat,
        scale_modifier: 1.0,
        prefiltered: false,
    };

    // let rasterizer = dgr.GaussianRasterizer(raster_settings=raster_settings);
    let (image, radii) = forward(
        xyz.clone(),
        screenspace_points.clone(),
        shs.clone(),
        opacity.clone(),
        scale.clone(),
        rotation.clone(),
        raster_settings,
        device,
    );

    RenderPackage {
        image,
        radii,
        screenspace_points,
    }
}
