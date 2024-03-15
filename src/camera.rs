use ndarray::{s, Array1, Array2};

#[derive(Debug, Default, Clone)]
pub(crate) struct Camera {
    pub znear: f32,
    pub zfar: f32,
    pub width: u32,
    pub height: u32,

    pub fovx: f32,
    pub fovy: f32,
    pub proj_mat: ndarray::Array2<f32>,
    pub transform: ndarray::Array2<f32>,
}

#[derive(Debug, Default)]
pub(crate) struct InputView {
    img_path: String,
    image: Array2<u8>,
    image_mask: Array2<u8>,
}

#[derive(Debug, Default)]
pub(crate) struct InputData {
    camera: Camera,
    view: InputView,
}

impl Camera {
    pub(crate) fn new(
        fovx: f32,
        fovy: f32,
        width: u32,
        height: u32,
        znear: f32,
        zfar: f32,
    ) -> Self {
        Camera {
            znear,
            zfar,
            width,
            height,
            fovx,
            fovy,
            proj_mat: create_projection_matrix(znear, zfar, fovx, fovy),
            transform: ndarray::Array::eye(4),
        }
    }
}

// Constructs a projection matrix from znear, zfar, fovx, fovy.
fn create_projection_matrix(znear: f32, zfar: f32, fovx: f32, fovy: f32) -> Array2<f32> {
    let top = (fovy / 2.0).tan() * znear;
    let bottom = -top;
    let right = (fovx / 2.0).tan() * znear;
    let left = -right;
    let z_sign = 1.0;

    let mut proj_mat = Array2::zeros([4, 4]);
    proj_mat[[0, 0]] = 2.0 * znear / (right - left);
    proj_mat[[1, 1]] = 2.0 * znear / (top - bottom);
    proj_mat[[0, 2]] = (right + left) / (right - left);
    proj_mat[[1, 2]] = (top + bottom) / (top - bottom);
    proj_mat[[3, 2]] = z_sign;
    proj_mat[[2, 2]] = z_sign * zfar / (zfar - znear);
    proj_mat[[2, 3]] = -(zfar * znear) / (zfar - znear);
    proj_mat
}

// Constructs a world to view matrix from rotation and translation.
// TODO: Is this even world2view? This just looks like view to world?
fn world2view_from_rotation_translation(
    rotation: Array2<f32>,
    translation: Array1<f32>,
) -> Array2<f32> {
    let mut rt = Array2::eye(4);
    rt.slice_mut(s![..3, ..3]).assign(&rotation);
    rt.slice_mut(s![..3, 3]).assign(&translation);
    rt
}

// Converts field of view to focal length
fn fov_to_focal(fov: f32, pixels: i32) -> f32 {
    (pixels as f32) / (2.0 * (fov / 2.0).tan())
}

// Converts focal length to field of view.
fn focal_to_fov(focal: f32, pixels: i32) -> f32 {
    2.0 * ((pixels as f32) / (2.0 * focal)).atan()
}
