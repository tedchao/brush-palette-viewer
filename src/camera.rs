use ndarray::Array3;

#[derive(Debug, Default, Clone)]
pub(crate) struct Camera {
    pub width: u32,
    pub height: u32,

    pub fovx: f32,
    pub fovy: f32,

    pub proj_mat: glam::Mat4,
    pub transform: glam::Mat4,
}

#[derive(Debug, Default)]
pub(crate) struct InputView {
    pub(crate) image: Array3<f32>, // RGBA image.
}

#[derive(Debug, Default)]
pub(crate) struct InputData {
    pub(crate) camera: Camera,
    pub(crate) view: InputView,
}

impl Camera {
    pub(crate) fn new(
        translation: glam::Vec3,
        rotation: glam::Quat,
        fovx: f32,
        fovy: f32,
        width: u32,
        height: u32,
        znear: f32,
        zfar: f32,
    ) -> Self {
        Camera {
            width,
            height,
            fovx,
            fovy,
            proj_mat: create_projection_matrix(znear, zfar, fovx, fovy),
            transform: glam::Mat4::from_rotation_translation(rotation, translation),
        }
    }

    pub(crate) fn position(&self) -> glam::Vec3 {
        let (_, _, trans) = self.transform.to_scale_rotation_translation();
        trans
    }

    pub(crate) fn rotation(&self) -> glam::Quat {
        let (_, rot, _) = self.transform.to_scale_rotation_translation();
        rot
    }

    pub(crate) fn focal(&self) -> glam::Vec2 {
        glam::vec2(
            fov_to_focal(self.fovx, self.width),
            fov_to_focal(self.fovy, self.width),
        )
    }

    pub(crate) fn intrins(&self) -> glam::Vec4 {
        // TODO: Does this need the tan business?
        glam::vec4(
            self.fovx,
            self.fovy,
            (self.width as f32) / 2.0,
            (self.height as f32) / 2.0,
        )
    }
}

// Constructs a projection matrix from znear, zfar, fovx, fovy.
fn create_projection_matrix(znear: f32, zfar: f32, fovx: f32, fovy: f32) -> glam::Mat4 {
    let top = (fovy / 2.0).tan() * znear;
    let bottom = -top;
    let right = (fovx / 2.0).tan() * znear;
    let left = -right;
    let z_sign = 1.0;

    glam::Mat4::from_cols_array_2d(&[
        [2.0 * znear / (right - left), 0.0, 0.0, 0.0],
        [0.0, 2.0 * znear / (top - bottom), 0.0, 0.0],
        [
            (right + left) / (right - left),
            (top + bottom) / (top - bottom),
            z_sign * zfar / (zfar - znear),
            z_sign,
        ],
        [0.0, 0.0, -(zfar * znear) / (zfar - znear), 0.0],
    ])
}

// Converts field of view to focal length
pub(crate) fn fov_to_focal(fov: f32, pixels: u32) -> f32 {
    (pixels as f32) / (2.0 * (fov / 2.0).tan())
}

// Converts focal length to field of view.
pub(crate) fn focal_to_fov(focal: f32, pixels: u32) -> f32 {
    2.0 * ((pixels as f32) / (2.0 * focal)).atan()
}
