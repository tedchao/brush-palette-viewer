use ndarray::Array3;

#[derive(Debug, Default, Clone)]
pub(crate) struct Camera {
    pub width: u32,
    pub height: u32,
    pub fovx: f32,
    pub fovy: f32,
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
    pub fn new(
        translation: glam::Vec3,
        rotation: glam::Quat,
        fovx: f32,
        fovy: f32,
        width: u32,
        height: u32,
    ) -> Self {
        Camera {
            width,
            height,
            fovx,
            fovy,
            transform: glam::Mat4::from_rotation_translation(rotation, translation),
        }
    }

    pub fn position(&self) -> glam::Vec3 {
        let (_, _, trans) = self.transform.to_scale_rotation_translation();
        trans
    }

    pub fn rotation(&self) -> glam::Quat {
        let (_, rot, _) = self.transform.to_scale_rotation_translation();
        rot
    }

    pub fn focal(&self) -> glam::Vec2 {
        glam::vec2(
            fov_to_focal(self.fovx, self.width),
            fov_to_focal(self.fovy, self.width),
        )
    }

    pub fn center(&self) -> glam::Vec2 {
        glam::vec2((self.width as f32) / 2.0, (self.height as f32) / 2.0)
    }

    pub fn viewmatrix(&self) -> glam::Mat4 {
        self.transform.inverse()
    }
}

// Converts field of view to focal length
pub(crate) fn fov_to_focal(fov: f32, pixels: u32) -> f32 {
    (pixels as f32) / (2.0 * (fov / 2.0).tan())
}

// Converts focal length to field of view.
pub(crate) fn focal_to_fov(focal: f32, pixels: u32) -> f32 {
    2.0 * ((pixels as f32) / (2.0 * focal)).atan()
}
