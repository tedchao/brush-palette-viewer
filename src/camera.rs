use ndarray::Array3;

#[derive(Debug, Default, Clone)]
pub(crate) struct Camera {
    pub width: u32,
    pub height: u32,
    pub fovx: f32,
    pub fovy: f32,
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
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
        position: glam::Vec3,
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
            position,
            rotation,
        }
    }

    pub fn focal(&self) -> glam::Vec2 {
        glam::vec2(
            fov_to_focal(self.fovx, self.width),
            fov_to_focal(self.fovy, self.height),
        )
    }

    pub fn center(&self) -> glam::Vec2 {
        glam::vec2((self.width as f32) / 2.0, (self.height as f32) / 2.0)
    }

    pub fn local_to_world(&self) -> glam::Mat4 {
        glam::Mat4::from_rotation_translation(self.rotation, self.position)
    }

    pub fn world_to_local(&self) -> glam::Mat4 {
        self.local_to_world().inverse()
    }

    pub(crate) fn forward(&self) -> glam::Vec3 {
        self.rotation * glam::vec3(0.0, 0.0, 1.0)
    }

    pub(crate) fn right(&self) -> glam::Vec3 {
        self.rotation * glam::vec3(1.0, 0.0, 0.0)
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
