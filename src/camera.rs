#[derive(Debug, Default, Clone)]
pub(crate) struct Camera {
    pub fovx: f32,
    pub fovy: f32,
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
}

impl Camera {
    pub fn new(position: glam::Vec3, rotation: glam::Quat, fovx: f32, fovy: f32) -> Self {
        Camera {
            fovx,
            fovy,
            position,
            rotation,
        }
    }

    pub fn focal(&self, img_size: glam::UVec2) -> glam::Vec2 {
        glam::vec2(
            fov_to_focal(self.fovx, img_size.x),
            fov_to_focal(self.fovy, img_size.y),
        )
    }

    pub fn center(&self, img_size: glam::UVec2) -> glam::Vec2 {
        glam::vec2((img_size.x as f32) / 2.0, (img_size.y as f32) / 2.0)
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
