use core::f32;

use brush_render::camera::Camera;
use glam::{Affine3A, Quat, Vec2, Vec3};

pub struct OrbitControls {
    pub focus: Vec3,
    pub dirty: bool,
    pub base_transform: Affine3A,

    pan_momentum: Vec2,
    rotate_momentum: Vec2,
}

impl OrbitControls {
    pub fn new(base_transform: Affine3A) -> Self {
        Self {
            focus: Vec3::ZERO,
            pan_momentum: Vec2::ZERO,
            rotate_momentum: Vec2::ZERO,
            dirty: false,
            base_transform,
        }
    }

    pub fn pan_orbit_camera(
        &mut self,
        camera: &mut Camera,
        pan: Vec2,
        rotate: Vec2,
        scroll: f32,
        window: Vec2,
        delta_time: f32,
    ) -> bool {
        let mv = glam::Mat4::from_rotation_translation(camera.rotation, camera.position);
        let v = self.base_transform.inverse() * mv;

        let (_, mut rotation, mut position) = v.to_scale_rotation_translation();

        let mut radius = (position - self.focus).length();
        // Adjust momentum with the new input
        self.pan_momentum += pan;
        self.rotate_momentum += rotate;

        // Apply damping to the momentum
        let damping = 0.0005f32.powf(delta_time);
        self.pan_momentum *= damping;
        self.rotate_momentum *= damping;

        // Update velocities based on momentum
        let pan_velocity = self.pan_momentum * delta_time;
        let rotate_velocity = self.rotate_momentum * delta_time;

        let delta_x = rotate_velocity.x * std::f32::consts::PI * 2.0 / window.x;
        let delta_y = rotate_velocity.y * std::f32::consts::PI / window.y;
        let yaw = Quat::from_rotation_y(delta_x);
        let pitch = Quat::from_rotation_x(-delta_y);
        rotation = yaw * rotation * pitch;

        let scaled_pan = pan_velocity * Vec2::new(1.0 / window.x, 1.0 / window.y);

        let right = rotation * Vec3::X * -scaled_pan.x;
        let up = rotation * Vec3::Y * -scaled_pan.y;

        let translation = (right + up) * radius;
        self.focus += translation;
        radius -= scroll * radius * 0.2;

        let min = 0.25;
        let max = 35.0;
        // smooth clamp to min/max radius.
        if radius < min {
            radius = radius * 0.5 + min * 0.5;
        }

        if radius > max {
            radius = radius * 0.5 + max * 0.5;
        }

        position = self.focus + rotation * Vec3::new(0.0, 0.0, -radius);

        let cam_to_world = glam::Mat4::from_rotation_translation(rotation, position);
        let mv = self.base_transform * cam_to_world;
        let (_, rotation, position) = mv.to_scale_rotation_translation();
        camera.position = position;
        camera.rotation = rotation;

        scroll.abs() > 0.0
            || pan.length_squared() > 0.0
            || rotate.length_squared() > 0.0
            || self.pan_momentum.length_squared() > 0.001
            || self.rotate_momentum.length_squared() > 0.001
            || self.dirty
    }

    pub fn forward(&self) -> Vec3 {
        self.base_transform.transform_vector3(Vec3::Z)
    }
}
