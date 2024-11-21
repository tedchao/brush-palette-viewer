use core::f32;

use glam::{Affine3A, Mat3A, Vec2, Vec3A};

pub struct OrbitControls {
    pub transform: Affine3A,

    pub focus: Vec3A,
    pub dirty: bool,

    pan_momentum: Vec2,
    rotate_momentum: Vec2,
}

impl OrbitControls {
    pub fn new(transform: Affine3A) -> Self {
        Self {
            transform,
            focus: Vec3A::ZERO,
            pan_momentum: Vec2::ZERO,
            rotate_momentum: Vec2::ZERO,
            dirty: false,
        }
    }

    pub fn pan_orbit_camera(
        &mut self,
        pan: Vec2,
        rotate: Vec2,
        scroll: f32,
        window: Vec2,
        delta_time: f32,
    ) -> bool {
        let mut rotation = self.transform.matrix3;
        let mut radius = (self.transform.translation - self.focus).length();
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
        let yaw = Mat3A::from_rotation_y(delta_x);
        let pitch = Mat3A::from_rotation_x(-delta_y);
        rotation = yaw * rotation * pitch;

        let scaled_pan = pan_velocity * Vec2::new(1.0 / window.x, 1.0 / window.y);

        let right = rotation * Vec3A::X * -scaled_pan.x;
        let up = rotation * Vec3A::Y * -scaled_pan.y;

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

        self.transform.translation = self.focus + rotation * Vec3A::new(0.0, 0.0, -radius);
        self.transform.matrix3 = rotation;

        scroll.abs() > 0.0
            || pan.length_squared() > 0.0
            || rotate.length_squared() > 0.0
            || self.pan_momentum.length_squared() > 0.001
            || self.rotate_momentum.length_squared() > 0.001
            || self.dirty
    }
}
