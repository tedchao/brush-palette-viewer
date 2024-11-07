use brush_render::camera::Camera;
use glam::{Mat3, Quat, Vec2, Vec3};

pub struct OrbitControls {
    pub focus: Vec3,
    pan_momentum: Vec2,
    rotate_momentum: Vec2,
}

impl OrbitControls {
    pub fn new() -> Self {
        Self {
            focus: Vec3::ZERO,
            pan_momentum: Vec2::ZERO,
            rotate_momentum: Vec2::ZERO,
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
    ) {
        let mut radius = (camera.position - self.focus).length();
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
        camera.rotation = yaw * camera.rotation * pitch;

        let scaled_pan = pan_velocity * Vec2::new(1.0 / window.x, 1.0 / window.y);

        let right = camera.rotation * Vec3::X * -scaled_pan.x;
        let up = camera.rotation * Vec3::Y * -scaled_pan.y;

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

        let rot_matrix = Mat3::from_quat(camera.rotation);
        camera.position = self.focus + rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, -radius));
    }
}
