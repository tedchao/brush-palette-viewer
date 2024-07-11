use glam::{Mat3, Quat, Vec2, Vec3};

pub struct OrbitControls {
    pub focus: Vec3,
    pub radius: f32,
    pub rotation: Quat,
    pub position: Vec3,
    pan_momentum: Vec2,
    rotate_momentum: Vec2,
    scroll_momentum: f32,
}

impl OrbitControls {
    pub fn new(radius: f32) -> Self {
        Self {
            focus: Vec3::ZERO,
            radius,
            rotation: Quat::IDENTITY,
            position: Vec3::NEG_Z * radius,
            pan_momentum: Vec2::ZERO,
            rotate_momentum: Vec2::ZERO,
            scroll_momentum: 0.0,
        }
    }

    pub fn pan_orbit_camera(&mut self, pan: Vec2, rotate: Vec2, scroll: f32, window: Vec2) {
        // Adjust momentum with the new input
        self.pan_momentum += pan * 5.0;
        self.rotate_momentum += rotate * 5.0;
        self.scroll_momentum += scroll * 5.0;

        // Should use an actual delta time but this is fine for now.
        let delta_time = 1.0 / 60.0;

        // Apply damping to the momentum
        let damping = 0.01f32.powf(delta_time);
        self.pan_momentum *= damping;
        self.rotate_momentum *= damping;
        self.scroll_momentum *= damping;

        // Update velocities based on momentum
        let pan_velocity = self.pan_momentum * delta_time;
        let rotate_velocity = self.rotate_momentum * delta_time;
        let scroll_velocity = self.scroll_momentum * delta_time;

        if rotate_velocity.length_squared() > 0.0 {
            let delta_x = rotate_velocity.x * std::f32::consts::PI * 2.0 / window.x;
            let delta_y = rotate_velocity.y * std::f32::consts::PI / window.y;
            let yaw = Quat::from_rotation_y(delta_x);
            let pitch = Quat::from_rotation_x(-delta_y);
            self.rotation = yaw * self.rotation * pitch;
        }

        if pan_velocity.length_squared() > 0.0 {
            let scaled_pan = pan_velocity * Vec2::new(1.0 / window.x, 1.0 / window.y);

            let right = self.rotation * Vec3::X * -scaled_pan.x;
            let up = self.rotation * Vec3::Y * -scaled_pan.y;

            let translation = (right + up) * self.radius;
            self.focus += translation;
        }

        if scroll_velocity.abs() > 0.0 {
            self.radius -= scroll_velocity * self.radius * 0.2;
            self.radius = f32::max(self.radius, 0.05);
        }

        let rot_matrix = Mat3::from_quat(self.rotation);
        self.position = self.focus + rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, -self.radius));
    }

    pub fn is_animating(&self) -> bool {
        self.pan_momentum.length_squared() > 0.0
            || self.rotate_momentum.length_squared() > 0.0
            || self.scroll_momentum.abs() > 0.0
    }
}
