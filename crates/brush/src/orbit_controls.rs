use glam::{Mat3, Quat, Vec2, Vec3};

pub struct OrbitControls {
    pub focus: Vec3,
    pub radius: f32,
    pub rotation: glam::Quat,
    pub position: glam::Vec3,
}

impl OrbitControls {
    pub fn new(radius: f32) -> Self {
        Self {
            focus: Vec3::ZERO,
            radius,
            rotation: Quat::IDENTITY,
            position: Vec3::NEG_Z * radius,
        }
    }
}

impl OrbitControls {
    pub fn pan_orbit_camera(&mut self, pan: Vec2, rotate: Vec2, scroll: f32, window: Vec2) {
        let mut any = false;
        if rotate.length_squared() > 0.0 {
            any = true;
            let delta_x = rotate.x * std::f32::consts::PI * 2.0 / window.x;
            let delta_y = rotate.y * std::f32::consts::PI / window.y;
            let yaw = Quat::from_rotation_y(delta_x);
            let pitch = Quat::from_rotation_x(-delta_y);
            self.rotation = yaw * self.rotation * pitch;
        } else if pan.length_squared() > 0.0 {
            any = true;
            // make panning distance independent of resolution and FOV,
            let scaled_pan = pan * Vec2::new(1.0 / window.x, 1.0 / window.y);

            // translate by local axes
            let right = self.rotation * Vec3::X * -scaled_pan.x;
            let up = self.rotation * Vec3::Y * -scaled_pan.y;

            // make panning proportional to distance away from focus point
            let translation = (right + up) * self.radius;
            self.focus += translation;
        } else if scroll.abs() > 0.0 {
            any = true;
            self.radius -= scroll * self.radius * 0.2;
            // dont allow zoom to reach zero or you get stuck
            self.radius = f32::max(self.radius, 0.05);
        }

        if any {
            // emulating parent/child to make the yaw/y-axis rotation behave like a turntable
            // parent = x and y rotation
            // child = z-offset
            let rot_matrix = Mat3::from_quat(self.rotation);
            self.position = self.focus + rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, -self.radius));
        }
    }
}
