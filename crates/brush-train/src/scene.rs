use std::sync::{Arc, RwLock, RwLockReadGuard};

use brush_render::{bounding_box::BoundingBox, camera::Camera};
use glam::Vec3;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ViewType {
    Train,
    Eval,
    Test,
}

#[derive(Debug, Clone)]
pub struct SceneView {
    pub name: String,
    pub camera: Camera,
    pub image: image::DynamicImage,
}

// Encapsulates a multi-view scene including cameras and the splats.
// Also provides methods for checkpointing the training process.
#[derive(Debug, Clone)]
pub struct Scene {
    views: Arc<RwLock<Vec<SceneView>>>,
    pub background: Vec3,
}

fn camera_similarity_score(cam: &Camera, reference: &Camera) -> f32 {
    let distance = (cam.position - reference.position).length();
    let forward_ref = reference.rotation * Vec3::Z;
    let forward_cam = cam.rotation * Vec3::Z;
    distance * (1.0 - forward_ref.dot(forward_cam))
}

impl Scene {
    pub fn new(views: Vec<SceneView>, background: Vec3) -> Self {
        Scene {
            views: Arc::new(RwLock::new(views)),
            background,
        }
    }

    // Returns the extent of the cameras in the scene.
    pub fn bounds(&self, cam_far: f32) -> BoundingBox {
        let views = self.views.read().expect("Lock got poisoned somehow");
        let (min, max) = views.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(min, max), view| {
                let cam = &view.camera;
                let pos1 = cam.position;
                let pos2 = cam.position + cam.rotation * Vec3::Z * cam_far;
                (min.min(pos1).min(pos2), max.max(pos1).max(pos2))
            },
        );
        BoundingBox::from_min_max(min, max)
    }

    pub fn center_cameras(&mut self) {
        let scene_center = self.bounds(0.0).center;
        let mut views = self.views.write().unwrap();
        // Adjust camera positions
        for view in views.iter_mut() {
            view.camera.position -= scene_center;
        }
    }

    pub fn add_view(&mut self, view: SceneView) {
        self.views.write().expect("lock was poisoned").push(view);
    }

    pub fn views(&self) -> RwLockReadGuard<'_, Vec<SceneView>> {
        self.views.read().expect("Poisoned lock")
    }

    pub fn get_nearest_view(&self, reference: &Camera) -> Option<usize> {
        self.views
            .read()
            .expect("Lock was poisoned somehow.")
            .iter()
            .enumerate() // This will give us (index, view) pairs
            .min_by(|(_, a), (_, b)| {
                let score_a = camera_similarity_score(&a.camera, reference);
                let score_b = camera_similarity_score(&b.camera, reference);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index) // We return the index instead of the camera
    }
}
