use std::sync::{Arc, RwLock};

use brush_render::camera::Camera;

#[derive(Debug, Clone)]
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
    pub background: glam::Vec3,
}

impl Scene {
    pub fn new(views: Vec<SceneView>, background: glam::Vec3) -> Self {
        Scene {
            views: Arc::new(RwLock::new(views)),
            background,
        }
    }

    // Returns the extent of the cameras in the scene.
    // TODO: Cache this?
    pub fn cameras_extent(&self) -> f32 {
        let camera_centers = &self
            .views
            .read()
            .expect("Lock got poisoned somehow")
            .iter()
            .map(|x| x.camera.position)
            .collect::<Vec<_>>();

        let scene_center: glam::Vec3 = camera_centers
            .iter()
            .copied()
            .fold(glam::Vec3::ZERO, |x, y| x + y)
            / (camera_centers.len() as f32);

        camera_centers
            .iter()
            .copied()
            .map(|x| (scene_center - x).length())
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap()
            * 1.1
    }

    pub fn add_view(&mut self, view: SceneView) {
        self.views.write().expect("lock was poisoned").push(view);
    }

    pub fn get_view(&self, index: usize) -> Option<SceneView> {
        self.views
            .read()
            .expect("Poisoned lock")
            .get(index)
            .cloned()
    }

    fn camera_similarity_score(&self, cam: &Camera, reference: &Camera) -> f32 {
        let distance = (cam.position - reference.position).length();
        let forward_ref = reference.rotation * glam::Vec3::Z;
        let forward_cam = cam.rotation * glam::Vec3::Z;
        distance * (1.0 - forward_ref.dot(forward_cam))
    }

    pub fn get_nearest_view(&self, reference: &Camera) -> Option<usize> {
        self.views
            .read()
            .expect("Lock was poisoned somehow.")
            .iter()
            .enumerate() // This will give us (index, view) pairs
            .min_by(|(_, a), (_, b)| {
                let score_a = self.camera_similarity_score(&a.camera, reference);
                let score_b = self.camera_similarity_score(&b.camera, reference);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index) // We return the index instead of the camera
    }

    pub fn view_count(&self) -> usize {
        self.views.read().unwrap().len()
    }
}
