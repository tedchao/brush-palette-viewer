use brush_render::camera::Camera;

#[derive(Debug, Default, Clone)]
pub struct SceneView {
    pub name: String,
    pub camera: Camera,
    pub image: image::DynamicImage,
}

// Encapsulates a multi-view scene including cameras and the splats.
// Also provides methods for checkpointing the training process.
#[derive(Debug, Clone)]
pub struct Scene {
    pub views: Vec<SceneView>,
    pub background_color: glam::Vec3,
}

impl Scene {
    pub fn new(views: Vec<SceneView>, background_color: glam::Vec3) -> Self {
        Scene {
            views,
            background_color,
        }
    }

    // Returns the extent of the cameras in the scene.
    // TODO: Cache this?
    pub fn cameras_extent(&self) -> f32 {
        let camera_centers = &self
            .views
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

    pub fn get_view(&self, index: usize) -> Option<SceneView> {
        self.views.get(index).cloned()
    }

    fn camera_similarity_score(&self, cam: &Camera, reference: &Camera) -> f32 {
        // Normalize distance by the scene extent
        let max_distance = self.cameras_extent();
        let distance = (cam.position - reference.position).length() / max_distance;

        // Calculate orientation similarity
        let forward_ref = reference.rotation * glam::Vec3::Z;
        let forward_cam = cam.rotation * glam::Vec3::Z;

        // Combine distance and orientation with equal weights
        distance * (1.0 - forward_ref.dot(forward_cam))
    }

    pub fn get_nearest_view(&self, reference: &Camera) -> Option<usize> {
        self.views
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
}
