use brush_render::{camera::Camera, Backend};
use burn::tensor::Tensor;
use ndarray::Array3;

#[derive(Debug, Default, Clone)]
pub struct SceneView {
    pub camera: Camera,
    pub image: Array3<f32>,
}

// Encapsulates a multi-view scene including cameras and the splats.
// Also provides methods for checkpointing the training process.
#[derive(Debug)]
pub struct Scene {
    pub(crate) views: Vec<SceneView>,
}

impl Scene {
    pub fn new(views: Vec<SceneView>) -> Self {
        Scene { views }
    }

    // Returns the extent of the cameras in the scene.
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

    pub fn view_count(&self) -> usize {
        self.views.len()
    }
}

#[derive(Clone, Debug)]
pub struct SceneBatch<B: Backend> {
    pub gt_image: Tensor<B, 4>,
    pub cameras: Vec<Camera>,
}
