use crate::{
    camera::{self, Camera, InputData, InputView},
    gaussian_splats,
};
use burn::module::Module;
use ndarray::{Array1, Array2};

// Encapsulates a multi-view scene including cameras and the splats.
// Also provides methods for checkpointing the training process.
#[derive(Debug)]
pub(crate) struct Scene {
    train_cameras: Vec<InputData>,
    test_cameras: Vec<InputData>,
    default_bg_color: Array1<f32>,
}

impl Scene {
    pub(crate) fn new(model_path: &str) -> Scene {
        Scene {
            train_cameras: vec![],
            test_cameras: vec![],
            default_bg_color: Array1::zeros(3),
        }
    }

    pub(crate) fn initialize(
        &mut self,
        train_cameras: Vec<InputData>,
        test_cameras: Vec<InputData>,
    ) {
        self.train_cameras = train_cameras;
        self.test_cameras = test_cameras;
    }

    /// Performs a single optimizer step for all necessary parameters.
    // fn optimizer_step(&mut self) {
    //     // TODO: Port to Burn.
    //     // self.gaussian_splats.optimizer.step()
    //     // self.gaussian_splats.optimizer.zero_grad(set_to_none=True)

    //      // TODO: Is thsi optimising the cameras or something:?
    //     for cam in self.train_cameras {
    //         cam.optimizer_step()
    //     }
    // }

    // Returns the extent of the cameras in the scene.
    // TODO: Math this out.
    fn cameras_extent(&self) -> f32 {
        0.0
        // camera_centers = torch.stack(
        //     [cam.camera_center for cam in self.train_cameras]
        // );

        // scene_center = camera_centers.mean(dim=0, keepdims=True);
        // cam_to_center = (camera_centers - scene_center).norm(dim=1);
        // max_distance = torch.max(cam_to_center) * 1.1;
        // return max_distance.item();
    }
}
