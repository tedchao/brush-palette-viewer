use anyhow::Result;
use rerun::Image;

use crate::dataset_readers::InputData;

// Encapsulates a multi-view scene including cameras and the splats.
// Also provides methods for checkpointing the training process.
#[derive(Debug)]
pub(crate) struct Scene {
    pub(crate) train_data: Vec<InputData>,
    pub(crate) test_data: Vec<InputData>,
    pub(crate) default_bg_color: glam::Vec3,
}

impl Scene {
    pub(crate) fn new(train_data: Vec<InputData>, test_data: Vec<InputData>) -> Scene {
        Scene {
            train_data,
            test_data,
            default_bg_color: glam::Vec3::ZERO,
        }
    }

    pub(crate) fn visualize(&self, rec: &rerun::RecordingStream) -> Result<()> {
        rec.log_timeless("world", &rerun::ViewCoordinates::RIGHT_HAND_Z_UP)?;

        for (i, data) in self.train_data.iter().enumerate() {
            let path = format!("world/dataset/camera/{i}");
            let vis_size = glam::uvec2(512, 512);
            let rerun_camera = rerun::Pinhole::from_focal_length_and_resolution(
                data.camera.focal(vis_size),
                glam::vec2(vis_size.x as f32, vis_size.y as f32),
            );
            rec.log_timeless(path.clone(), &rerun_camera)?;
            rec.log_timeless(
                path.clone(),
                &rerun::Transform3D::from_translation_rotation(
                    data.camera.position,
                    data.camera.rotation,
                ),
            )?;
            rec.log_timeless(path + "/image", &Image::try_from(data.view.image.clone())?)?;
        }

        Ok(())
    }

    // Returns the extent of the cameras in the scene.
    fn cameras_extent(&self) -> f32 {
        // TODO: This is definitely not as pretty as in numpy.
        let camera_centers = &self
            .train_data
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
}
