use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    tensor::Tensor,
};

use crate::{camera::Camera, dataset_readers::InputView, utils};

// Encapsulates a multi-view scene including cameras and the splats.
// Also provides methods for checkpointing the training process.
#[derive(Debug)]
pub(crate) struct Scene {
    pub(crate) views: Vec<InputView>,
}

impl Scene {
    pub(crate) fn new(views: Vec<InputView>) -> Self {
        Scene { views }
    }

    #[cfg(feature = "rerun")]
    pub(crate) fn visualize(&self, rec: &rerun::RecordingStream) -> anyhow::Result<()> {
        rec.log_static("world", &rerun::ViewCoordinates::RIGHT_HAND_Z_UP)?;

        for (i, data) in self.views.iter().enumerate() {
            let path = format!("world/dataset/camera/{i}");
            let (width, height, _) = data.image.dim();
            let vis_size = glam::uvec2(width as u32, height as u32);
            let rerun_camera = rerun::Pinhole::from_focal_length_and_resolution(
                data.camera.focal(vis_size),
                glam::vec2(vis_size.x as f32, vis_size.y as f32),
            );
            rec.log_static(path.clone(), &rerun_camera)?;
            rec.log_static(
                path.clone(),
                &rerun::Transform3D::from_translation_rotation(
                    data.camera.position,
                    data.camera.rotation,
                ),
            )?;
            rec.log_static(
                path + "/image",
                &rerun::Image::try_from(data.image.clone())?,
            )?;
        }

        Ok(())
    }

    // Returns the extent of the cameras in the scene.
    fn cameras_extent(&self) -> f32 {
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
}

impl Dataset<InputView> for Scene {
    fn get(&self, index: usize) -> Option<InputView> {
        self.views.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.views.len()
    }
}

#[derive(Clone)]
pub struct SceneBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> SceneBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct SceneBatch<B: Backend> {
    pub gt_image: Tensor<B, 3>,
    pub camera: Camera,
}

impl<B: Backend> Batcher<InputView, SceneBatch<B>> for SceneBatcher<B> {
    fn batch(&self, items: Vec<InputView>) -> SceneBatch<B> {
        let item = &items[0]; // ATM we just support having one view.
        let gt_image: Tensor<B, 3> = utils::ndarray_to_burn(item.image.view(), &self.device);
        SceneBatch {
            gt_image,
            camera: item.camera.clone(),
        }
    }
}

// Normally the scene batcher is used with a Burn "dataloader" which handles some things
// like shuffling & multithreading, but, we need some more control, and I also can't figure out how
// to make their dataset loader work with lifetimes.
pub struct SceneLoader<B: Backend> {
    dataset: Scene,
    batcher: SceneBatcher<B>,
    index: usize,
}

fn miller_shuffle(inx: usize, shuffle_id: usize, list_size: usize) -> usize {
    let p1: usize = 24317;
    let p2: usize = 32141;
    let p3: usize = 63629; // for shuffling 60,000+ indexes

    let shuffle_id = shuffle_id + 131 * (inx / list_size); // have inx overflow effect the mix
    let mut si = (inx + shuffle_id) % list_size; // cut the deck
    let r1 = shuffle_id % p1 + 42; // randomizing factors crafted empirically (by automated trial and error)
    let r2 = ((shuffle_id * 0x89) ^ r1) % p2;
    let r3 = (r1 + r2 + p3) % list_size;
    let r4 = r1 ^ r2 ^ r3;
    let rx = (shuffle_id / list_size) % list_size + 1;
    let rx2 = (shuffle_id / list_size / list_size) % list_size + 1;
    // perform conditional multi-faceted mathematical spin-mixing (on avg 2 1/3 shuffle ops done + 2 simple Xors)
    if si % 3 == 0 {
        si = (((si / 3) * p1 + r1) % ((list_size + 2) / 3)) * 3;
    } // spin multiples of 3
    if si % 2 == 0 {
        si = (((si / 2) * p2 + r2) % ((list_size + 1) / 2)) * 2;
    } // spin multiples of 2
    if si < list_size / 2 {
        si = (si * p3 + r4) % (list_size / 2);
    }
    if (si ^ rx) < list_size {
        si ^= rx;
    } // flip some bits with Xor
    si = (si * p3 + r3) % list_size; // relatively prime gears turning operation
    if (si ^ rx2) < list_size {
        si ^= rx2;
    }
    si
}

impl<B: Backend> SceneLoader<B> {
    pub fn new(dataset: Scene, batcher: SceneBatcher<B>) -> Self {
        Self {
            dataset,
            batcher,
            index: 0,
        }
    }

    pub fn next(&mut self) -> Option<SceneBatch<B>> {
        let list_index = miller_shuffle(
            self.index % self.dataset.len(),
            self.index / self.dataset.len(),
            self.dataset.len(),
        );

        self.index += 1;

        self.dataset
            .get(list_index)
            .map(|x| self.batcher.batch(vec![x.clone()]))
    }
}
