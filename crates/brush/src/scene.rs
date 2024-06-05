use brush_render::camera::Camera;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    tensor::Tensor,
};

use crate::{dataset_readers::InputView, utils};

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
    pub gt_image: Tensor<B, 4>,
    pub cameras: Vec<Camera>,
}

impl<B: Backend> Batcher<InputView, SceneBatch<B>> for SceneBatcher<B> {
    fn batch(&self, items: Vec<InputView>) -> SceneBatch<B> {
        let burn_tensors = items
            .iter()
            .map(|x| utils::ndarray_to_burn::<B, 3>(x.image.view(), &self.device))
            .collect::<Vec<_>>();

        let img_cat = Tensor::stack(burn_tensors, 0);

        SceneBatch {
            gt_image: img_cat,
            cameras: items.iter().map(|x| x.camera.clone()).collect(),
        }
    }
}

// Normally the scene batcher is used with a Burn "dataloader" which handles some things
// like shuffling & multithreading, but, we need some more control, and I also can't figure out how
// to make their dataset loader work with lifetimes.
pub struct SceneLoader<B: Backend> {
    total_batch: SceneBatch<B>,
    index: usize,
    batch_size: usize,
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
    pub fn new(scene: Scene, batcher: SceneBatcher<B>, batch_size: usize) -> Self {
        let total_batch = batcher.batch(
            (0..scene.views.len())
                .filter_map(|x| scene.get(x))
                .collect(),
        );

        Self {
            total_batch,
            index: 0,
            batch_size,
        }
    }

    pub fn next(&mut self) -> SceneBatch<B> {
        let len = self.total_batch.gt_image.dims()[0];

        let indexes: Vec<_> = (0..self.batch_size)
            .map(|_| {
                let list_index = miller_shuffle(self.index % len, self.index / len, len);
                self.index += 1;
                list_index as i32
            })
            .collect();

        let index_tensor =
            Tensor::from_ints(indexes.as_slice(), &self.total_batch.gt_image.device());

        SceneBatch {
            gt_image: self.total_batch.gt_image.clone().select(0, index_tensor),
            cameras: indexes
                .into_iter()
                .map(|x| self.total_batch.cameras[x as usize].clone())
                .collect(),
        }
    }
}
