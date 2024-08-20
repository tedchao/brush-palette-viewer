use brush_render::Backend;
use burn::tensor::{Shape, Tensor};
use ndarray::{ArrayView, Dim, Dimension};

use brush_train::scene::Scene;
use brush_train::scene::SceneBatch;
use brush_train::scene::SceneView;

// Normally the scene batcher is used with a Burn "dataloader" which handles some things
// like shuffling & multithreading, but, we need some more control, and I also can't figure out how
// to make their dataset loader work with lifetimes. Also also, it doesn't work on wasm.
#[derive(Clone)]
pub struct SceneBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> SceneBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

pub(crate) fn ndarray_to_burn<B: Backend, const D: usize>(
    arr: ArrayView<f32, Dim<[usize; D]>>,
    device: &B::Device,
) -> Tensor<B, D>
where
    Dim<[usize; D]>: Dimension,
{
    let shape = Shape::new(arr.shape().try_into().unwrap());
    Tensor::<_, 1>::from_floats(arr.as_slice().unwrap(), device).reshape(shape)
}

impl<B: Backend> SceneBatcher<B> {
    fn batch(&self, items: Vec<SceneView>) -> SceneBatch<B> {
        let burn_tensors = items
            .iter()
            .map(|x| ndarray_to_burn::<B, 3>(x.image.view(), &self.device))
            .collect::<Vec<_>>();

        let img_cat = Tensor::stack(burn_tensors, 0);

        SceneBatch {
            gt_image: img_cat,
            cameras: items.iter().map(|x| x.camera.clone()).collect(),
        }
    }
}

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
                .filter_map(|x| scene.get_view(x))
                .collect(),
        );

        Self {
            total_batch,
            index: 0,
            batch_size,
        }
    }

    pub fn next_batch(&mut self) -> SceneBatch<B> {
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
