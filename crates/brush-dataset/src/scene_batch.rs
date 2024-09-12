use brush_render::camera::Camera;
use brush_render::Backend;
use burn::tensor::{Shape, Tensor};
use ndarray::{ArrayView, Dim, Dimension};

use brush_train::scene::Scene;
use brush_train::scene::SceneBatch;

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

pub struct SceneLoader<B: Backend> {
    images: Vec<Tensor<B, 3>>,
    cameras: Vec<Camera>,
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
    pub fn new(scene: Scene, device: &B::Device, batch_size: usize) -> Self {
        let burn_tensors = scene.views
            .iter()
            .map(|x| ndarray_to_burn::<B, 3>(x.image.view(), device))
            .collect::<Vec<_>>();

        Self {
            images: burn_tensors,
            cameras: scene.views.iter().map(|x| x.camera.clone()).collect(),
            index: 0,
            batch_size,
        }
    }

    pub fn next_batch(&mut self) -> SceneBatch<B> {
        let len = self.images.len();

        let indexes: Vec<_> = (0..self.batch_size)
            .map(|_| {
                let list_index = miller_shuffle(self.index % len, self.index / len, len);
                self.index += 1;
                list_index as i32
            })
            .collect();

        let selected_tensors = indexes.iter().map(|x| self.images[*x as usize].clone()).collect();
        let batch_tensor = Tensor::stack(selected_tensors, 0);

        SceneBatch {
            gt_images: batch_tensor,
            cameras: indexes
                .into_iter()
                .map(|x| self.cameras[x as usize].clone())
                .collect(),
        }
    }
}
