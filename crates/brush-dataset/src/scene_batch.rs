use async_std::channel::Receiver;
use brush_render::Backend;
use brush_train::image::image_to_tensor;
use brush_train::scene::Scene;
use brush_train::train::SceneBatch;
use burn::tensor::Tensor;

use crate::spawn_future;

pub struct SceneLoader<B: Backend> {
    receiver: Receiver<SceneBatch<B>>,
}

impl<B: Backend> SceneLoader<B> {
    pub fn new(scene: &Scene, batch_size: usize, seed: u64, device: &B::Device) -> Self {
        let scene = scene.clone();
        // The bounded size == number of batches to prefetch.
        let (tx, rx) = async_std::channel::bounded(5);
        let device = device.clone();
        let len = scene.views.len() as u64;
        let scene_extent = scene.bounds(0.0, 0.0).extent.length() as f64;

        let fut = async move {
            // Nb: Index works as a "seed" to the dataloader.
            let mut index = seed;

            loop {
                let indices: Vec<_> = (0..batch_size)
                    .map(|i| {
                        miller_shuffle((index + i as u64) % len, (index + i as u64) / len, len)
                    })
                    .collect();
                let gt_views: Vec<_> = indices
                    .iter()
                    .map(|&x| scene.views[x as usize].clone())
                    .collect();
                let selected_tensors: Vec<_> = gt_views
                    .iter()
                    .map(|view| image_to_tensor(&view.image, &device))
                    .collect();

                let batch_tensor = Tensor::stack(selected_tensors, 0);

                let scene_batch = SceneBatch {
                    gt_images: batch_tensor,
                    gt_views,
                    scene_extent,
                };

                if tx.send(scene_batch).await.is_err() {
                    break;
                }

                index = index.wrapping_add(1);
            }
        };

        spawn_future(fut);
        Self { receiver: rx }
    }

    pub async fn next_batch(&mut self) -> SceneBatch<B> {
        self.receiver
            .recv()
            .await
            .expect("Somehow lost data loading channel!")
    }
}

// Simple rust port of https://github.com/RondeSC/Miller_Shuffle_Algo/blob/main/MillerShuffle.c,
// "Miller Shuffle Algorithm E variant".
// Copyright 2022 Ronald R. Miller
// http://www.apache.org/licenses/LICENSE-2.0
fn miller_shuffle(inx: u64, shuffle_id: u64, list_size: u64) -> u64 {
    let p1 = 24317;
    let p2 = 32141;
    let p3 = 63629; // for shuffling 60,000+ indexes

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
