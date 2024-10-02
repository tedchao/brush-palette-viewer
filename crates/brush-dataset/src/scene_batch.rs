use crate::scene::Scene;
use brush_render::Backend;
use brush_train::train::SceneBatch;
use burn::tensor::{Tensor, TensorData};
use image::DynamicImage;
use image::GenericImageView;

use std::sync::mpsc::sync_channel;
use std::sync::mpsc::Receiver;
use std::thread;

fn image_to_tensor<B: Backend>(
    image: &DynamicImage,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 3>> {
    let (w, h) = image.dimensions();
    let num_channels = image.color().channel_count();
    let data = image.to_rgb32f().into_vec();
    let tensor_data = TensorData::new(data, [h as usize, w as usize, num_channels as usize]);
    Ok(Tensor::from_data(tensor_data, device))
}

pub struct SceneLoader<B: Backend> {
    receiver: Receiver<SceneBatch<B>>,
}

impl<B: Backend> SceneLoader<B> {
    pub fn new(scene: Scene, batch_size: usize, device: &B::Device) -> Self {
        // Bound == number of batches to prefix.
        let (tx, rx) = sync_channel(5);

        let len = scene.views.len();
        let views = scene.views.clone();

        let device = device.clone();

        // TODO: On wasm, don't thread but async.
        thread::spawn(move || {
            let mut index = 0;

            loop {
                let indexes: Vec<_> = (0..batch_size)
                    .map(|i| {
                        let list_index = miller_shuffle((index + i) % len, (index + i) / len, len);
                        list_index as i32
                    })
                    .collect();
                let cameras = indexes
                    .iter()
                    .map(|&x| views[x as usize].camera.clone())
                    .collect();
                let selected_tensors = indexes
                    .iter()
                    .map(|&x| {
                        image_to_tensor(&views[x as usize].image, &device)
                            .expect("Failed to upload img")
                    })
                    .collect::<Vec<_>>();

                let batch_tensor = Tensor::stack(selected_tensors, 0);

                let scene_batch = SceneBatch {
                    gt_images: batch_tensor,
                    cameras,
                };

                if tx.send(scene_batch).is_err() {
                    break;
                }

                index += 1;
            }
        });

        Self { receiver: rx }
    }

    pub async fn next_batch(&mut self) -> SceneBatch<B> {
        self.receiver.recv().unwrap()
    }
}

// Simple rust port of https://github.com/RondeSC/Miller_Shuffle_Algo/blob/main/MillerShuffle.c,
// "Miller Shuffle Algorithm E variant".
// Copyright 2022 Ronald R. Miller
// http://www.apache.org/licenses/LICENSE-2.0
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
