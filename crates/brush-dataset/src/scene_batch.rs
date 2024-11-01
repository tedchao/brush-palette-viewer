use async_std::channel::Receiver;
use brush_render::Backend;
use brush_train::image::image_to_tensor;
use brush_train::scene::Scene;
use brush_train::train::SceneBatch;
use burn::tensor::Tensor;
use rand::{Rng, SeedableRng};

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
        let scene_extent = scene.bounds(0.0, 0.0).extent.length() as f64;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let fut = async move {
            // Nb: Index works as a "seed" to the dataloader.
            let mut index = seed;

            loop {
                let indices: Vec<_> = (0..batch_size)
                    .map(|_| rng.gen_range(0..scene.views.len()))
                    .collect();
                let gt_views: Vec<_> = indices.iter().map(|&x| scene.views[x].clone()).collect();
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
