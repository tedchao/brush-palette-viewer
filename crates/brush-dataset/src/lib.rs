mod formats;
pub mod scene_batch;
pub mod splat_export;
pub mod splat_import;
pub mod zip;

pub use formats::{load_dataset, load_initial_splat};

use anyhow::Result;
use async_fn_stream::fn_stream;
use async_std::stream::Stream;
use async_std::task::{self, JoinHandle};
use brush_train::scene::{Scene, SceneView};
use glam::Vec3;
use image::DynamicImage;
use std::future::Future;
use std::num::NonZero;
use std::pin::Pin;

#[derive(Clone)]
pub struct LoadDatasetArgs {
    pub max_frames: Option<usize>,
    pub max_resolution: Option<u32>,
    pub eval_split_every: Option<usize>,
}

#[derive(Clone)]
pub struct LoadInitArgs {
    pub sh_degree: u32,
}

#[derive(Clone)]
pub struct Dataset {
    pub train: Scene,
    pub eval: Option<Scene>,
}

impl Dataset {
    pub fn empty() -> Self {
        Dataset {
            train: Scene::new(vec![], Vec3::ZERO),
            eval: None,
        }
    }

    pub fn from_views(
        train_views: Vec<SceneView>,
        eval_views: Vec<SceneView>,
        background: glam::Vec3,
    ) -> Self {
        Dataset {
            train: Scene::new(train_views, background),
            eval: if eval_views.is_empty() {
                None
            } else {
                Some(Scene::new(eval_views.clone(), background))
            },
        }
    }
}

pub(crate) fn clamp_img_to_max_size(image: DynamicImage, max_size: u32) -> DynamicImage {
    if image.width() <= max_size && image.height() <= max_size {
        return image;
    }

    let aspect_ratio = image.width() as f32 / image.height() as f32;
    let (new_width, new_height) = if image.width() > image.height() {
        (max_size, (max_size as f32 / aspect_ratio) as u32)
    } else {
        ((max_size as f32 * aspect_ratio) as u32, max_size)
    };
    image.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
}

pub(crate) type DataStream<T> = Pin<Box<dyn Stream<Item = Result<T>> + Send + 'static>>;

/// Spawn a future (on the async executor on native, as a JS promise on web).
#[cfg(not(target_family = "wasm"))]
mod async_helpers {
    use super::*;

    pub(super) fn spawn_future<T: Send + 'static>(
        future: impl Future<Output = T> + Send + 'static,
    ) -> JoinHandle<T> {
        task::spawn(future)
    }
}

#[cfg(target_family = "wasm")]
mod async_helpers {
    use super::*;

    pub(super) fn spawn_future<T: 'static>(
        future: impl Future<Output = T> + 'static,
    ) -> JoinHandle<T> {
        // On wasm, just spawn locally.
        task::spawn_local(future)
    }
}

pub(crate) use async_helpers::*;

pub(crate) fn stream_fut_parallel<T: Send + 'static>(
    futures: Vec<impl Future<Output = T> + Send + 'static>,
) -> impl Stream<Item = T> {
    let parallel = if cfg!(target_family = "wasm") {
        1
    } else {
        std::thread::available_parallelism()
            .unwrap_or(NonZero::new(8).unwrap())
            .get()
    };

    let mut futures = futures;
    fn_stream(|emitter| async move {
        while !futures.is_empty() {
            // Spawn a batch of threads.
            let handles: Vec<_> = futures
                .drain(..futures.len().min(parallel))
                .map(|fut| spawn_future(fut))
                .collect();
            // Stream each of them.
            for handle in handles {
                emitter.emit(handle.await).await;
            }
        }
    })
}
