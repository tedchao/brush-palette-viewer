#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![feature(iter_array_chunks)]
#![feature(let_chains)]

use std::error::Error;
mod camera;
mod dataset_readers;
mod gaussian_splats;
mod scene;
mod spherical_harmonics;
mod splat_render;
mod train;
mod utils;
mod viewer;

use burn::backend::wgpu::{AutoGraphicsApi, JitBackend, WgpuRuntime};
use tracing_subscriber::layer::SubscriberExt;

fn main() -> Result<(), Box<dyn Error>> {
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default()),
    )
    .expect("setup tracy layer");

    // type BackGPU = Wgpu<AutoGraphicsApi, f32, i32>;
    type BackGPU = JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;
    // type DiffBack = Autodiff<BackGPU>;
    // let config = TrainConfig::new("../nerf_synthetic/lego/".to_owned());
    // train::train::<DiffBack>(&config, &device)?;
    viewer::view(
        "../models/bonsai/point_cloud/iteration_30000/point_cloud.ply",
        "../models/bonsai/cameras.json",
    )?;
    Ok(())
}
