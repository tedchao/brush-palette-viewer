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
mod splat_import;
mod splat_render;
mod train;
mod utils;
mod viewer;

use tracing_subscriber::layer::SubscriberExt;

fn main() -> Result<(), Box<dyn Error>> {
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default()),
    )
    .expect("Failed to setup tracy layer");

    // type BackGPU = Wgpu<AutoGraphicsApi, f32, i32>;
    // type BackGPU = JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;
    // type DiffBack = Autodiff<BackGPU>;
    // let config = TrainConfig::new("../nerf_synthetic/lego/".to_owned());
    // TODDO: When training with viewer enabled, needs to be existing UI device.. ?
    // let device = Default::default();
    // train::train::<DiffBack>(&config, &device)?;
    viewer::view(
        "../models/bonsai/point_cloud/iteration_30000/point_cloud.ply",
        "../models/bonsai/cameras.json",
    )?;
    Ok(())
}
