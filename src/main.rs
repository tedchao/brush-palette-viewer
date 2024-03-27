#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![feature(iter_array_chunks)]
#![feature(let_chains)]

use std::error::Error;
mod camera;
mod dataset_readers;
mod gaussian_splats;
mod loss_utils;
mod scene;
mod spherical_harmonics;
mod splat_render;
mod train;
mod utils;

use burn::backend::{
    wgpu::{AutoGraphicsApi, JitBackend, WgpuRuntime},
    Autodiff,
};

use train::TrainConfig;

fn main() -> Result<(), Box<dyn Error>> {
    let device = Default::default();

    // type BackGPU = Wgpu<AutoGraphicsApi, f32, i32>;
    type BackGPU = JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;
    type DiffBack = Autodiff<BackGPU>;

    let config = TrainConfig::new("../nerf_synthetic/lego/".to_owned()).with_train_steps(1);
    train::train::<DiffBack>(&config, &device)?;
    Ok(())
}
