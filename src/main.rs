use std::error::Error;
mod camera;
mod dataset_readers;
mod gaussian_splats;
mod renderer;
mod scene;
mod spherical_harmonics;
mod train;
mod utils;

use burn::{
    backend::{
        wgpu::{compute::WgpuRuntime, AutoGraphicsApi},
        Autodiff,
    },
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    tensor::{
        activation::{relu, sigmoid},
        backend::{AutodiffBackend, Backend},
        Data, Tensor,
    },
};

use burn::tensor::ElementConversion;

use image::io::Reader as ImageReader;

fn main() {}
