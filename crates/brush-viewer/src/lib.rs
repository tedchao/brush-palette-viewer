#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::single_range_in_vec_init)]

mod burn_texture;
mod dataset_readers;
mod gaussian_splats;
mod orbit_controls;
mod scene;
mod splat_import;
mod ssim;
mod train;
mod utils;

#[cfg(feature = "rerun")]
mod visualize;

pub mod viewer;
pub mod wgpu_config;
