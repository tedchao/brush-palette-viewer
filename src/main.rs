#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
mod camera;
mod dataset_readers;
mod gaussian_splats;
mod scene;
mod splat_import;
mod splat_render;
mod train;
mod utils;
mod viewer;

use burn::backend::Autodiff;
#[cfg(feature = "tracy")]
use tracing_subscriber::layer::SubscriberExt;

use crate::{splat_render::BurnBack, train::TrainConfig};

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "tracy")]
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default()),
    )
    .expect("Failed to setup tracy layer");

    type DiffBack = Autodiff<BurnBack>;
    let config = TrainConfig::new("../nerf_synthetic/lego/".to_owned());
    let device = Default::default();

    if true {
        train::train::<DiffBack>(&config, &device)?;
    } else {
        viewer::start()?;
    }
    Ok(())
}
