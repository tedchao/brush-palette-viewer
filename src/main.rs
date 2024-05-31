#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::single_range_in_vec_init)]

mod camera;
mod dataset_readers;
mod gaussian_splats;
mod orbit_controls;
mod scene;
mod splat_import;
mod splat_render;
mod train;
mod utils;
mod viewer;

use std::sync::Arc;

use eframe::{egui_wgpu::WgpuConfiguration, NativeOptions};
#[cfg(feature = "tracy")]
use tracing_subscriber::layer::SubscriberExt;
use viewer::Viewer;

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "tracy")]
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default()),
    )?;

    // Build app display.
    let native_options = NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(egui::Vec2::new(1280.0, 720.0)),
        vsync: false,
        // Need a slightly more careful wgpu init to support burn.
        wgpu_options: WgpuConfiguration {
            device_descriptor: Arc::new(|adapter| wgpu::DeviceDescriptor {
                label: Some("egui+burn wgpu device"),
                required_features: wgpu::Features::default(),
                required_limits: adapter.limits(),
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "Brush 🖌️",
        native_options,
        Box::new(move |cc| Box::new(Viewer::new(cc))),
    )
    .unwrap();

    Ok(())
}
