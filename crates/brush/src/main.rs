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
mod viewer;

use eframe::{egui_wgpu::WgpuConfiguration, NativeOptions};
use std::sync::Arc;
use viewer::Viewer;

#[cfg(feature = "tracy")]
use tracing_subscriber::layer::SubscriberExt;

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "tracy")]
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default()),
    )?;

    let wgpu_options = WgpuConfiguration {
        device_descriptor: Arc::new(|adapter| wgpu::DeviceDescriptor {
            label: Some("egui+burn wgpu device"),
            required_features: wgpu::Features::default(),
            required_limits: adapter.limits(),
        }),
        supported_backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Build app display.
        let native_options = NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size(egui::Vec2::new(1280.0, 720.0))
                .with_active(true),
            vsync: false,
            // Need a slightly more careful wgpu init to support burn.
            wgpu_options,
            ..Default::default()
        };
        eframe::run_native(
            "Brush 🖌️",
            native_options,
            Box::new(move |cc| Box::new(Viewer::new(cc))),
        )
        .unwrap();
    }

    #[cfg(target_arch = "wasm32")]
    {
        let web_options = eframe::WebOptions {
            wgpu_options,
            ..Default::default()
        };

        wasm_bindgen_futures::spawn_local(async {
            eframe::WebRunner::new()
                .start(
                    "main_canvas", // hardcode it
                    web_options,
                    Box::new(|cc| Box::new(Viewer::new(cc))),
                )
                .await
                .expect("failed to start eframe");
        });
    }

    Ok(())
}
