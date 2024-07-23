#![cfg(target_os = "android")]

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

#[no_mangle]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    use eframe::{egui, egui_wgpu::WgpuConfiguration};
    use std::sync::Arc;
    use winit::platform::android::EventLoopBuilderExtAndroid;

    let wgpu_options = WgpuConfiguration {
        device_descriptor: Arc::new(|adapter| wgpu::DeviceDescriptor {
            label: Some("egui+burn wgpu device"),
            required_features: wgpu::Features::default(),
            required_limits: adapter.limits(),
        }),
        supported_backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    };

    eframe::run_native(
        "Brush 🖌️",
        eframe::NativeOptions {
            event_loop_builder: Some(Box::new(|builder| {
                builder.with_android_app(app);
            })),
            wgpu_options,
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(viewer::Viewer::new(cc)))),
    )
    .unwrap();
}
