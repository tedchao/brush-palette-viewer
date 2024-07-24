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
mod wgpu_config;

#[no_mangle]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;

    let wgpu_options = wgpu_config::get_config();

    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );

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
