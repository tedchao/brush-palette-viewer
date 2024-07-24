#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;

    let wgpu_options = brush_viewer::wgpu_config::get_config();

    // SAFETY: Winit manages this pointer to be valid.
    // let vm = unsafe { jni::JavaVM::from_raw(app.vm_as_ptr() as *mut jni::sys::JavaVM) }.unwrap();

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
        Box::new(|cc| Ok(Box::new(brush_viewer::viewer::Viewer::new(cc)))),
    )
    .unwrap();
}
