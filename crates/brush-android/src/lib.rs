#![cfg(target_os = "android")]

use brush_train::create_wgpu_setup;
use jni::sys::{jint, JNI_VERSION_1_6};
use std::os::raw::c_void;
use std::sync::Arc;

use eframe::egui_wgpu::{WgpuConfiguration, WgpuSetup};
use tokio_with_wasm::alias as tokio;

#[allow(non_snake_case)]
#[no_mangle]
pub extern "system" fn JNI_OnLoad(vm: jni::JavaVM, _: *mut c_void) -> jint {
    let vm_ref = Arc::new(vm);
    rrfd::android::jni_initialize(vm_ref);
    JNI_VERSION_1_6
}

#[no_mangle]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        let setup = create_wgpu_setup().await;

        let wgpu_options = WgpuConfiguration {
            wgpu_setup: WgpuSetup::Existing {
                instance: setup.instance,
                adapter: setup.adapter,
                device: setup.device,
                queue: setup.queue,
            },
            ..Default::default()
        };

        android_logger::init_once(
            android_logger::Config::default().with_max_level(log::LevelFilter::Info),
        );

        eframe::run_native(
            "Brush",
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
    });
}
