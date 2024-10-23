#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

fn main() -> anyhow::Result<()> {
    let wgpu_options = brush_viewer::wgpu_config::get_config();

    #[cfg(not(target_family = "wasm"))]
    {
        env_logger::init();

        let native_options = eframe::NativeOptions {
            // Build app display.
            viewport: egui::ViewportBuilder::default()
                .with_inner_size(egui::Vec2::new(1450.0, 900.0))
                .with_active(true),
            // Need a slightly more careful wgpu init to support burn.
            wgpu_options,
            ..Default::default()
        };

        eframe::run_native(
            "Brush 🖌️",
            native_options,
            Box::new(move |cc| Ok(Box::new(brush_viewer::viewer::Viewer::new(cc)))),
        )
        .unwrap();
    }

    #[cfg(target_family = "wasm")]
    {
        use wasm_bindgen::JsCast;
        eframe::WebLogger::init(log::LevelFilter::Debug).ok();

        let web_options = eframe::WebOptions {
            wgpu_options,
            ..Default::default()
        };

        let document = web_sys::window().unwrap().document().unwrap();
        let canvas = document
            .get_element_by_id("main_canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();

        async_std::task::spawn_local(async {
            eframe::WebRunner::new()
                .start(
                    canvas,
                    web_options,
                    Box::new(|cc| Ok(Box::new(brush_viewer::viewer::Viewer::new(cc)))),
                )
                .await
                .expect("failed to start eframe");
        });
    }

    Ok(())
}
