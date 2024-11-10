pub mod burn_texture;

pub fn draw_checkerboard(ui: &mut egui::Ui, rect: egui::Rect) {
    let id = egui::Id::new("checkerboard");
    let handle = ui
        .ctx()
        .data(|data| data.get_temp::<egui::TextureHandle>(id));

    let handle = if let Some(handle) = handle {
        handle
    } else {
        let color_1 = [190, 190, 190, 255];
        let color_2 = [240, 240, 240, 255];

        let pixels = vec![color_1, color_2, color_2, color_1]
            .into_iter()
            .flatten()
            .collect::<Vec<u8>>();

        let texture_options = egui::TextureOptions {
            magnification: egui::TextureFilter::Nearest,
            minification: egui::TextureFilter::Nearest,
            wrap_mode: egui::TextureWrapMode::Repeat,
            mipmap_mode: None,
        };

        let tex_data = egui::ColorImage::from_rgba_unmultiplied([2, 2], &pixels);

        let handle = ui.ctx().load_texture("checker", tex_data, texture_options);
        ui.ctx().data_mut(|data| {
            data.insert_temp(id, handle.clone());
        });
        handle
    };

    let uv = egui::Rect::from_min_max(
        egui::pos2(0.0, 0.0),
        egui::pos2(rect.width() / 24.0, rect.height() / 24.0),
    );

    ui.painter()
        .image(handle.id(), rect, uv, egui::Color32::WHITE);
}
