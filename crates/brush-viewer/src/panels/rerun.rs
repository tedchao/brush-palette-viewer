use crate::{viewer::ViewerContext, ViewerPane};

#[derive(Default)]
pub(crate) struct RerunPanel {
    log_splat_ellipsoids: bool,
}

impl ViewerPane for RerunPanel {
    fn title(&self) -> String {
        "Rerun".to_owned()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &mut ViewerContext) -> egui_tiles::UiResponse {
        ui.toggle_value(&mut self.log_splat_ellipsoids, "Log splat ellipsoids.");
        egui_tiles::UiResponse::None
    }
}
