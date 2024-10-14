use crate::{viewer::ViewerContext, ViewerPane};

#[derive(Default)]
pub(crate) struct TracingPanel {
    constant_redraw: bool,
}

impl ViewerPane for TracingPanel {
    fn title(&self) -> String {
        "Load data".to_owned()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &mut ViewerContext) -> egui_tiles::UiResponse {
        ui.checkbox(&mut self.constant_redraw, "Constant redraw");
        let mut checked = sync_span::is_enabled();
        ui.checkbox(&mut checked, "Sync scopes");
        sync_span::set_enabled(checked);

        // Nb: this redraws the whole context so this will include the splat views.
        if self.constant_redraw {
            ui.ctx().request_repaint();
        }

        egui_tiles::UiResponse::None
    }
}
