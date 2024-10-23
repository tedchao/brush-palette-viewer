use crate::{viewer::ViewerContext, ViewerPanel};
use egui::Hyperlink;

pub(crate) struct PresetsPanel {}

impl PresetsPanel {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl PresetsPanel {
    fn url_button(&mut self, label: &str, url: &str, ui: &mut egui::Ui) {
        ui.add(Hyperlink::from_label_and_url(label, url).open_in_new_tab(true));
    }
}

impl ViewerPanel for PresetsPanel {
    fn title(&self) -> String {
        "Presets".to_owned()
    }

    fn on_message(&mut self, _: crate::viewer::ViewerMessage, _: &mut ViewerContext) {}

    fn ui(&mut self, ui: &mut egui::Ui, _: &mut ViewerContext) {
        ui.heading("Mipnerf scenes");

        egui::Grid::new("mip_grid")
            .num_columns(3)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                self.url_button("bicylce", "", ui);
                self.url_button("bonsai", "", ui);
                self.url_button("counter", "", ui);
                ui.end_row();

                self.url_button("garden", "", ui);
                self.url_button("kitchen", "", ui);
                self.url_button("room", "", ui);
                ui.end_row();

                self.url_button("stump", "", ui);
                ui.end_row();
            });

        ui.heading("Synthetic blender scenes");
        egui::Grid::new("blend_grid")
            .num_columns(4)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                self.url_button("chair", "", ui);
                self.url_button("drums", "", ui);
                self.url_button("ficus", "", ui);
                ui.end_row();

                self.url_button("hotdog", "", ui);
                self.url_button("lego", "", ui);
                self.url_button("materials", "", ui);
                ui.end_row();

                self.url_button("mic", "", ui);
                self.url_button("ship", "", ui);
            });
    }
}
