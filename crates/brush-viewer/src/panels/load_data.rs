use crate::{train_loop::TrainArgs, viewer::ViewerContext, ViewerPane};
use egui::Slider;

pub(crate) struct LoadDataPanel {
    target_train_resolution: Option<u32>,
    max_frames: Option<usize>,
}

impl LoadDataPanel {
    pub(crate) fn new() -> Self {
        Self {
            // High resolution performance just isn't great at the moment... limit this for now by default.
            target_train_resolution: Some(800),
            max_frames: None,
        }
    }
}

impl ViewerPane for LoadDataPanel {
    fn title(&self) -> String {
        "Load data".to_owned()
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut ViewerContext) -> egui_tiles::UiResponse {
        ui.label("Select a .ply to visualize, or a .zip with training data.");

        if ui.button("Pick a file").clicked() {
            let train_args = TrainArgs {
                frame_count: self.max_frames,
                target_resolution: self.target_train_resolution,
            };
            context.start_data_load(train_args);
        }

        ui.add_space(10.0);
        ui.heading("Train settings");

        let mut limit_res = self.target_train_resolution.is_some();
        if ui
            .checkbox(&mut limit_res, "Limit training resolution")
            .clicked()
        {
            self.target_train_resolution = if limit_res { Some(800) } else { None };
        }

        if let Some(target_res) = self.target_train_resolution.as_mut() {
            ui.add(Slider::new(target_res, 32..=2048));
        }

        let mut limit_frames = self.max_frames.is_some();
        if ui.checkbox(&mut limit_frames, "Limit max frames").clicked() {
            self.max_frames = if limit_frames { Some(32) } else { None };
        }

        if let Some(max_frames) = self.max_frames.as_mut() {
            ui.add(Slider::new(max_frames, 1..=256));
        }

        ui.add_space(15.0);

        if ui.input(|r| r.key_pressed(egui::Key::Escape)) {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }

        egui_tiles::UiResponse::None
    }
}
