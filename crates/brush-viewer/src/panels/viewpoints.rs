use crate::{viewer::ViewerContext, ViewerPane};
use brush_train::scene::ViewType;
use egui::{Slider, TextureHandle, TextureOptions};

pub(crate) struct ViewpointsPane {
    view_type: ViewType,
    selected_view: Option<(usize, TextureHandle)>,
}

impl ViewpointsPane {
    pub(crate) fn new() -> Self {
        Self {
            view_type: ViewType::Train,
            selected_view: None,
        }
    }
}

impl ViewerPane for ViewpointsPane {
    fn title(&self) -> String {
        "Dataset".to_owned()
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut ViewerContext) -> egui_tiles::UiResponse {
        // Empty scene, nothing to show.
        if context.dataset.train.views.len() == 0 {
            ui.label("Load a dataset to get started");
            return egui_tiles::UiResponse::None;
        }

        let scene = if let Some(eval_scene) = context.dataset.eval.as_ref() {
            match self.view_type {
                ViewType::Train => &context.dataset.train,
                _ => eval_scene,
            }
        } else {
            &context.dataset.train
        };
        let scene = scene.clone();

        // TODO: HOw to get the current camera..
        let mut nearest_view = scene.get_nearest_view(&context.camera);

        let Some(nearest) = nearest_view.as_mut() else {
            return egui_tiles::UiResponse::None;
        };

        // Update image if dirty.
        let mut dirty = self.selected_view.is_none();

        if let Some(view) = self.selected_view.as_ref() {
            dirty |= view.0 != *nearest;
        }

        let view_count = scene.views.len();

        ui.horizontal(|ui| {
            let mut sel_view = false;

            if ui.button("⏪").clicked() {
                sel_view = true;
                *nearest = (*nearest + view_count - 1) % view_count;
            }
            sel_view |= ui
                .add(
                    Slider::new(nearest, 0..=view_count - 1)
                        .suffix(format!("/ {view_count}"))
                        .custom_formatter(|num, _| format!("{}", num as usize + 1))
                        .custom_parser(|s| s.parse::<usize>().ok().map(|n| n as f64 - 1.0)),
                )
                .dragged();
            if ui.button("⏩").clicked() {
                sel_view = true;
                *nearest = (*nearest + 1) % view_count;
            }

            if sel_view {
                let camera = &context.dataset.train.views[*nearest].camera.clone();
                context.focus_view(camera);
            }

            ui.add_space(10.0);

            if context.dataset.eval.is_some() {
                for (t, l) in [ViewType::Train, ViewType::Eval]
                    .into_iter()
                    .zip(["train", "eval"])
                {
                    if ui.selectable_label(self.view_type == t, l).clicked() {
                        self.view_type = t;
                        dirty = true;
                    };
                }
            }
        });

        if dirty {
            let view = &scene.views[*nearest];
            let color_img = egui::ColorImage::from_rgb(
                [view.image.width() as usize, view.image.height() as usize],
                &view.image.to_rgb8().into_vec(),
            );
            self.selected_view = Some((
                *nearest,
                ui.ctx()
                    .load_texture("nearest_view_tex", color_img, TextureOptions::default()),
            ));
        }

        if let Some(view) = self.selected_view.as_ref() {
            ui.add(egui::Image::new(&view.1).shrink_to_fit());
            ui.label(&scene.views[*nearest].name);
        }

        egui_tiles::UiResponse::None
    }
}
