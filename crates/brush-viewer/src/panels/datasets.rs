use crate::{
    viewer::{ViewerContext, ViewerMessage},
    ViewerPanel,
};
use brush_train::scene::ViewType;
use egui::{Slider, TextureHandle, TextureOptions};

pub(crate) struct DatasetPanel {
    view_type: ViewType,
    selected_view: Option<(usize, TextureHandle)>,
    loading: bool,
}

impl DatasetPanel {
    pub(crate) fn new() -> Self {
        Self {
            view_type: ViewType::Train,
            selected_view: None,
            loading: false,
        }
    }
}

impl ViewerPanel for DatasetPanel {
    fn title(&self) -> String {
        "Dataset".to_owned()
    }

    fn on_message(&mut self, message: ViewerMessage, context: &mut ViewerContext) {
        match message {
            ViewerMessage::PickFile => {
                self.loading = false;
            }
            ViewerMessage::StartLoading { training } => {
                self.loading = training;
            }
            ViewerMessage::Dataset { data: d } => {
                // Set train view to last loaded camera.
                if let Some(view) = d.train.views.last() {
                    context.focus_view(&view.camera);
                }
                context.dataset = d;
            }
            ViewerMessage::DoneLoading { training: _ } => {
                self.loading = false;
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut ViewerContext) {
        // Empty scene, nothing to show.
        if !self.loading && context.dataset.train.views.is_empty() {
            ui.label("Load a dataset to train a splat.");
            ui.label(r#"
Datasets have to be provided in a single zip file. The format of this archive can be:
    - the format used in the synthetic NeRF example data, containing a transform_train.json and images, please see a reference `zip` linked below.
    - COLMAP data, by zipping the folder containing the `images` & `sparse` folder.
            "#);
            return;
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

        let mut nearest_view_ind = scene.get_nearest_view(&context.camera);
        if let Some(nearest) = nearest_view_ind.as_mut() {
            let nearest_view = &scene.views[*nearest];

            // Update image if dirty.
            let mut dirty = self.selected_view.is_none();

            if let Some(view) = self.selected_view.as_ref() {
                dirty |= view.0 != *nearest;
            }

            if dirty {
                let color_img = egui::ColorImage::from_rgb(
                    [
                        nearest_view.image.width() as usize,
                        nearest_view.image.height() as usize,
                    ],
                    &nearest_view.image.to_rgb8().into_vec(),
                );
                self.selected_view = Some((
                    *nearest,
                    ui.ctx()
                        .load_texture("nearest_view_tex", color_img, TextureOptions::default()),
                ));
            }

            if let Some(selected) = self.selected_view.as_ref() {
                ui.add(egui::Image::new(&selected.1).shrink_to_fit());
                let info = format!(
                    "{} ({}x{})",
                    nearest_view.name,
                    nearest_view.image.width(),
                    nearest_view.image.height()
                );
                ui.label(info);
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
        }

        if self.loading {
            ui.label("Loading...");
        }
    }
}
