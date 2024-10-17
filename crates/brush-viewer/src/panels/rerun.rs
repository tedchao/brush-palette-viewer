use std::sync::Arc;

use crate::{viewer::ViewerContext, visualize::VisualizeTools, ViewerPanel};
use brush_dataset::Dataset;
use brush_train::spawn_future;
use burn_wgpu::WgpuDevice;

pub(crate) struct RerunPanel {
    visualize: Option<Arc<VisualizeTools>>,
    dataset: Dataset,
    device: WgpuDevice,
    eval_every: u32,
    log_train_stats_every: u32,
    visualize_splats_every: Option<u32>,
}

impl RerunPanel {
    pub(crate) fn new(device: WgpuDevice) -> Self {
        RerunPanel {
            visualize: None,
            dataset: Dataset::empty(),
            eval_every: 500,
            log_train_stats_every: 50,
            visualize_splats_every: None,
            device,
        }
    }
}

impl ViewerPanel for RerunPanel {
    fn title(&self) -> String {
        "Rerun".to_owned()
    }

    fn on_message(&mut self, message: crate::viewer::ViewerMessage, _: &mut ViewerContext) {
        let Some(visualize) = self.visualize.clone() else {
            return;
        };

        match message {
            // TODO: New stream on start load?
            // crate::viewer::ViewerMessage::StartLoading => self.visualize = VisualizeTools::new(),
            crate::viewer::ViewerMessage::Dataset { data, final_data } => {
                // Log the final dataset to rerun.
                if final_data {
                    self.dataset = data;
                    let visualize = visualize.clone();
                    let train_scene = self.dataset.train.clone();

                    spawn_future(async move {
                        if let Err(e) = visualize.log_scene(&train_scene) {
                            log::error!("Error logging initial scene: {}", e);
                        }
                    });
                }
            }
            crate::viewer::ViewerMessage::TrainStep {
                splats,
                stats,
                iter,
                timestamp: _,
            } => {
                if iter % self.log_train_stats_every == 0 {
                    let splats = splats.clone();
                    let visualize = visualize.clone();
                    let stats = stats.clone();

                    spawn_future(async move {
                        if let Err(e) = visualize.log_train_stats(iter, splats.clone(), stats).await
                        {
                            log::error!("Error logging train stats: {}", e);
                        }
                    });
                }

                if let Some(eval_scene) = self.dataset.eval.clone() {
                    if iter % self.eval_every == 0 {
                        let device = self.device.clone();
                        let visualize = visualize.clone();
                        let splats = splats.clone();
                        spawn_future(async move {
                            let eval = brush_train::eval::eval_stats(
                                splats,
                                &eval_scene,
                                Some(4),
                                &device,
                            )
                            .await;

                            if let Err(e) = visualize.log_eval_stats(iter, eval) {
                                log::error!("Error logging eval stats: {}", e);
                            }
                        });
                    }
                }

                if let Some(every) = self.visualize_splats_every {
                    if iter % every == 0 {
                        let visualize = visualize.clone();
                        let splats = splats.clone();

                        // TODO: Spawn a task with this?
                        spawn_future(async move {
                            if let Err(e) = visualize.log_splats(splats).await {
                                log::error!("Error logging splats: {}", e);
                            }
                        });
                    }
                }
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &mut ViewerContext) {
        if self.visualize.is_none() {
            if ui.button("Enable rerun").clicked() {
                self.visualize = Some(Arc::new(VisualizeTools::new()));
            }
            return;
        }

        ui.add(
            egui::Slider::new(&mut self.log_train_stats_every, 1..=5000)
                .text("Log train stats every"),
        );

        ui.add(egui::Slider::new(&mut self.eval_every, 1..=5000).text("Evaluate every"));

        let mut visualize_splats = self.visualize_splats_every.is_some();
        ui.checkbox(&mut visualize_splats, "Visualize splats");
        if visualize_splats != self.visualize_splats_every.is_some() {
            self.visualize_splats_every = if visualize_splats { Some(500) } else { None };
        }

        if let Some(every) = self.visualize_splats_every.as_mut() {
            ui.add(egui::Slider::new(every, 1..=5000).text("Visualize splats every"));
        }
    }
}
