use std::sync::Arc;

use crate::{viewer::ViewerContext, visualize::VisualizeTools, ViewerPanel};
use brush_train::spawn_future;
use burn_wgpu::WgpuDevice;

pub(crate) struct RerunPanel {
    visualize: Option<Arc<VisualizeTools>>,
    device: WgpuDevice,
    eval_every: u32,
    eval_view_count: Option<usize>,

    log_train_stats_every: u32,
    visualize_splats_every: Option<u32>,

    read_to_log_dataset: bool,
}

impl RerunPanel {
    pub(crate) fn new(device: WgpuDevice) -> Self {
        RerunPanel {
            visualize: None,
            eval_every: 1000,
            eval_view_count: None,
            log_train_stats_every: 50,
            visualize_splats_every: None,
            device,
            read_to_log_dataset: false,
        }
    }
}

impl ViewerPanel for RerunPanel {
    fn title(&self) -> String {
        "Rerun".to_owned()
    }

    fn on_message(&mut self, message: crate::viewer::ViewerMessage, context: &mut ViewerContext) {
        match message {
            // TODO: New stream on start load?
            // crate::viewer::ViewerMessage::StartLoading => self.visualize = VisualizeTools::new(),
            crate::viewer::ViewerMessage::DoneLoading { training } => {
                if training {
                    self.read_to_log_dataset = true;
                }
            }
            crate::viewer::ViewerMessage::TrainStep {
                splats,
                stats,
                iter,
                timestamp: _,
            } => {
                let Some(visualize) = self.visualize.clone() else {
                    return;
                };

                if iter % self.log_train_stats_every == 0 {
                    let splats = splats.clone();
                    let visualize = visualize.clone();
                    let stats = stats.clone();

                    spawn_future(async move {
                        if let Err(e) = visualize.log_train_stats(iter, splats, stats).await {
                            log::error!("Error logging train stats: {}", e);
                        }
                    });
                }
                if let Some(eval_scene) = context.dataset.eval.clone() {
                    if iter % self.eval_every == 0 {
                        log::info!("Logging eval stats");
                        let device = self.device.clone();
                        let visualize = visualize.clone();
                        let splats = splats.clone();
                        let num_eval = self.eval_view_count;

                        spawn_future(async move {
                            log::info!("Gathering eval stats");
                            let eval = brush_train::eval::eval_stats(
                                splats,
                                &eval_scene,
                                num_eval,
                                &device,
                            )
                            .await;

                            log::info!("Visualizing eval stats");
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

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut ViewerContext) {
        let Some(visualize) = self.visualize.clone() else {
            if ui.button("Enable rerun").clicked() {
                self.visualize = Some(Arc::new(VisualizeTools::new()));
            }
            return;
        };

        if self.read_to_log_dataset {
            let visualize = visualize.clone();
            let train_scene = context.dataset.train.clone();
            spawn_future(async move {
                if let Err(e) = visualize.log_scene(&train_scene) {
                    log::error!("Error logging initial scene: {}", e);
                }
            });
            self.read_to_log_dataset = false;
        }

        ui.add(
            egui::Slider::new(&mut self.log_train_stats_every, 1..=5000)
                .text("Log train stats every"),
        );

        ui.add(egui::Slider::new(&mut self.eval_every, 1..=5000).text("Evaluate every"));

        let mut limit_eval_views = self.eval_view_count.is_some();
        ui.checkbox(&mut limit_eval_views, "Limit eval views");
        if limit_eval_views != self.eval_view_count.is_some() {
            self.eval_view_count = if limit_eval_views { Some(5) } else { None };
        }

        if let Some(count) = self.eval_view_count.as_mut() {
            ui.add(egui::Slider::new(count, 1..=100).text("Eval view count"));
        }

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
