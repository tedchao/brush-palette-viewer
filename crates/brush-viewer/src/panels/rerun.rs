use std::{future::Future, pin::Pin, sync::Arc};

use crate::{
    train_loop::TrainMessage,
    viewer::{ViewerContext, ViewerMessage},
    visualize::VisualizeTools,
    ViewerPanel,
};
use async_std::{
    channel::{self, Sender},
    task,
};
use burn_wgpu::WgpuDevice;

pub(crate) struct RerunPanel {
    visualize: Option<Arc<VisualizeTools>>,
    device: WgpuDevice,
    eval_every: u32,
    eval_view_count: Option<usize>,

    log_train_stats_every: u32,
    visualize_splats_every: Option<u32>,

    read_to_log_dataset: bool,

    // TODO: This async logic is maybe better moved to the visualize module.
    task_queue: Sender<Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>>>,
}

impl RerunPanel {
    pub(crate) fn new(device: WgpuDevice) -> Self {
        let (queue_send, queue_receive) = channel::unbounded();

        // Spawn a task to handle futures one by one as they come in.
        task::spawn(async move {
            while let Ok(fut) = queue_receive.recv().await {
                if let Err(e) = fut.await {
                    log::error!("Error logging to rerun: {}", e);
                }
            }
        });

        RerunPanel {
            visualize: None,
            eval_every: 1000,
            eval_view_count: None,
            log_train_stats_every: 50,
            visualize_splats_every: None,
            device,
            read_to_log_dataset: false,
            task_queue: queue_send,
        }
    }
}

impl RerunPanel {
    fn queue_task(&self, fut: impl Future<Output = anyhow::Result<()>> + Send + 'static) {
        // Ignore this error - if the channel is closed we just don't do anything and drop
        // the future.
        let _ = self.task_queue.try_send(Box::pin(fut));
    }
}

impl ViewerPanel for RerunPanel {
    fn title(&self) -> String {
        "Rerun".to_owned()
    }

    fn on_message(&mut self, message: crate::viewer::ViewerMessage, context: &mut ViewerContext) {
        match message {
            crate::viewer::ViewerMessage::StartLoading { training } => {
                if training {
                    if self.visualize.is_some() {
                        self.visualize = Some(Arc::new(VisualizeTools::new()));
                    }
                } else {
                    self.visualize = None;
                }
            }
            crate::viewer::ViewerMessage::DoneLoading { training } => {
                if training {
                    self.read_to_log_dataset = true;
                }
            }
            crate::viewer::ViewerMessage::Splats { iter, splats } => {
                let Some(visualize) = self.visualize.clone() else {
                    return;
                };

                if let Some(every) = self.visualize_splats_every {
                    if iter % every == 0 {
                        self.queue_task(visualize.log_splats(*splats));
                    }
                }
            }
            crate::viewer::ViewerMessage::TrainStep {
                stats,
                iter,
                timestamp: _,
            } => {
                let Some(visualize) = self.visualize.clone() else {
                    return;
                };

                // If needed, start an eval run.
                if iter % self.eval_every == 0 {
                    context.send_train_message(TrainMessage::Eval {
                        view_count: self.eval_view_count,
                    });
                }

                // Log out train stats.
                if iter % self.log_train_stats_every == 0 {
                    self.queue_task(visualize.log_train_stats(iter, *stats));
                }
            }
            ViewerMessage::EvalResult { iter, eval } => {
                let Some(visualize) = self.visualize.clone() else {
                    return;
                };

                self.queue_task(visualize.log_eval_stats(iter, eval));
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
            self.queue_task(visualize.log_scene(context.dataset.train.clone()));
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
            self.eval_view_count = if limit_eval_views {
                Some(
                    context
                        .dataset
                        .eval
                        .as_ref()
                        .map_or(0, |eval| eval.views.len()),
                )
            } else {
                None
            };
        }

        if let Some(count) = self.eval_view_count.as_mut() {
            ui.add(
                egui::Slider::new(
                    count,
                    1..=context
                        .dataset
                        .eval
                        .as_ref()
                        .map_or(1, |eval| eval.views.len()),
                )
                .text("Eval view count"),
            );
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
