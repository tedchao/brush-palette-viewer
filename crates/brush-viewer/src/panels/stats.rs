use crate::{
    train_loop::TrainMessage,
    viewer::{ViewerContext, ViewerMessage},
    ViewerPanel,
};
use burn_jit::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use web_time::Instant;

pub(crate) struct StatsPanel {
    device: WgpuDevice,

    last_train_step: (Instant, u32),
    train_iter_per_s: f32,
    last_eval_psnr: Option<f32>,

    training_started: bool,
    paused: bool,
    num_splats: usize,
}

impl StatsPanel {
    pub(crate) fn new(device: WgpuDevice) -> Self {
        Self {
            device,
            last_train_step: (Instant::now(), 0),
            train_iter_per_s: 0.0,
            last_eval_psnr: None,
            training_started: false,
            paused: false,
            num_splats: 0,
        }
    }
}

impl ViewerPanel for StatsPanel {
    fn title(&self) -> String {
        "Stats".to_owned()
    }

    fn on_message(&mut self, message: crate::viewer::ViewerMessage, _: &mut ViewerContext) {
        match message {
            ViewerMessage::StartLoading { training } => {
                self.last_train_step = (Instant::now(), 0);
                self.train_iter_per_s = 0.0;
                self.num_splats = 0;
                self.last_eval_psnr = None;
                self.training_started = training;
            }
            ViewerMessage::TrainStep {
                stats: _,
                iter,
                timestamp,
            } => {
                self.train_iter_per_s = (iter - self.last_train_step.1) as f32
                    / (timestamp - self.last_train_step.0).as_secs_f32();
                self.last_train_step = (timestamp, iter);
            }
            ViewerMessage::Splats { iter: _, splats } => {
                self.num_splats = splats.num_splats();
            }
            ViewerMessage::EvalResult { iter: _, eval } => {
                let avg_psnr =
                    eval.samples.iter().map(|s| s.psnr).sum::<f32>() / (eval.samples.len() as f32);
                self.last_eval_psnr = Some(avg_psnr);
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut ViewerContext) {
        ui.label(format!("Splats: {}", self.num_splats));
        if self.training_started {
            ui.label(format!("Train step: {}", self.last_train_step.1));
            ui.label(format!("steps/s: {:.1} ", self.train_iter_per_s));

            if let Some(psnr) = self.last_eval_psnr {
                ui.label(format!("Last eval psnr: {:.2} ", psnr));
            }
            let label = if self.paused {
                "⏸ paused"
            } else {
                "⏵ training"
            };

            if ui.selectable_label(self.paused, label).clicked() {
                self.paused = !self.paused;
                context.send_train_message(TrainMessage::Paused(self.paused));
            }
        }

        ui.add_space(10.0);

        let client = WgpuRuntime::client(&self.device);
        let memory = client.memory_usage();
        ui.label(format!("GPU memory \n {}", memory));
    }
}
