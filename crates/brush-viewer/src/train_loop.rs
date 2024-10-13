use std::sync::{Arc, RwLock};

use async_channel::{Sender, TrySendError};
use brush_dataset::{scene_batch::SceneLoader, Dataset};
use brush_render::gaussian_splats::RandomSplatsConfig;
use brush_train::train::{SplatTrainer, TrainConfig};
use burn::{
    backend::Autodiff, lr_scheduler::exponential::ExponentialLrSchedulerConfig,
    module::AutodiffModule, tensor::ElementConversion,
};
use burn_wgpu::WgpuDevice;
use futures_lite::StreamExt;
use tracing::info_span;
use web_time::Instant;
use zip::ZipArchive;

use crate::viewer::ViewerMessage;

pub(crate) struct TrainState {
    pub last_train_step: (Instant, u32),
    pub train_iter_per_s: f32,
    pub dataset: Dataset,
    pub selected_view: Option<(usize, egui::TextureHandle)>,
    pub shared: Arc<RwLock<SharedTrainState>>,
}

impl TrainState {
    pub fn new() -> Self {
        Self {
            last_train_step: (Instant::now(), 0),
            train_iter_per_s: 0.0,
            dataset: Dataset::empty(),
            selected_view: None,
            shared: Arc::new(RwLock::new(SharedTrainState { paused: false })),
        }
    }

    pub fn on_iter(&mut self, stamp: Instant, iter: u32) {
        self.train_iter_per_s =
            (iter - self.last_train_step.1) as f32 / (stamp - self.last_train_step.0).as_secs_f32();
        self.last_train_step = (stamp, iter);
    }
}

pub(crate) struct TrainArgs {
    pub frame_count: Option<usize>,
    pub target_resolution: Option<u32>,
}

pub(crate) struct SharedTrainState {
    pub paused: bool,
}

pub(crate) async fn train_loop(
    zip_data: &[u8],
    device: WgpuDevice,
    sender: Sender<ViewerMessage>,
    egui_ctx: egui::Context,
    train_args: TrainArgs,
    shared_state: Arc<RwLock<SharedTrainState>>,
) -> anyhow::Result<()> {
    let total_steps = 30000;

    let archive = ZipArchive::new(std::io::Cursor::new(zip_data))?;
    let data_stream = brush_dataset::read_dataset(
        archive,
        train_args.frame_count,
        train_args.target_resolution,
    )?;

    let mut dataset = Dataset::empty();

    let mut data_stream = std::pin::pin!(data_stream);
    while let Some(d) = data_stream.next().await {
        dataset = d?;
        if sender
            .send(ViewerMessage::Dataset(dataset.clone()))
            .await
            .is_err()
        {
            anyhow::bail!("Failed to send dataset")
        }
        egui_ctx.request_repaint();
    }

    let mut train_scene = dataset.train_scene().clone();
    train_scene.center_cameras();

    // Some extra distance to add to camera extents.
    let bounds = train_scene.bounds(0.0);
    let bounds_extent = bounds.extent.length();
    let adjusted_bounds = train_scene.bounds(bounds_extent);
    let splat_config = RandomSplatsConfig::new(adjusted_bounds);

    let config = TrainConfig::new(
        ExponentialLrSchedulerConfig::new(1.6e-4, 1e-2f64.powf(1.0 / total_steps as f64)),
        splat_config,
    )
    .with_total_steps(total_steps);

    let visualize = crate::visualize::VisualizeTools::new();
    visualize.log_scene(&train_scene)?;

    let mut splats = config
        .initial_model_config
        .init::<Autodiff<brush_render::PrimaryBackend>>(&device);
    let mut dataloader = SceneLoader::new(&train_scene, 1, &device);
    let mut trainer = SplatTrainer::new(splats.num_splats(), &config, &splats);

    loop {
        if shared_state.read().unwrap().paused {
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::yield_now();
            #[cfg(target_arch = "wasm32")]
            gloo_timers::future::TimeoutFuture::new(0).await;
            continue;
        }

        if let Some(eval_scene) = dataset.eval_scene() {
            if trainer.iter % config.eval_every == 0 {
                let eval =
                    brush_train::eval::eval_stats(&splats, eval_scene, Some(4), &device).await;
                visualize.log_eval_stats(trainer.iter, &eval)?;
            }
        }

        if trainer.iter % config.visualize_splats_every == 0 {
            visualize.log_splats(&splats).await?;
        }

        let batch = {
            let _span = info_span!("Get batch").entered();
            dataloader.next_batch().await
        };
        let gt_image = batch.gt_images.clone();
        let (new_splats, stats) = trainer.step(batch, train_scene.background, splats).await?;
        splats = new_splats;
        if trainer.iter % config.visualize_splats_every == 0 {
            visualize.log_train_stats(&splats, &stats, gt_image).await?;
        }
        if trainer.iter % 5 == 0 {
            let _span = info_span!("Send batch").entered();

            let msg = ViewerMessage::TrainStep {
                splats: splats.valid(),
                loss: stats.loss.into_scalar_async().await.elem::<f32>(),
                iter: trainer.iter,
                timestamp: Instant::now(),
            };

            match sender.try_send(msg) {
                Ok(_) => (),
                Err(TrySendError::Full(_)) => (),
                Err(_) => {
                    break; // channel closed, bail.
                }
            }
            egui_ctx.request_repaint();
        }

        // On wasm, yield to the browser.
        #[cfg(target_arch = "wasm32")]
        gloo_timers::future::TimeoutFuture::new(0).await;
    }

    Ok(())
}
