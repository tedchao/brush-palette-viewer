use std::io::Read;

use anyhow::Context;
use async_std::{
    channel::{Sender, TrySendError},
    stream::StreamExt,
};
use brush_dataset::{scene_batch::SceneLoader, Dataset, ZipData};
use brush_render::{gaussian_splats::RandomSplatsConfig, PrimaryBackend};
use brush_train::train::{SplatTrainer, TrainConfig};
use burn::{
    backend::Autodiff, lr_scheduler::exponential::ExponentialLrSchedulerConfig,
    module::AutodiffModule,
};
use burn_wgpu::WgpuDevice;
use web_time::Instant;
use zip::ZipArchive;

use crate::{splat_import, viewer::ViewerMessage};

pub(crate) struct TrainArgs {
    pub frame_count: Option<usize>,
    pub target_resolution: Option<u32>,
}

pub(crate) struct SharedTrainState {
    pub paused: bool,
}

pub(crate) async fn train_loop(
    zip_data: ZipData,
    device: WgpuDevice,
    sender: Sender<ViewerMessage>,
    egui_ctx: egui::Context,
    train_args: TrainArgs,
) -> anyhow::Result<()> {
    let total_steps = 30000;

    let mut archive = ZipArchive::new(zip_data.open_for_read())?;

    // Load initial splats if included
    let mut initial_splats = None;

    let data = archive
        .by_name("init.ply")
        .map(|f| f.bytes().collect::<std::io::Result<Vec<u8>>>());

    if let Ok(Ok(data)) = data {
        let total_count = splat_import::ply_count(&data).context("Invalid ply file")?;
        let splat_stream =
            splat_import::load_splat_from_ply::<Autodiff<PrimaryBackend>>(&data, device.clone());

        let mut splat_stream = std::pin::pin!(splat_stream);
        while let Some(splats) = splat_stream.next().await {
            egui_ctx.request_repaint();

            let splats = splats?;
            let msg = ViewerMessage::SplatLoad {
                splats: splats.valid(),
                total_count,
            };

            sender
                .send(msg)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to send message: {}", e))?;

            initial_splats = Some(splats);
        }
    }

    let data_stream = brush_dataset::read_dataset::<Autodiff<PrimaryBackend>>(
        archive.clone(),
        train_args.frame_count,
        train_args.target_resolution,
    )?;

    let mut dataset = Dataset::empty();
    let mut data_stream = std::pin::pin!(data_stream);

    while let Some(d) = data_stream.next().await {
        dataset = d?;

        if sender
            .send(ViewerMessage::Dataset {
                data: dataset.clone(),
                final_data: false,
            })
            .await
            .is_err()
        {
            anyhow::bail!("Channel closed");
        }

        egui_ctx.request_repaint();
    }

    // Resend last dataset now that we know it's final one.
    if sender
        .send(ViewerMessage::Dataset {
            data: dataset.clone(),
            final_data: true,
        })
        .await
        .is_err()
    {
        anyhow::bail!("Channel closed");
    }

    egui_ctx.request_repaint();

    let mut splats = if let Some(splats) = initial_splats {
        splats
    } else {
        // By default, spawn the splats in bounds.
        let bounds = dataset.train.bounds(0.0);
        let bounds_extent = bounds.extent.length();
        let adjusted_bounds = dataset.train.bounds(bounds_extent);
        let config = RandomSplatsConfig::new(adjusted_bounds);
        config.init(&device)
    };

    let train_scene = dataset.train.clone();

    let lr_scale = train_scene.bounds(0.0).extent.length() as f64;
    let lr_max = 1.6e-4 * lr_scale;
    let decay = 1e-2f64.powf(1.0 / total_steps as f64);

    let config = TrainConfig::new(ExponentialLrSchedulerConfig::new(lr_max, decay))
        .with_total_steps(total_steps);

    let mut dataloader = SceneLoader::new(&train_scene, 1, &device);
    let mut trainer = SplatTrainer::new(splats.num_splats(), &config, &splats);

    loop {
        // TODO: Restore the pause button but better.
        // if shared_state.read().paused {
        //     #[cfg(not(target_arch = "wasm32"))]
        //     std::thread::yield_now();
        //     #[cfg(target_arch = "wasm32")]
        //     gloo_timers::future::TimeoutFuture::new(0).await;
        //     continue;
        // }
        //

        let batch = {
            // let _span = trace_span!("Get batch").entered();
            dataloader.next_batch().await
        };
        let (new_splats, stats) = trainer.step(batch, train_scene.background, splats).await?;

        splats = new_splats;

        if trainer.iter % 5 == 0 {
            let msg = ViewerMessage::TrainStep {
                splats: splats.clone(),
                stats,
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
    }

    Ok(())
}
