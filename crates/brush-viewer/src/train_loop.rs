use anyhow::anyhow;
use async_std::{
    channel::{Receiver, Sender, TryRecvError, TrySendError},
    stream::StreamExt,
};
use brush_dataset::{scene_batch::SceneLoader, Dataset, LoadDatasetArgs, LoadInitArgs, ZipData};
use brush_render::{
    gaussian_splats::{RandomSplatsConfig, Splats},
    PrimaryBackend,
};
use brush_train::train::{SplatTrainer, TrainConfig};
use burn::{lr_scheduler::exponential::ExponentialLrSchedulerConfig, module::AutodiffModule};
use burn_wgpu::WgpuDevice;
use web_time::Instant;
use zip::ZipArchive;

use crate::viewer::ViewerMessage;

#[derive(Debug, Clone)]
pub enum TrainMessage {
    Paused(bool),
    Eval { view_count: Option<usize> },
}

pub(crate) async fn train_loop(
    zip_data: ZipData,
    device: WgpuDevice,
    sender: Sender<ViewerMessage>,
    receiver: Receiver<TrainMessage>,
    load_data_args: LoadDatasetArgs,
    load_init_args: LoadInitArgs,
) -> anyhow::Result<()> {
    let batch_size = 1;

    // Maybe good if the seed would be configurable.
    let seed = 42;
    <PrimaryBackend as burn::prelude::Backend>::seed(seed);

    let archive = ZipArchive::new(zip_data.open_for_read())?;

    // Load initial splats if included
    let mut initial_splats = None;
    let mut splat_stream =
        brush_dataset::read_dataset_init(archive.clone(), &device, &load_init_args);

    if let Ok(splat_stream) = splat_stream.as_mut() {
        while let Some(splats) = splat_stream.next().await {
            let splats = splats?;
            let msg = ViewerMessage::Splats {
                iter: 0,
                splats: Box::new(splats.valid()),
            };
            sender
                .send(msg)
                .await
                .map_err(|e| anyhow!("Failed to send message: {}", e))?;
            initial_splats = Some(splats);
        }
    }

    let mut dataset = Dataset::empty();
    let data_stream = brush_dataset::read_dataset_views(archive.clone(), &load_data_args)?;
    let mut data_stream = std::pin::pin!(data_stream);
    while let Some(d) = data_stream.next().await {
        dataset = d?;

        sender
            .send(ViewerMessage::Dataset {
                data: dataset.clone(),
            })
            .await
            .map_err(|e| anyhow!("Failed to send message: {}", e))?;
    }
    sender
        .send(ViewerMessage::DoneLoading { training: true })
        .await
        .map_err(|e| anyhow!("Failed to send message: {}", e))?;

    // TODO: Maybe just move this default to the dataset reader.
    let mut splats = if let Some(splats) = initial_splats {
        splats
    } else {
        // By default, spawn the splats in bounds.
        let bounds = dataset.train.bounds(0.0);
        let bounds_extent = bounds.extent.length();
        let adjusted_bounds = dataset.train.bounds(bounds_extent);
        let config = RandomSplatsConfig::new().with_sh_degree(load_init_args.sh_degree);
        Splats::from_random_config(config, adjusted_bounds, &device)
    };

    let train_scene = dataset.train.clone();
    let eval_scene = dataset.eval.clone();

    let total_steps = 30000;

    let scene_extent = train_scene.bounds(0.0).extent.length() as f64;
    let lr_max = 1.6e-4 * scene_extent;
    let decay = 1e-2f64.powf(1.0 / total_steps as f64);
    let config = TrainConfig::new(ExponentialLrSchedulerConfig::new(lr_max, decay))
        .with_total_steps(total_steps);

    let mut dataloader = SceneLoader::new(&train_scene, batch_size, seed, &device);
    let mut trainer = SplatTrainer::new(splats.num_splats(), &config, &splats);

    let mut is_paused = false;

    loop {
        let message = if is_paused {
            // When paused, wait for a message async and handle it. The "default" train iteration
            // won't be hit.
            match receiver.recv().await {
                Ok(message) => Some(message),
                Err(_) => break, // if channel is closed, stop.
            }
        } else {
            // Otherwise, check for messages, and if there isn't any just proceed training.
            match receiver.try_recv() {
                Ok(message) => Some(message),
                Err(TryRecvError::Empty) => None, // Nothing special to do.
                Err(TryRecvError::Closed) => break, // If channel is closed, stop.
            }
        };

        match message {
            Some(TrainMessage::Paused(paused)) => {
                is_paused = paused;
            }
            Some(TrainMessage::Eval { view_count }) => {
                if let Some(eval_scene) = eval_scene.as_ref() {
                    let eval = brush_train::eval::eval_stats(
                        splats.valid(),
                        eval_scene,
                        view_count,
                        &device,
                    )
                    .await;

                    let _ = sender
                        .send(ViewerMessage::Eval {
                            iter: trainer.iter,
                            eval,
                        })
                        .await;
                }
            }
            // By default, continue train steps.
            None => {
                let batch = {
                    // let _span = trace_span!("Get batch").entered();
                    dataloader.next_batch().await
                };
                let (new_splats, stats) =
                    trainer.step(batch, train_scene.background, splats).await?;
                splats = new_splats;

                if trainer.iter % 5 == 0 {
                    if let Err(TrySendError::Closed(_)) = sender.try_send(ViewerMessage::Splats {
                        iter: trainer.iter,
                        splats: Box::new(splats.valid()),
                    }) {
                        break; // channel closed, bail.
                    }

                    if let Err(TrySendError::Closed(_)) =
                        sender.try_send(ViewerMessage::TrainStep {
                            stats,
                            iter: trainer.iter,
                            timestamp: Instant::now(),
                        })
                    {
                        break; // channel closed, bail.
                    }
                }
            }
        }
    }

    Ok(())
}
