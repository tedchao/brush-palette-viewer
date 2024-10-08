use std::sync::{Arc, RwLock};

use anyhow::Context;
use async_channel::{Receiver, Sender, TryRecvError, TrySendError};
use brush_dataset::scene_batch::SceneLoader;
use brush_render::gaussian_splats::{RandomSplatsConfig, Splats};
use burn::lr_scheduler::exponential::ExponentialLrSchedulerConfig;
use burn::tensor::ElementConversion;
use burn::{backend::Autodiff, module::AutodiffModule};
use burn_wgpu::{ RuntimeOptions, Wgpu, WgpuDevice};
use egui::{Hyperlink, Slider, TextureOptions};
use futures_lite::StreamExt;

use tracing::info_span;
use web_time::Instant;

use brush_dataset::{self, Dataset};
use brush_train::scene::Scene;
use brush_train::train::{SplatTrainer, TrainConfig};

use crate::splat_import;
use crate::splat_view::SplatView;

enum ViewerMessage {
    Error(anyhow::Error),

    // Initial splat cloud to be created.
    SplatLoad {
        splats: Splats<Wgpu>,
        total_count: usize,
    },

    // Loaded a bunch of viewpoints to train on.
    Dataset(Dataset),

    TrainStep {
        splats: Splats<Wgpu>,
        loss: f32,
        iter: u32,
        timestamp: Instant,
    },
}

struct SharedTrainState {
    paused: bool,
}

struct TrainState {
    last_train_step: (Instant, u32),
    train_iter_per_s: f32,
    dataset: Dataset,
    selected_view: Option<(usize, egui::TextureHandle)>,
    shared: Arc<RwLock<SharedTrainState>>,
}

impl TrainState {
    fn new() -> Self {
        Self {
            last_train_step: (Instant::now(), 0),
            train_iter_per_s: 0.0,
            dataset: Dataset {
                train: Scene::new(vec![], glam::Vec3::ZERO),
                test: None,
                eval: None,
            },
            selected_view: None,
            shared: Arc::new(RwLock::new(SharedTrainState { paused: false })),
        }
    }

    fn on_iter(&mut self, stamp: Instant, iter: u32) {
        self.train_iter_per_s =
            (iter - self.last_train_step.1) as f32 / (stamp - self.last_train_step.0).as_secs_f32();
        self.last_train_step = (stamp, iter);
    }
}

pub struct Viewer {
    device: WgpuDevice,
    adapeter_info: wgpu::AdapterInfo,

    receiver: Option<Receiver<ViewerMessage>>,

    last_message: Option<ViewerMessage>,

    ctx: egui::Context,
    splat_view: SplatView,

    file_path: String,

    target_train_resolution: Option<u32>,
    max_frames: Option<usize>,
    train_state: TrainState,

    constant_redraww: bool,
}

struct TrainArgs {
    frame_count: Option<usize>,
    target_resolution: Option<u32>,
}

async fn load_ply_loop(
    data: &[u8],
    device: WgpuDevice,
    sender: Sender<ViewerMessage>,
    egui_ctx: egui::Context,
) -> anyhow::Result<()> {
    let total_count = splat_import::ply_count(data).context("Invalid ply file")?;

    let splat_stream = splat_import::load_splat_from_ply::<Wgpu>(data, device.clone());

    let mut splat_stream = std::pin::pin!(splat_stream);
    while let Some(splats) = splat_stream.next().await {
        egui_ctx.request_repaint();

        let splats = splats?;
        let msg = ViewerMessage::SplatLoad {
            splats,
            total_count,
        };

        sender
            .send(msg)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send message: {}", e))?;
    }

    Ok(())
}

async fn train_loop(
    data: &[u8],
    device: WgpuDevice,
    sender: Sender<ViewerMessage>,
    egui_ctx: egui::Context,
    train_args: TrainArgs,
    shared_state: Arc<RwLock<SharedTrainState>>,
) -> anyhow::Result<()> {
    let total_steps = 30000;
    let config = TrainConfig::new(
        ExponentialLrSchedulerConfig::new(1.6e-4, 1e-2f64.powf(1.0 / total_steps as f64)),
        RandomSplatsConfig::new(),
    )
    .with_total_steps(total_steps);

    let dataset =
        brush_dataset::read_dataset(data, train_args.frame_count, train_args.target_resolution)?;
    let train_scene = dataset.train.clone();
    let eval_scene = dataset.eval.clone();

    let msg = ViewerMessage::Dataset(dataset);
    sender.send(msg).await.unwrap();

    let visualize = crate::visualize::VisualizeTools::new();
    visualize.log_scene(&train_scene)?;

    let mut splats = config.initial_model_config.init::<Autodiff<Wgpu>>(&device);

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

        let batch = {
            let _span = info_span!("Get batch").entered();
            dataloader.next_batch().await
        };

        let gt_image = batch.gt_images.clone();

        let (new_splats, stats) = trainer
            .step(batch, train_scene.background_color, splats)
            .await
            .unwrap();
        splats = new_splats;

        if trainer.iter % config.visualize_splats_every == 0 {
            visualize.log_train_stats(&splats, &stats, gt_image).await?;
        }

        if let Some(eval_scene) = eval_scene.as_ref() {
            if trainer.iter % config.eval_every == 0 {
                let eval =
                    brush_train::eval::eval_stats(&splats, eval_scene, Some(4), &device).await;
                visualize.log_eval_stats(trainer.iter, &eval)?;
            }
        }

        if trainer.iter % config.visualize_splats_every == 0 {
            visualize.log_splats(&splats).await?;
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

        if trainer.iter == 100 {
            let mem = splats.means.val().into_primitive().tensor().into_primitive().client.memory_usage();
            println!("Memory usage: {}", mem);
        }

        // On wasm, yield to the browser.
        #[cfg(target_arch = "wasm32")]
        gloo_timers::future::TimeoutFuture::new(0).await;
    }

    Ok(())
}

async fn process_loop(
    device: WgpuDevice,
    sender: Sender<ViewerMessage>,
    egui_ctx: egui::Context,
    train_args: TrainArgs,
    shared_state: Arc<RwLock<SharedTrainState>>,
) -> anyhow::Result<()> {
    let picked = rrfd::pick_file().await?;
    if picked.file_name.contains(".ply") {
        load_ply_loop(&picked.data, device, sender, egui_ctx).await
    } else if picked.file_name.contains(".zip") {
        train_loop(
            &picked.data,
            device,
            sender,
            egui_ctx,
            train_args,
            shared_state,
        )
        .await
    } else {
        anyhow::bail!("Only .ply and .zip files are supported.")
    }
}

impl Viewer {
    pub fn new(cc: &eframe::CreationContext) -> Self {
        let state = cc.wgpu_render_state.as_ref().unwrap();

        let adapeter_info = state.adapter.get_info();

        // Run the burn backend on the egui WGPU device.
        let device = burn::backend::wgpu::init_existing_device(
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
            // Splatting workload is much more granular, so don't want to flush as often.
            RuntimeOptions {
                tasks_max: 64,
                memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
            },
        );

        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                use tracing_subscriber::layer::SubscriberExt;

                let subscriber = tracing_subscriber::registry().with(tracing_wasm::WASMLayer::new(Default::default()));
                tracing::subscriber::set_global_default(subscriber)
                    .expect("Failed to set tracing subscriber");
            } else if #[cfg(feature = "tracy")] {
                use tracing_subscriber::layer::SubscriberExt;
                let subscriber = tracing_subscriber::registry()
                    .with(tracing_tracy::TracyLayer::default())
                    .with(sync_span::SyncLayer::new(device.clone()));
                tracing::subscriber::set_global_default(subscriber)
                    .expect("Failed to set tracing subscriber");
            }
        }

        Viewer {
            receiver: None,
            last_message: None,
            adapeter_info,
            train_state: TrainState::new(),
            target_train_resolution: None,
            max_frames: None,
            ctx: cc.egui_ctx.clone(),
            splat_view: SplatView::new(),
            device,
            file_path: "/path/to/file".to_string(),
            constant_redraww: false,
        }
    }

    pub fn start_data_load(&mut self) {
        <Wgpu as burn::prelude::Backend>::seed(42);

        // create a channel for the train loop.
        let (sender, receiver) = async_channel::bounded(2);
        let device = self.device.clone();
        self.receiver = Some(receiver);
        let ctx = self.ctx.clone();

        // Reset view and train state.
        self.splat_view = SplatView::new();
        self.train_state = TrainState::new();

        self.last_message = None;

        async fn inner_process_loop(
            device: WgpuDevice,
            sender: Sender<ViewerMessage>,
            egui_ctx: egui::Context,
            train_args: TrainArgs,
            shared_state: Arc<RwLock<SharedTrainState>>,
        ) {
            match process_loop(device, sender.clone(), egui_ctx, train_args, shared_state).await {
                Ok(_) => (),
                Err(e) => {
                    let _ = sender.send(ViewerMessage::Error(e)).await;
                }
            }
        }

        let train_args = TrainArgs {
            frame_count: self.max_frames,
            target_resolution: self.target_train_resolution,
        };

        let shared_state = self.train_state.shared.clone();

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            pollster::block_on(inner_process_loop(
                device,
                sender,
                ctx,
                train_args,
                shared_state,
            ))
        });

        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(inner_process_loop(
            device,
            sender,
            ctx,
            train_args,
            shared_state,
        ));
    }

    fn url_button(&mut self, label: &str, url: &str, ui: &mut egui::Ui) {
        ui.add(Hyperlink::from_label_and_url(label, url).open_in_new_tab(true));
    }

    fn tracing_debug_ui(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.constant_redraww, "Constant redraw");
        let mut checked = sync_span::is_enabled();
        ui.checkbox(&mut checked, "Sync scopes");
        sync_span::set_enabled(checked);
    }

    fn viewpoints_window(&mut self, ctx: &egui::Context) {
        let train_scene = &self.train_state.dataset.train;

        // Empty scene, nothing to show.
        if train_scene.views.is_empty() {
            return;
        }

        egui::Window::new("Viewpoints")
            .collapsible(true)
            .resizable(true)
            .show(ctx, |ui| {
                let mut nearest_view = train_scene.get_nearest_view(&self.splat_view.camera);

                if let Some(nearest) = nearest_view.as_mut() {
                    let mut buttoned = false;

                    ui.horizontal(|ui| {
                        let view_count = train_scene.views.len();

                        if ui.button("⏪").clicked() {
                            buttoned = true;
                            *nearest = (*nearest + view_count - 1) % view_count;
                        }
                        buttoned |= ui
                            .add(Slider::new(nearest, 0..=train_scene.views.len() - 1))
                            .dragged();
                        if ui.button("⏩").clicked() {
                            buttoned = true;
                            *nearest = (*nearest + 1) % view_count;
                        }
                    });

                    if buttoned {
                        let view = train_scene.views[*nearest].clone();
                        self.splat_view.camera = view.camera.clone();
                        self.splat_view.controls.focus =
                            view.camera.position + view.camera.rotation * glam::Vec3::Z * 5.0;
                    }

                    // Update image if dirty.
                    let mut dirty = self.train_state.selected_view.is_none();
                    if let Some(view) = self.train_state.selected_view.as_ref() {
                        dirty = view.0 != *nearest;
                    }

                    if dirty {
                        let view = train_scene.views[*nearest].clone();
                        let color_img = egui::ColorImage::from_rgb(
                            [view.image.width() as usize, view.image.height() as usize],
                            &view.image.to_rgb8().into_vec(),
                        );
                        self.train_state.selected_view = Some((
                            *nearest,
                            ctx.load_texture(
                                "nearest_view_tex",
                                color_img,
                                TextureOptions::default(),
                            ),
                        ));
                    }

                    if let Some(view) = self.train_state.selected_view.as_ref() {
                        ui.add(egui::Image::new(&view.1).shrink_to_fit());
                        ui.label(&train_scene.views[*nearest].name);
                    }
                }
            });
    }
}

impl eframe::App for Viewer {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if self.constant_redraww {
            ctx.request_repaint();
        }

        let _span = info_span!("Draw UI").entered();

        egui::Window::new("Load data")
            .anchor(egui::Align2::RIGHT_TOP, (0.0, 0.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Select a .ply to visualize, or a .zip with training data.");

                    ui.add_space(15.0);

                    if ui.button("Pick a file").clicked() {
                        self.start_data_load();
                    }
                });

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
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(format!("{}, {:?}", self.adapeter_info.name, self.adapeter_info.device_type));

            ui.add_space(25.0);

            if let Some(rx) = self.receiver.as_mut() {
                match rx.try_recv() {
                    Ok(message) => {
                        if let ViewerMessage::TrainStep {
                            iter, timestamp, ..
                        } = message
                        {
                            self.train_state.on_iter(timestamp, iter);
                        };

                        self.last_message = Some(message)
                    }
                    Err(TryRecvError::Empty) => (), // nothing to do.
                    Err(TryRecvError::Closed) => self.receiver = None, // channel closed.
                }
            }

            #[cfg(feature = "tracing")]
            self.tracing_debug_ui(ui);

            self.viewpoints_window(ctx);

            // Move the dataset if received. This is just to prevent it from having to be
            // cloned in the match below.
            let message = self.last_message.take();
            if let Some(ViewerMessage::Dataset(d)) = message {
                self.train_state.dataset = d;
            } else {
                self.last_message = message;
            }

            if let Some(message) = self.last_message.as_ref() {
                match message {
                    ViewerMessage::Error(e) => {
                        ui.label("Error: ".to_owned() + &e.to_string());
                    }
                    ViewerMessage::SplatLoad {
                        splats,
                        total_count,
                    } => {
                        ui.horizontal(|ui| {
                            ui.label(format!("{} splats", splats.num_splats()));

                            if splats.num_splats() < *total_count {
                                ui.label(format!(
                                    "Loading... ({}%)",
                                    splats.num_splats() as f32 / *total_count as f32 * 100.0
                                ));
                            }
                        });
                        self.splat_view
                            .draw_splats(splats, glam::Vec3::ZERO, ui, ctx, frame);
                    }
                    ViewerMessage::TrainStep {
                        splats,
                        loss,
                        iter,
                        timestamp: _,
                    } => {
                        ui.horizontal(|ui| {
                            ui.label(format!("{} splats", splats.num_splats()));
                            ui.label(format!(
                                "Train step {iter}, {:.1} steps/s",
                                self.train_state.train_iter_per_s
                            ));

                            ui.label(format!("loss: {loss:.3e}"));

                            let mut shared = self.train_state.shared.write().unwrap();
                            let paused = shared.paused;
                            ui.toggle_value(&mut shared.paused, if paused { "⏵" } else { "⏸" });
                        });

                        self.splat_view.draw_splats(
                            splats,
                            self.train_state.dataset.train.background_color,
                            ui,
                            ctx,
                            frame,
                        );
                    }
                    _ => {}
                }
            } else if self.receiver.is_some() {
                ui.label("Loading...");
            }
        });

        if self.splat_view.is_animating() {
            ctx.request_repaint();
        }
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 1.0]
    }
}
