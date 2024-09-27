use anyhow::Context;
use async_channel::{Receiver, Sender, TryRecvError, TrySendError};
use brush_dataset::scene_batch::SceneLoader;
use brush_render::gaussian_splats::Splats;
use burn::tensor::ElementConversion;
use burn::{backend::Autodiff, module::AutodiffModule};
use burn_wgpu::{JitBackend, RuntimeOptions, WgpuDevice, WgpuRuntime};
use egui::{Hyperlink, Slider, TextureOptions};
use futures_lite::StreamExt;

use tracing::info_span;
use web_time::Instant;

use brush_dataset;
use brush_train::scene::{Scene, SceneView};
use brush_train::train::{LrConfig, SplatTrainer, TrainConfig};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

use crate::splat_import;
use crate::splat_view::SplatView;

type Backend = JitBackend<WgpuRuntime, f32, i32>;

enum ViewerMessage {
    Initial,
    Error(anyhow::Error),

    // Initial splat cloud to be created.
    SplatLoad {
        splats: Splats<Backend>,
        total_count: usize,
    },

    // Loaded a bunch of viewpoints to train on.
    Viewpoints(Vec<SceneView>),

    TrainStep {
        splats: Splats<Backend>,
        loss: f32,
        iter: u32,
        timestamp: Instant,
    },
}

struct TrainState {
    last_train_step: (Instant, u32),
    train_iter_per_s: f32,
    paused: bool,
    viewpoints: Vec<SceneView>,
    selected_view: Option<egui::TextureHandle>,
}

impl TrainState {
    fn new() -> Self {
        Self {
            last_train_step: (Instant::now(), 0),
            train_iter_per_s: 0.0,
            paused: false,
            viewpoints: Vec::new(),
            selected_view: None,
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

    receiver: Option<Receiver<ViewerMessage>>,
    last_message: Option<ViewerMessage>,

    ctx: egui::Context,
    splat_view: SplatView,

    file_path: String,
    target_train_resolution: u32,
    max_frames: usize,

    constant_redraww: bool,

    train_state: TrainState,
}

struct TrainArgs {
    frame_count: usize,
    target_resolution: u32,
}

async fn load_ply_loop(
    data: &[u8],
    device: WgpuDevice,
    sender: Sender<ViewerMessage>,
    egui_ctx: egui::Context,
) -> anyhow::Result<()> {
    let total_count = splat_import::ply_count(data).context("Invalid ply file")?;

    let splat_stream = splat_import::load_splat_from_ply::<Backend>(data, device.clone());

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
) -> anyhow::Result<()> {
    let config = TrainConfig::new(
        LrConfig::new().with_max_lr(4e-5).with_min_lr(2e-5),
        LrConfig::new().with_max_lr(8e-2).with_min_lr(2e-2),
        LrConfig::new().with_max_lr(2e-2).with_min_lr(1e-2),
    );

    let views = brush_dataset::read_dataset(
        data,
        Some(train_args.frame_count),
        Some(train_args.target_resolution),
    )?;
    let msg = ViewerMessage::Viewpoints(views.clone());
    sender.send(msg).await.unwrap();

    let scene = Scene::new(views);

    #[cfg(feature = "rerun")]
    {
        let visualize = crate::visualize::VisualizeTools::new();
        visualize.log_scene(&scene)?;
    }

    let mut splats =
        Splats::<Autodiff<Backend>>::init_random(config.init_splat_count, 2.0, &device);

    let mut dataloader = SceneLoader::new(scene, &device, 1);
    let mut trainer = SplatTrainer::new(splats.num_splats(), &config, &splats);

    loop {
        let batch = {
            let _span = info_span!("Get batch").entered();
            dataloader.next_batch()
        };

        #[cfg(feature = "rerun")]
        let gt_image = batch.gt_images.clone();

        let (new_splats, stats) = trainer.step(batch, splats).await.unwrap();

        #[cfg(feature = "rerun")]
        {
            if trainer.iter % config.visualize_splats_every == 0 {
                visualize
                    .log_train_stats(&new_splats, &stats, gt_image)
                    .await?;
            }

            if trainer.iter % config.visualize_splats_every == 0 {
                visualize.log_splats(&new_splats).await?;
            }
        }

        splats = new_splats;

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

async fn process_loop(
    device: WgpuDevice,
    sender: Sender<ViewerMessage>,
    egui_ctx: egui::Context,
    train_args: TrainArgs,
) -> anyhow::Result<()> {
    let picked = rrfd::pick_file().await?;
    if picked.file_name.contains(".ply") {
        load_ply_loop(&picked.data, device, sender, egui_ctx).await
    } else if picked.file_name.contains(".zip") {
        train_loop(&picked.data, device, sender, egui_ctx, train_args).await
    } else {
        anyhow::bail!("Only .ply and .zip files are supported.")
    }
}

impl Viewer {
    pub fn new(cc: &eframe::CreationContext) -> Self {
        let state = cc.wgpu_render_state.as_ref().unwrap();

        // Run the burn backend on the egui WGPU device.
        let device = burn::backend::wgpu::init_existing_device(
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
            // Splatting workload is much more granular, so don't want to flush as often.
            RuntimeOptions { tasks_max: 64 },
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
            train_state: TrainState::new(),
            ctx: cc.egui_ctx.clone(),
            splat_view: SplatView::new(),
            device,
            file_path: "/path/to/file".to_string(),
            target_train_resolution: 800,
            max_frames: 32,
            constant_redraww: false,
        }
    }

    pub fn start_data_load(&mut self) {
        <Backend as burn::prelude::Backend>::seed(42);

        // create a channel for the train loop.
        let (sender, receiver) = async_channel::bounded(2);
        let device = self.device.clone();
        self.receiver = Some(receiver);
        let ctx = self.ctx.clone();

        // Reset view and train state.
        self.splat_view = SplatView::new();
        self.train_state = TrainState::new();

        async fn inner_process_loop(
            device: WgpuDevice,
            sender: Sender<ViewerMessage>,
            egui_ctx: egui::Context,
            train_args: TrainArgs,
        ) {
            match process_loop(device, sender.clone(), egui_ctx, train_args).await {
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

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            pollster::block_on(inner_process_loop(device, sender, ctx, train_args))
        });

        #[cfg(target_arch = "wasm32")]
        spawn_local(inner_process_loop(device, sender, ctx, train_args));
    }

    fn url_button(&mut self, label: &str, url: &str, ui: &mut egui::Ui) {
        ui.add(Hyperlink::from_label_and_url(label, url).open_in_new_tab(true));
    }

    fn tracy_debug_ui(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.constant_redraww, "Constant redraw");
        let mut checked = sync_span::is_enabled();
        ui.checkbox(&mut checked, "Sync scopes");
        sync_span::set_enabled(checked);
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
                ui.add(
                    Slider::new(&mut self.target_train_resolution, 32..=2048)
                        .text("Target train resolution"),
                );
                ui.add(Slider::new(&mut self.max_frames, 1..=256).text("Max frames"));

                ui.add_space(15.0);

                if ui.input(|r| r.key_pressed(egui::Key::Escape)) {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(25.0);

            if let Some(rx) = self.receiver.as_mut() {
                if !self.train_state.paused {
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
            }

            #[cfg(feature = "tracy")]
            self.tracy_debug_ui(ui);

            if let Some(message) = self.last_message.as_ref() {
                match message {
                    ViewerMessage::Initial => (),
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
                        self.splat_view.draw_splats(splats, ui, ctx, frame);
                    }
                    ViewerMessage::Viewpoints(vec) => self.train_state.viewpoints = vec.clone(),
                    ViewerMessage::TrainStep {
                        splats,
                        loss,
                        iter,
                        timestamp: _,
                    } => {
                        egui::Window::new("Viewpoints")
                            .collapsible(true)
                            .show(ctx, |ui| {
                                ui.horizontal(|ui| {
                                    // egui::ScrollArea::vertical().show(ui, |ui| {
                                    ui.vertical(|ui| {
                                        for (i, view) in
                                            self.train_state.viewpoints.iter().enumerate()
                                        {
                                            if ui.button(&format!("View {i}")).clicked() {
                                                self.splat_view.camera = view.camera.clone();
                                                let color_img =
                                                    egui::ColorImage::from_rgba_unmultiplied(
                                                        [
                                                            view.image.width() as usize,
                                                            view.image.height() as usize,
                                                        ],
                                                        view.image.as_bytes(),
                                                    );
                                                self.train_state.selected_view =
                                                    Some(ctx.load_texture(
                                                        "Debug",
                                                        color_img,
                                                        TextureOptions::default(),
                                                    ));
                                            }
                                        }
                                    });
                                    // });

                                    if let Some(view) = &self.train_state.selected_view {
                                        ui.add(egui::Image::new(view));
                                    }
                                });
                            });

                        ui.horizontal(|ui| {
                            ui.label(format!("{} splats", splats.num_splats()));
                            ui.label(format!(
                                "Train step {iter}, {:.1} steps/s",
                                self.train_state.train_iter_per_s
                            ));

                            ui.label(format!("loss: {loss:.3e}"));

                            let paused = self.train_state.paused;
                            ui.toggle_value(
                                &mut self.train_state.paused,
                                if paused { "⏵" } else { "⏸" },
                            );
                        });

                        self.splat_view.draw_splats(splats, ui, ctx, frame);
                    }
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
