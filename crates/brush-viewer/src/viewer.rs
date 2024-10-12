use std::sync::{Arc, RwLock};

use anyhow::Context;
use async_channel::{Receiver, Sender, TryRecvError};
use brush_render::gaussian_splats::Splats;
use brush_render::PrimaryBackend;
use brush_train::scene::ViewType;
use burn_wgpu::{RuntimeOptions, Wgpu, WgpuDevice};
use egui::{Hyperlink, Slider, TextureOptions};
use futures_lite::StreamExt;

use tracing::trace_span;
use web_time::Instant;

use brush_dataset::{self, Dataset};

use crate::splat_view::SplatView;
use crate::train_loop::{SharedTrainState, TrainArgs, TrainState};
use crate::{splat_import, train_loop};

pub(crate) enum ViewerMessage {
    Error(anyhow::Error),

    // Initial splat cloud to be created.
    SplatLoad {
        splats: Splats<PrimaryBackend>,
        total_count: usize,
    },

    // Loaded a bunch of viewpoints to train on.
    Dataset(Dataset),

    TrainStep {
        splats: Splats<PrimaryBackend>,
        loss: f32,
        iter: u32,
        timestamp: Instant,
    },
}

pub struct Viewer {
    device: WgpuDevice,
    adapeter_info: wgpu::AdapterInfo,

    receiver: Option<Receiver<ViewerMessage>>,

    last_message: Option<ViewerMessage>,

    ctx: egui::Context,
    splat_view: SplatView,

    target_train_resolution: Option<u32>,
    max_frames: Option<usize>,
    train_state: TrainState,
    viewpoints_view_type: ViewType,
    constant_redraww: bool,
}

async fn load_ply_loop(
    data: &[u8],
    device: WgpuDevice,
    sender: Sender<ViewerMessage>,
    egui_ctx: egui::Context,
) -> anyhow::Result<()> {
    let total_count = splat_import::ply_count(data).context("Invalid ply file")?;

    let splat_stream = splat_import::load_splat_from_ply::<PrimaryBackend>(data, device.clone());

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
        train_loop::train_loop(
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
            constant_redraww: false,
            viewpoints_view_type: ViewType::Train,
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
            futures_lite::future::block_on(inner_process_loop(
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
        let dataset = &self.train_state.dataset;

        // Empty scene, nothing to show.
        if dataset.train_scene().view_count() == 0 {
            return;
        }

        egui::Window::new("Viewpoints")
            .collapsible(true)
            .resizable(true)
            .show(ctx, |ui| {
                let scene = if let Some(eval_scene) = dataset.eval_scene() {
                    match self.viewpoints_view_type {
                        ViewType::Train => dataset.train_scene(),
                        _ => eval_scene,
                    }
                } else {
                    dataset.train_scene()
                };

                let mut nearest_view = scene.get_nearest_view(&self.splat_view.camera);

                let Some(nearest) = nearest_view.as_mut() else {
                    return;
                };

                // Update image if dirty.
                let mut dirty = self.train_state.selected_view.is_none();

                if let Some(view) = self.train_state.selected_view.as_ref() {
                    dirty |= view.0 != *nearest;
                }

                let mut buttoned = false;
                let view_count = scene.view_count();
                let out_of = format!("/ {view_count}");

                ui.horizontal(|ui| {
                    if ui.button("⏪").clicked() {
                        buttoned = true;
                        *nearest -= 1;
                    }
                    buttoned |= ui
                        .add(Slider::new(nearest, 0..=scene.view_count() - 1).suffix(out_of))
                        .dragged();
                    if ui.button("⏩").clicked() {
                        buttoned = true;
                        *nearest += 1;
                    }

                    ui.add_space(10.0);

                    if dataset.eval_scene().is_some() {
                        for (t, l) in [ViewType::Train, ViewType::Eval]
                            .into_iter()
                            .zip(["train", "eval"])
                        {
                            if ui
                                .selectable_label(self.viewpoints_view_type == t, l)
                                .clicked()
                            {
                                self.viewpoints_view_type = t;
                                dirty = true;
                            };
                        }
                    }
                });

                if buttoned {
                    let view = scene.get_view(*nearest).unwrap().clone();
                    self.splat_view.camera = view.camera.clone();
                    self.splat_view.controls.focus =
                        view.camera.position + view.camera.rotation * glam::Vec3::Z * 5.0;
                }

                if dirty {
                    let view = scene.get_view(*nearest).unwrap().clone();
                    let color_img = egui::ColorImage::from_rgb(
                        [view.image.width() as usize, view.image.height() as usize],
                        &view.image.to_rgb8().into_vec(),
                    );
                    self.train_state.selected_view = Some((
                        *nearest,
                        ctx.load_texture("nearest_view_tex", color_img, TextureOptions::default()),
                    ));
                }

                if let Some(view) = self.train_state.selected_view.as_ref() {
                    ui.add(egui::Image::new(&view.1).shrink_to_fit());
                    ui.label(&scene.get_view(*nearest).unwrap().name);
                }
            });
    }
}

impl eframe::App for Viewer {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if self.constant_redraww {
            ctx.request_repaint();
        }

        let _span = trace_span!("Draw UI").entered();

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
            ui.label(format!(
                "{}, {:?}",
                self.adapeter_info.name, self.adapeter_info.device_type
            ));

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
                            self.train_state.dataset.train_scene().background,
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
