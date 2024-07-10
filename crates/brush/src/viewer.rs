use crate::{
    burn_texture::BurnTexture,
    dataset_readers,
    gaussian_splats::Splats,
    orbit_controls::OrbitControls,
    scene::{self, SceneBatcher, SceneLoader},
    splat_import,
    train::{LrConfig, SplatTrainer, TrainConfig},
};
use async_channel::{Receiver, Sender, TryRecvError};
use brush_render::{camera::Camera, sync_span::SyncSpan};
use burn::{backend::Autodiff, module::AutodiffModule};
use burn_wgpu::{JitBackend, RuntimeOptions, WgpuDevice, WgpuRuntime};
use egui::{pos2, CollapsingHeader, Color32, Rect};
use futures_lite::StreamExt;
use glam::{Mat4, Quat, Vec2, Vec3};

use rfd::AsyncFileDialog;
use tracing::info_span;
use wgpu::CommandEncoderDescriptor;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

type Backend = JitBackend<WgpuRuntime, f32, i32>;

enum ViewerMessage {
    Initial,
    Error(anyhow::Error),
    SplatLoad {
        splats: Splats<Backend>,
        total_count: usize,
    },
    TrainStep {
        splats: Splats<Backend>,
        iter: u32,
    },
}

pub struct Viewer {
    receiver: Option<Receiver<ViewerMessage>>,
    last_message: Option<ViewerMessage>,

    last_train_iter: u32,
    ctx: egui::Context,

    device: WgpuDevice,

    splat_view: SplatView,
}

struct SplatView {
    camera: Camera,
    start_transform: Mat4,

    controls: OrbitControls,
    backbuffer: Option<BurnTexture>,
}

async fn train_loop(
    config: TrainConfig,
    device: WgpuDevice,
    updater: Sender<ViewerMessage>,
    egui_ctx: egui::Context,
) {
    #[cfg(feature = "rerun")]
    let rec = rerun::RecordingStreamBuilder::new("visualize training")
        .spawn()
        .expect("Failed to start rerun");

    let file = AsyncFileDialog::new()
        .add_filter("scene", &["ply", "zip"])
        .set_directory("/")
        .pick_file()
        .await;

    let Some(file) = file else {
        return;
    };

    let file_data = file.read().await;

    if file.file_name().contains(".ply") {
        let total_count = splat_import::ply_count(&file_data);

        let Ok(total_count) = total_count else {
            let _ = updater
                .send(ViewerMessage::Error(anyhow::anyhow!("Invalid ply file")))
                .await;
            return;
        };

        let splat_stream = splat_import::load_splat_from_ply::<Backend>(&file_data, device.clone());

        // let splats =
        // TODO: Send over the channel? Or whats good here?
        // let (tx, rx) = single_value_channel::channel();
        // self.reference_cameras =
        //     reference_view.and_then(|s| dataset_readers::read_viewpoint_data(s).ok());
        // if let Some(refs) = self.reference_cameras.as_ref() {
        //     self.camera = refs[0].clone();
        // }

        let mut splat_stream = std::pin::pin!(splat_stream);
        while let Some(splats) = splat_stream.next().await {
            egui_ctx.request_repaint();

            match splats {
                Ok(splats) => {
                    let msg = ViewerMessage::SplatLoad {
                        splats,
                        total_count,
                    };

                    if updater.send(msg).await.is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let _ = updater.send(ViewerMessage::Error(e)).await.is_err();
                    return;
                }
            }
        }
    } else {
        let cameras = dataset_readers::read_synthetic_nerf_data(&file_data, None).unwrap();
        let scene = scene::Scene::new(cameras);

        #[cfg(feature = "rerun")]
        scene.visualize(&rec).expect("Failed to visualize scene");

        let mut splats =
            Splats::<Autodiff<Backend>>::init_random(config.init_splat_count, 2.0, &device);

        let batcher_train = SceneBatcher::<Autodiff<Backend>>::new(device.clone());
        let mut dataloader = SceneLoader::new(scene, batcher_train, 1);
        let mut trainer = SplatTrainer::new(splats.num_splats(), &config, &splats);

        loop {
            let batch = {
                let _span = info_span!("Get batch").entered();
                dataloader.next()
            };

            splats = trainer
                .step(
                    batch,
                    splats,
                    #[cfg(feature = "rerun")]
                    &rec,
                )
                .await
                .unwrap();

            if trainer.iter % 5 == 0 {
                let _span = info_span!("Send batch").entered();

                egui_ctx.request_repaint();
                let msg = ViewerMessage::TrainStep {
                    splats: splats.valid(),
                    iter: trainer.iter,
                };

                if updater.send(msg).await.is_err() {
                    break; // channel closed, bail.
                }
            }
        }
    }
}

impl SplatView {
    fn draw_splats(
        &mut self,
        splats: &Splats<Backend>,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        frame: &mut eframe::Frame,
    ) {
        CollapsingHeader::new("View Splats")
            .default_open(true)
            .show(ui, |ui| {
                let size = ui.available_size();

                // Round to 64 pixels for buffer alignment.
                let size = glam::uvec2(
                    ((size.x as u32).div_ceil(64) * 64).max(32),
                    (size.y as u32).max(32),
                );

                let (rect, response) = ui.allocate_exact_size(
                    egui::Vec2::new(size.x as f32, size.y as f32),
                    egui::Sense::drag(),
                );

                let mouse_delta = glam::vec2(response.drag_delta().x, response.drag_delta().y);

                let (pan, rotate) = if response.dragged_by(egui::PointerButton::Primary) {
                    (Vec2::ZERO, mouse_delta)
                } else if response.dragged_by(egui::PointerButton::Secondary) {
                    (mouse_delta, Vec2::ZERO)
                } else {
                    (Vec2::ZERO, Vec2::ZERO)
                };

                let scrolled = ui.input(|r| r.smooth_scroll_delta).y;

                // TODO: Controls can be pretty borked.
                self.controls.pan_orbit_camera(
                    pan * 0.5,
                    rotate * 0.5,
                    scrolled * 0.02,
                    glam::vec2(rect.size().x, rect.size().y),
                );

                let controls_transform = glam::Mat4::from_rotation_translation(
                    self.controls.rotation,
                    self.controls.position,
                );

                let total_transform = self.start_transform * controls_transform;
                let (_, rot, pos) = total_transform.to_scale_rotation_translation();
                self.camera.position = pos;
                self.camera.rotation = rot;

                // TODO: For reference cameras just need to match fov?
                self.camera.fovx = 0.5;
                self.camera.fovy = 0.5 * (size.y as f32) / (size.x as f32);

                // If there's actual rendering to do, not just an imgui update.
                if ctx.has_requested_repaint() {
                    // Check whether any work needs to be flushed.
                    {
                        let device = &splats.means.device();
                        let _span = SyncSpan::<Backend>::new("pre setup", device);
                    }

                    let (img, _) = {
                        let _span = info_span!("Render splats").entered();
                        splats.render(&self.camera, size, glam::vec3(0.0, 0.0, 0.0), true)
                    };

                    let back = self
                        .backbuffer
                        .get_or_insert_with(|| BurnTexture::new(img.clone(), frame));

                    {
                        let state = frame.wgpu_render_state();
                        let state = state.as_ref().unwrap();
                        let mut encoder =
                            state
                                .device
                                .create_command_encoder(&CommandEncoderDescriptor {
                                    label: Some("viewer encoder"),
                                });
                        back.update_texture(img, frame, &mut encoder);
                        let cmd = encoder.finish();
                        state.queue.submit([cmd]);
                    }
                }

                if let Some(back) = self.backbuffer.as_ref() {
                    ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
                    ui.painter().image(
                        back.id,
                        rect,
                        Rect {
                            min: pos2(0.0, 0.0),
                            max: pos2(1.0, 1.0),
                        },
                        Color32::WHITE,
                    );
                }
            });
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
            // Splatting workload is much more granular, so don't want to flush nearly as often.
            RuntimeOptions { tasks_max: 1024 },
        );

        Viewer {
            receiver: None,
            last_message: None,
            last_train_iter: 0,
            ctx: cc.egui_ctx.clone(),
            splat_view: SplatView {
                camera: Camera::new(Vec3::ZERO, Quat::IDENTITY, 0.5, 0.5),
                backbuffer: None,
                controls: OrbitControls::new(15.0),
                start_transform: glam::Mat4::IDENTITY,
            },
            device,
        }
    }

    fn stop_training(&mut self) {
        self.receiver = None; // This drops the receiver, which closes the channel.
    }

    pub fn start_training(&mut self) {
        <Backend as burn::prelude::Backend>::seed(42);

        let config = TrainConfig::new(
            LrConfig::new().with_max_lr(2e-5).with_min_lr(5e-6),
            LrConfig::new().with_max_lr(4e-2).with_min_lr(2e-2),
            LrConfig::new().with_max_lr(1e-2).with_min_lr(6e-3),
        );

        // create a channel for the train loop.
        let (sender, receiver) = async_channel::bounded(3);
        let device = self.device.clone();
        self.receiver = Some(receiver);
        let ctx = self.ctx.clone();

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || pollster::block_on(train_loop(config, device, sender, ctx)));

        #[cfg(target_arch = "wasm32")]
        spawn_local(train_loop(config, device, sender, ctx));
    }
}

impl eframe::App for Viewer {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let _span = info_span!("Draw UI").entered();

        egui_extras::install_image_loaders(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Brush splat viewer");

            if ui.button("Load file").clicked() {
                self.start_training();
            }
            if let Some(rx) = self.receiver.as_mut() {
                match rx.try_recv() {
                    Ok(message) => self.last_message = Some(message),
                    Err(TryRecvError::Empty) => (), // nothing to do.
                    Err(TryRecvError::Closed) => self.receiver = None, // channel closed.
                }
            }

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
                        if splats.num_splats() < *total_count {
                            ui.label(format!(
                                "Loading... ({}/{total_count} splats)",
                                splats.num_splats()
                            ));
                        }
                        self.splat_view.draw_splats(splats, ui, ctx, frame);
                    }
                    ViewerMessage::TrainStep { splats, iter } => {
                        ui.label(format!("Train step {iter}"));
                        self.splat_view.draw_splats(splats, ui, ctx, frame);
                    }
                }
            }
        });
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 1.0]
    }
}
