use crate::{
    burn_texture::BurnTexture,
    dataset_readers,
    gaussian_splats::Splats,
    orbit_controls::OrbitControls,
    scene::{SceneBatcher, SceneLoader},
    splat_import,
    train::{LrConfig, SplatTrainer, TrainConfig},
};
use brush_render::camera::Camera;
use burn::{backend::Autodiff, module::AutodiffModule};
use burn_wgpu::{JitBackend, RuntimeOptions, WgpuDevice, WgpuRuntime};
use egui::{pos2, CollapsingHeader, Color32, Rect};
use glam::{Mat4, Quat, Vec2, Vec3};

use tracing::info_span;
use wgpu::CommandEncoderDescriptor;

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
use std::{
    sync::mpsc::{self, Receiver, SyncSender, TrySendError},
    thread,
};

#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

type Backend = JitBackend<WgpuRuntime, f32, i32>;

struct TrainUpdate {
    pub splats: Splats<Backend>,
    pub iter: u32,
}

pub struct Viewer {
    camera: Camera,
    receiver: Option<Receiver<TrainUpdate>>,

    train_iter: u32,
    start_train_time: Instant,

    ctx: egui::Context,

    render_splats: Option<Splats<Backend>>,
    reference_cameras: Option<Vec<Camera>>,
    backbuffer: Option<BurnTexture>,
    controls: OrbitControls,
    device: WgpuDevice,
    start_transform: Mat4,
}

fn train_loop(config: &TrainConfig, device: &WgpuDevice, tx: SyncSender<TrainUpdate>, egui_ctx: egui::Context) {
    #[cfg(feature = "rerun")]
    let rec = rerun::RecordingStreamBuilder::new("visualize training")
        .spawn()
        .expect("Failed to start rerun");

    println!(
        "Loading dataset {:?}, {}",
        std::env::current_dir(),
        config.scene_path
    );
    let scene = dataset_readers::read_scene(&config.scene_path, "transforms_train.json", None);

    #[cfg(feature = "rerun")]
    scene.visualize(&rec).expect("Failed to visualize scene");

    let batcher_train = SceneBatcher::<Autodiff<Backend>>::new(device.clone());

    let mut splats = Splats::<Autodiff<Backend>>::init_random(config.init_splat_count, 2.0, device);
    let mut dataloader = SceneLoader::new(scene, batcher_train, 2);
    let mut trainer = SplatTrainer::new(splats.num_splats(), config, &splats);

    loop {
        // TODO: On non wasm this really should be threaded.
        let get_span = info_span!("Get batch").entered();
        let batch = dataloader.next();
        drop(get_span);

        splats = trainer
            .step(
                batch,
                splats,
                #[cfg(feature = "rerun")]
                &rec,
            )
            .unwrap();

        if trainer.iter % 10 == 0 {
            // Ideally this would drop the old value and set the new value to be consume - that somehow doesn't
            // seem to be available for channels. I guess modelling shared ownership like that is more like a
            // mutex than a channel - but the channel interface is nicer.

            let update = TrainUpdate {
                splats: splats.valid(),
                iter: trainer.iter,
            };

            match tx.try_send(update) {
                Ok(_) => egui_ctx.request_repaint(),
                // If full, just ignore this and don't send anything.
                Err(TrySendError::Full(_)) => (),
                // On a disconnect, we're done.
                Err(TrySendError::Disconnected(_)) => break,
            };
        }
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
            // Burn atm has pretty suboptimal memory management leading to weird behaviour. This effectively disables
            // it, with a GC run every couple seconds. Seems good enough for now.
            // (nb: it doesn't mean burn will never re-use memory, just that it hangs on to
            // GPU allocations for longer).
            RuntimeOptions {
                dealloc_strategy:
                    burn_compute::memory_management::simple::DeallocStrategy::PeriodTime {
                        period: Duration::from_secs(5),
                        state: Instant::now(),
                    },
                tasks_max: 128,
                ..Default::default()
            },
        );

        Viewer {
            receiver: None,
            train_iter: 0,
            start_train_time: Instant::now(),
            ctx: cc.egui_ctx.clone(),
            camera: Camera::new(Vec3::ZERO, Quat::IDENTITY, 0.5, 0.5),
            render_splats: None,
            reference_cameras: None,
            backbuffer: None,
            controls: OrbitControls::new(15.0),
            device,
            start_transform: glam::Mat4::IDENTITY,
        }
    }

    fn stop_training(&mut self) {
        self.receiver = None; // This drops the receiver, which closes the channel.
    }

    pub fn load_splats(&mut self, path: &str, reference_view: Option<&str>) {
        self.stop_training();

        self.reference_cameras =
            reference_view.and_then(|s| dataset_readers::read_viewpoint_data(s).ok());

        if let Some(refs) = self.reference_cameras.as_ref() {
            self.camera = refs[0].clone();
        }

        self.render_splats = splat_import::load_splat_from_ply(path, &self.device).ok();
    }

    pub fn start_training(&mut self, path: &str) {
        self.start_train_time = Instant::now();

        <Backend as burn::prelude::Backend>::seed(42);

        let config = TrainConfig::new(
            LrConfig::new().with_max_lr(2e-5).with_min_lr(5e-6),
            LrConfig::new().with_max_lr(4e-2).with_min_lr(2e-2),
            LrConfig::new().with_max_lr(1e-2).with_min_lr(6e-3),
            path.to_owned(),
        );

        // create a channel for the train loop.
        let (tx, rx) = mpsc::sync_channel(1);
        let device = self.device.clone();
        self.receiver = Some(rx);
        let ctx = self.ctx.clone();
        thread::spawn(move || train_loop(&config, &device, tx, ctx)); 
    }
}

impl eframe::App for Viewer {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui_extras::install_image_loaders(ctx);

        if let Some(cameras) = self.reference_cameras.as_ref() {
            egui::SidePanel::new(egui::panel::Side::Left, "cam panel").show(ctx, |ui| {
                for (i, camera) in cameras.iter().enumerate() {
                    if ui.button(format!("Camera {i}")).clicked() {
                        self.camera = camera.clone();
                        self.controls.position = self.camera.position;
                        self.controls.rotation = self.camera.rotation;
                    }
                }
            });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Brush splat viewer");
            ui.label("load pretrained models.");
            ui.horizontal(|ui| {
                for r in ["bonsai", "stump", "counter", "garden", "truck"] {
                    if ui.button(r.to_string()).clicked() {
                        self.load_splats(
                            &format!("./brush_data/pretrained/{r}/point_cloud/iteration_30000/point_cloud.ply"),
                            Some(&format!("./brush_data/pretrained/{r}/cameras.json")),
                        );
                    }
                }
            });

            ui.label("Train a new model.");
            ui.horizontal(|ui| {
                for r in ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"] {
                    if ui.button(r.to_string()).clicked() {
                        self.start_training(&format!("./brush_data/nerf_synthetic/{r}/"))
                    }
                }
            });

            if let Some(rx) = self.receiver.as_ref() {
                ui.label(format!("Training step {}", self.train_iter));

                let steps_per_second = (self.train_iter as f64) / (Instant::now() - self.start_train_time).as_secs_f64();
                ui.label(format!("Steps/s {:.1}", steps_per_second));
                
                // Update splats if a new step has arrived.
                match rx.try_recv() {
                    Ok(update) => { 
                        self.render_splats = Some(update.splats);
                        self.train_iter = update.iter;
                    },
                    Err(mpsc::TryRecvError::Empty) => (), // fine - just keep waiting.
                    Err(mpsc::TryRecvError::Disconnected) => panic!("Channel broke somehow."),
                }
            }

            if let Some(splats) = &self.render_splats {
                CollapsingHeader::new("View Splats").default_open(true).show(ui, |ui| {
                    let size = ui.available_size();

                    // Round to 64 pixels for buffer alignment.
                    let size = glam::uvec2(((size.x as u32).div_ceil(64) * 64).max(32), (size.y as u32).max(32));

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
                        let span = info_span!("Render splats").entered();
                        let (img, _) =
                            splats.clone().render(&self.camera, size, glam::vec3(0.0, 0.0, 0.0), true);
                        drop(span);

                        let back = self.backbuffer.get_or_insert_with(|| BurnTexture::new(img.clone(), frame));

                        {
                            let state = frame.wgpu_render_state();
                            let state = state.as_ref().unwrap();
                            let mut encoder = state
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
        });
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 1.0]
    }
}
