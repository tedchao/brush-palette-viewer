use crate::{
    burn_texture::BurnTexture,
    dataset_readers,
    gaussian_splats::Splats,
    orbit_controls::OrbitControls,
    scene::{self, SceneBatcher, SceneLoader},
    splat_import,
    train::{self, LrConfig, SplatTrainer, TrainConfig},
};
use async_channel::{Receiver, Sender, TryRecvError, TrySendError};
use brush_render::{camera::Camera, sync_span::SyncSpan};
use burn::{backend::Autodiff, data, module::AutodiffModule, tensor::ElementConversion};
use burn_wgpu::{JitBackend, RuntimeOptions, WgpuDevice, WgpuRuntime};
use egui::{pos2, CollapsingHeader, Color32, Hyperlink, Rect};
use futures_lite::StreamExt;
use glam::{Quat, Vec2, Vec3};

// use rfd::AsyncFileDialog;
use tracing::info_span;
use web_time::Instant;
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
        loss: f32,
        iter: u32,
        timestamp: Instant,
    },
}

pub struct Viewer {
    receiver: Option<Receiver<ViewerMessage>>,
    last_message: Option<ViewerMessage>,

    last_train_iter: u32,
    ctx: egui::Context,

    device: WgpuDevice,

    splat_view: SplatView,

    train_iter_per_s: f32,
    last_train_step: (Instant, u32),
    train_pause: bool,
}

struct SplatView {
    camera: Camera,
    controls: OrbitControls,
    backbuffer: Option<BurnTexture>,
    last_draw: Instant,
}

async fn train_loop(
    device: WgpuDevice,
    updater: Sender<ViewerMessage>,
    egui_ctx: egui::Context,
    data_url: String,
) {
    #[cfg(feature = "rerun")]
    let rec = rerun::RecordingStreamBuilder::new("visualize training")
        .spawn()
        .expect("Failed to start rerun");

    let mut file_data = vec![];
    let _ = ureq::get(&data_url)
        .call()
        .unwrap()
        .into_reader()
        .read_to_end(&mut file_data)
        .unwrap();

    // let file = AsyncFileDialog::new()
    //     .add_filter("scene", &["ply", "zip"])
    //     .set_directory("/")
    //     .pick_file()
    //     .await;

    // let Some(file) = file else {
    //     return;
    // };

    // let file_data = file.read().await;

    if data_url.contains(".ply") {
        let total_count = splat_import::ply_count(&file_data);

        let Ok(total_count) = total_count else {
            let _ = updater
                .send(ViewerMessage::Error(anyhow::anyhow!("Invalid ply file")))
                .await;
            return;
        };

        let splat_stream = splat_import::load_splat_from_ply::<Backend>(&file_data, device.clone());

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
        let config = TrainConfig::new(
            LrConfig::new().with_max_lr(2e-5).with_min_lr(1e-5),
            LrConfig::new().with_max_lr(4e-2).with_min_lr(1e-2),
            LrConfig::new().with_max_lr(1e-2).with_min_lr(5e-3),
        );

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

            let (new_splats, loss) = trainer
                .step(
                    batch,
                    splats,
                    #[cfg(feature = "rerun")]
                    &rec,
                )
                .await
                .unwrap();
            splats = new_splats;
            let _ = train::yield_macro::<Backend>(&device).await;

            if trainer.iter % 4 == 0 {
                let _span = info_span!("Send batch").entered();

                let _ = train::yield_macro::<Backend>(&device).await;

                let msg = ViewerMessage::TrainStep {
                    splats: splats.valid(),
                    loss: loss.into_scalar_async().await.elem::<f32>(),
                    iter: trainer.iter,
                    timestamp: Instant::now(),
                };

                match updater.send(msg).await {
                    Ok(_) => (),
                    // Err(TrySendError::Full(_)) => (),
                    Err(_) => {
                        break; // channel closed, bail.
                    }
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
                let size =
                    glam::uvec2(((size.x as u32 / 64) * 64).max(32), (size.y as u32).max(32));

                let (rect, response) = ui.allocate_exact_size(
                    egui::Vec2::new(size.x as f32, size.y as f32),
                    egui::Sense::drag(),
                );

                let mouse_delta = glam::vec2(response.drag_delta().x, response.drag_delta().y);

                let (pan, rotate) = if response.dragged_by(egui::PointerButton::Primary) {
                    (Vec2::ZERO, mouse_delta)
                } else if response.dragged_by(egui::PointerButton::Secondary)
                    || response.dragged_by(egui::PointerButton::Middle)
                {
                    (mouse_delta, Vec2::ZERO)
                } else {
                    (Vec2::ZERO, Vec2::ZERO)
                };

                let scrolled = ui.input(|r| r.smooth_scroll_delta).y;
                let cur_time = Instant::now();
                let delta_time = cur_time - self.last_draw;
                self.last_draw = cur_time;

                // TODO: Controls can be pretty borked.
                self.controls.pan_orbit_camera(
                    &mut self.camera,
                    pan * 5.0,
                    rotate * 5.0,
                    scrolled * 0.01,
                    glam::vec2(rect.size().x, rect.size().y),
                    delta_time.as_secs_f32(),
                );

                // TODO: For reference cameras just need to match fov?
                self.camera.fovx = 0.75;
                self.camera.fovy = self.camera.fovx * (size.y as f32) / (size.x as f32);

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
                controls: OrbitControls::new(7.0),
                last_draw: Instant::now(),
            },
            train_pause: false,
            train_iter_per_s: 0.0,
            last_train_step: (Instant::now(), 0),
            device,
        }
    }

    fn stop_training(&mut self) {
        self.receiver = None; // This drops the receiver, which closes the channel.
    }

    pub fn start_data_load(&mut self, data_url: String) {
        <Backend as burn::prelude::Backend>::seed(42);

        // create a channel for the train loop.
        let (sender, receiver) = async_channel::bounded(2);
        let device = self.device.clone();
        self.receiver = Some(receiver);
        let ctx = self.ctx.clone();

        // Reset camera & controls.
        self.splat_view.camera = Camera::new(
            Vec3::ZERO,
            Quat::from_rotation_y(std::f32::consts::PI / 2.0)
                * Quat::from_rotation_x(-std::f32::consts::PI / 8.0),
            0.5,
            0.5,
        );
        self.splat_view.controls = OrbitControls::new(7.0);
        self.train_pause = false;

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || pollster::block_on(train_loop(device, sender, ctx, data_url)));

        #[cfg(target_arch = "wasm32")]
        spawn_local(train_loop(device, sender, ctx, data_url));
    }

    fn url_button(&mut self, label: &str, url: &str, ui: &mut egui::Ui) {
        if ui.button(label).clicked() {
            self.start_data_load(url.to_owned());
        }
    }
}

impl eframe::App for Viewer {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let _span = info_span!("Draw UI").entered();

        egui_extras::install_image_loaders(ctx);

        egui::SidePanel::left("Data").show(ctx, |ui| {
            ui.add_space(55.0);

            // ui.vertical_centered(|ui| {
            //     if ui.button("Load file").clicked() {
            //         self.start_data_load();
            //     }

            //     ui.add_space(15.0);
            // });

            ui.label("Select a .ply to visualize, or a .zip with a transforms_train.json and training images. (limited formats are supported currently)");

            ui.add_space(15.0);

            ui.heading("Native app");
            ui.label("The native app is currently still a good amount faster than the web app. It also includes some more visualizations.");
            ui.collapsing("Download", |ui| {
                ui.add(Hyperlink::from_label_and_url("MacOS", "https://drive.google.com/file/d/1-wBAr94WlSVdrbLi9ImMK14j6DFXSGaJ/view?usp=sharing").open_in_new_tab(true));
                ui.add(Hyperlink::from_label_and_url("Windows", "https://drive.google.com/file/d/1hgHxM5Hprny-bhV1-329SdKYqkAY9dUX/view?usp=sharing").open_in_new_tab(true));
            });

            ui.heading("Pretrained splats");

            ui.collapsing("Polycam examples", |ui| {
                ui.label("Examples from poylcam showcase. Nb: Licenses apply, demo only!");
                self.url_button("city_sector (250MB)", "https://drive.google.com/file/d/12yoAvwsUh1TNRt4I1rfTxyRp-_6p0x0b/view?usp=sharing", ui);
                self.url_button("flowers (120MB)", "https://drive.google.com/file/d/1KD_IP-Qt782guD1PvNATQrJGM0kxIhGa/view?usp=sharing", ui);
                self.url_button("fountain (340MB)", "https://drive.google.com/file/d/13mQfEoSNy-hOh5Ir9NuHLgtj6JA6aL4d/view?usp=sharing", ui);
                self.url_button("hollywood sign (320MB)", "https://drive.google.com/file/d/1bZfsNe5DVVgq2FM49e7StRKcnKVrvAma/view?usp=sharing", ui);
                self.url_button("inveraray castle (300MB)", "https://drive.google.com/file/d/1EOir_xBPE9Ns5CToEw_eAINGNHg5mA1F/view?usp=sharing", ui);
                self.url_button("lighthouse (70MB)", "https://drive.google.com/file/d/1f_ZCp04wax_aD6M699zg8wlWmiMU2EFQ/view?usp=sharing", ui);
                self.url_button("room arch vis (160MB)", "https://drive.google.com/file/d/1wi6B-6fPn2cQuGiucg_AMvVW623-reEF/view?usp=sharing", ui);
                self.url_button("small bonsai (135MB)", "https://drive.google.com/file/d/1wXiW9vn32DXG7NP0MsBQP4E8RF4M8nV-/view?usp=sharing", ui);
                self.url_button("varenna (200MB)", "https://drive.google.com/file/d/1lvljIKlMjVSRjy4KfPhbRSjN8t6KJYM9/view?usp=sharing", ui);
            });

            ui.collapsing("Mipnerf scenes (warning: big)", |ui| {
                ui.label("Reference mipnerf ply files.");

                self.url_button("bicycle (1.4GB)", "https://drive.google.com/file/d/1kHkNqGFLLutRt3R7k2tGkjGwfXnPLnCi/view?usp=sharing", ui);
                self.url_button("bonsai (300MB)", "https://drive.google.com/file/d/1jf4bjaeTGeru1PQS_Ue716uc_edRbAPd/view?usp=sharing", ui);
                self.url_button("counter (290MB)", "https://drive.google.com/file/d/1O89SIHcWdmrWi75Cf6tDrv2Dl6yGndcz/view?usp=sharing", ui);
                self.url_button("drjohnson (800MB)", "https://drive.google.com/file/d/13FEQ7UZHYwymBTwxzpPeJob4cr8VxUTV/view?usp=sharing", ui);
                self.url_button("garden (1.3GB)", "https://drive.google.com/file/d/13FEQ7UZHYwymBTwxzpPeJob4cr8VxUTV/view?usp=sharing", ui);
                self.url_button("kitchen (440MB)", "https://drive.google.com/file/d/13FEQ7UZHYwymBTwxzpPeJob4cr8VxUTV/view?usp=sharing", ui);
                self.url_button("playroom (600MB)", "https://drive.google.com/file/d/13FEQ7UZHYwymBTwxzpPeJob4cr8VxUTV/view?usp=sharing", ui);
                self.url_button("room (375MB)", "https://drive.google.com/file/d/13FEQ7UZHYwymBTwxzpPeJob4cr8VxUTV/view?usp=sharing", ui);
                self.url_button("stump (1.15GB)", "https://drive.google.com/file/d/13FEQ7UZHYwymBTwxzpPeJob4cr8VxUTV/view?usp=sharing", ui);
            });

            ui.heading("Training Data");

            ui.collapsing("Train blender scenes", |ui| {
                self.url_button("Chair", "https://drive.google.com/file/d/13Q6s0agTW1_a7cFGcSmll1-Aikq_OPKe/view?usp=sharing", ui);
                self.url_button("Drums", "https://drive.google.com/file/d/1j8TuMiGb84YtlrZ0gnkMNOzUaIJqz0SY/view?usp=sharing", ui);
                self.url_button("Ficus", "https://drive.google.com/file/d/1VzT5SDiBefn9fvRw7LeYjUfDBZHCyzQ4/view?usp=sharing", ui);
                self.url_button("Hotdog", "https://drive.google.com/file/d/1hOjnCV8XdXClV2eC6c9H6PIQTUYv8zys/view?usp=sharing", ui);
                self.url_button("Lego", "https://storage.googleapis.com/example_brush_data/lego.zip", ui);
                self.url_button("Materials", "https://drive.google.com/file/d/1L7J5PNBcLcXde6CqzzkaNxHt7JtG2GIW/view?usp=sharing", ui);
                self.url_button("Mic", "https://drive.google.com/file/d/1SA0NNi0HsUHE6FgAP8XpD23N1xftsrr-/view?usp=sharing", ui);
                self.url_button("Ship", "https://drive.google.com/file/d/1rzL0KrWuLFebT1hLLm4uYnrNXNTkfjxM/view?usp=sharing", ui);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(rx) = self.receiver.as_mut() {
                // Always animate at at least 20FPS while receiving messages.
                ctx.request_repaint_after_secs(0.05);

                if !self.train_pause {
                    match rx.try_recv() {
                        Ok(message) => {
                            if let ViewerMessage::TrainStep {
                                iter, timestamp, ..
                            } = &message
                            {
                                self.train_iter_per_s = (iter - self.last_train_step.1) as f32
                                    / (*timestamp - self.last_train_step.0).as_secs_f32();
                                self.last_train_step = (*timestamp, *iter);
                            };

                            self.last_message = Some(message)
                        }
                        Err(TryRecvError::Empty) => (), // nothing to do.
                        Err(TryRecvError::Closed) => self.receiver = None, // channel closed.
                    }
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
                                self.train_iter_per_s
                            ));

                            ui.label(format!("loss: {loss:.3e}"));

                            let paused = self.train_pause;
                            ui.toggle_value(&mut self.train_pause, if paused { "⏵" } else { "⏸" });
                        });

                        self.splat_view.draw_splats(splats, ui, ctx, frame);
                    }
                }
            } else if self.receiver.is_some() {
                ui.label("Loading...");
            }
        });

        if self.splat_view.controls.is_animating() {
            ctx.request_repaint();
        }
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 1.0]
    }
}
