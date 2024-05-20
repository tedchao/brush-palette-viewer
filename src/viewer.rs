use crate::{
    camera::Camera, dataset_readers, gaussian_splats::Splats, splat_import, splat_render::BurnBack,
};
use anyhow::Result;
use burn::tensor::Tensor;
use burn_wgpu::{RuntimeOptions, WgpuDevice};
use core::ops::DerefMut;
use eframe::{egui_wgpu::WgpuConfiguration, NativeOptions};
use egui::{pos2, Color32, Rect, TextureId};
use glam::{Mat3, Mat4, Quat, Vec2, Vec3};
use std::{
    sync::Arc,
    time::{self, Duration},
};
use wgpu::ImageDataLayout;

/// Tags an entity as capable of panning and orbiting.
struct OrbitControls {
    pub focus: Vec3,
    pub radius: f32,
    pub rotation: glam::Quat,
    pub position: glam::Vec3,
}

impl OrbitControls {
    fn new(radius: f32) -> Self {
        Self {
            focus: Vec3::ZERO,
            radius,
            rotation: Quat::IDENTITY,
            position: Vec3::NEG_Z * radius,
        }
    }
}

impl OrbitControls {
    /// Pan the camera with middle mouse click, zoom with scroll wheel, orbit with right mouse click.
    fn pan_orbit_camera(&mut self, pan: Vec2, rotate: Vec2, scroll: f32, window: Vec2) {
        let mut any = false;
        if rotate.length_squared() > 0.0 {
            any = true;
            let delta_x = rotate.x * std::f32::consts::PI * 2.0 / window.x;
            let delta_y = rotate.y * std::f32::consts::PI / window.y;
            let yaw = Quat::from_rotation_y(delta_x);
            let pitch = Quat::from_rotation_x(-delta_y);
            self.rotation = yaw * self.rotation * pitch;
        } else if pan.length_squared() > 0.0 {
            any = true;
            // make panning distance independent of resolution and FOV,
            let scaled_pan = pan * Vec2::new(1.0 / window.x, 1.0 / window.y);

            // translate by local axes
            let right = self.rotation * Vec3::X * -scaled_pan.x;
            let up = self.rotation * Vec3::Y * -scaled_pan.y;

            // make panning proportional to distance away from focus point
            let translation = (right + up) * self.radius;
            self.focus += translation;
        } else if scroll.abs() > 0.0 {
            any = true;
            self.radius -= scroll * self.radius * 0.2;
            // dont allow zoom to reach zero or you get stuck
            self.radius = f32::max(self.radius, 0.05);
        }

        if any {
            // emulating parent/child to make the yaw/y-axis rotation behave like a turntable
            // parent = x and y rotation
            // child = z-offset
            let rot_matrix = Mat3::from_quat(self.rotation);
            self.position = self.focus + rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, -self.radius));
        }
    }
}

// TODO: This probably doesn't belong here but meh.
fn copy_buffer_to_texture(img: Tensor<BurnBack, 3>, texture: &wgpu::Texture) {
    let client = img.clone().into_primitive().client.clone();
    let [height, width, _] = img.shape().dims;
    let img_handle = img.into_primitive().handle;

    client.run_custom_command(move |server| {
        let img_res = server.get_resource_binding(img_handle.clone().binding());

        // Put compute passes in encoder before copying the buffer.
        let bytes_per_row = Some(4 * width as u32);
        let encoder = server.get_command_encoder();

        // Now copy the buffer to the texture.
        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: img_res.buffer.as_ref(),
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row,
                    rows_per_image: None,
                },
            },
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
        );
    });
}

struct BackBuffer {
    texture: wgpu::Texture,
    id: TextureId,
}

struct Viewer {
    camera: Camera,
    splats: Option<Splats<BurnBack>>,
    reference_cameras: Option<Vec<Camera>>,
    backbuffer: Option<BackBuffer>,
    controls: OrbitControls,
    device: WgpuDevice,
    start_transform: Mat4,
    last_render_time: time::Instant,
}

impl Viewer {
    pub fn new(cc: &eframe::CreationContext) -> Self {
        let state = cc.wgpu_render_state.as_ref().unwrap();
        // Run the burn backend on the egui WGPU device.
        let device = burn::backend::wgpu::init_existing_device(
            0,
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
            RuntimeOptions {
                dealloc_strategy: burn_compute::memory_management::DeallocStrategy::PeriodTime {
                    period: Duration::from_secs(10),
                    state: time::Instant::now(),
                },
                ..Default::default()
            },
        );

        let mut viewer = Viewer {
            camera: Camera::new(Vec3::ZERO, Quat::IDENTITY, 500.0, 500.0),
            splats: None,
            reference_cameras: None,
            backbuffer: None,
            controls: OrbitControls::new(15.0),
            device,
            start_transform: glam::Mat4::IDENTITY,
            last_render_time: time::Instant::now(),
        };
        viewer.load_splats(
            "./models/counter/point_cloud/iteration_30000/point_cloud.ply",
            Some("./models/counter/cameras.json"),
        );
        viewer
    }

    pub fn load_splats(&mut self, path: &str, reference_view: Option<&str>) {
        self.reference_cameras =
            reference_view.and_then(|s| dataset_readers::read_viewpoint_data(s).ok());
        self.splats = splat_import::load_splat_from_ply::<BurnBack>(path, &self.device).ok();
        if let Some(refs) = self.reference_cameras.as_ref() {
            self.camera = refs[0].clone();
        }
    }

    fn update_backbuffer(&mut self, size: glam::UVec2, frame: &mut eframe::Frame) {
        let dirty = !matches!(
            self.backbuffer.as_ref(),
            Some(back) if back.texture.width() == size.x && back.texture.height() == size.y
        );

        if dirty {
            let state = frame.wgpu_render_state().unwrap();
            let egui_device = state.device.clone();

            let texture = egui_device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Splat backbuffer"),
                size: wgpu::Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb, // Minspec for wgpu WebGL emulation is WebGL2, so this should always be supported.
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
            });

            let view = texture.create_view(&Default::default());

            let mut rend_guard = state.renderer.write();
            let renderer = rend_guard.deref_mut();

            if let Some(back) = self.backbuffer.as_mut() {
                back.texture = texture;
                renderer.update_egui_texture_from_wgpu_texture(
                    &egui_device,
                    &view,
                    wgpu::FilterMode::Linear,
                    back.id,
                )
            } else {
                self.backbuffer = Some(BackBuffer {
                    texture,
                    id: renderer.register_native_texture(
                        &egui_device,
                        &view,
                        wgpu::FilterMode::Linear,
                    ),
                });
            }
        }
    }
}

impl eframe::App for Viewer {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui_extras::install_image_loaders(ctx);

        egui::SidePanel::new(egui::panel::Side::Left, "cam panel").show(ctx, |ui| {
            if let Some(cameras) = self.reference_cameras.as_ref() {
                for (i, camera) in cameras.iter().enumerate() {
                    if ui.button(format!("Camera {i}")).clicked() {
                        self.camera = camera.clone();
                        self.controls.position = self.camera.position;
                        self.controls.rotation = self.camera.rotation;
                    }
                }
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Brush splat viewer");

            ui.horizontal(|ui| {
                for r in ["bonsai", "stump", "counter", "garden", "truck"] {
                    if ui.button(r.to_string()).clicked() {
                        self.load_splats(
                            &format!("./models/{r}/point_cloud/iteration_30000/point_cloud.ply"),
                            Some(&format!("./models/{r}/cameras.json")),
                        );
                    }
                }
            });

            let now = time::Instant::now();
            let ms = (now - self.last_render_time).as_secs_f64() * 1000.0;
            let fps = 1000.0 / ms;
            self.last_render_time = now;

            ui.label(format!("FPS: {fps:.0} {ms:.0} ms/frame"));

            let size = ui.available_size();
            // Round to 16 pixels for buffer alignment.
            // TODO: Ideally just alloc a backbuffer that's aligned
            // and render a slice of it.
            let size = glam::uvec2((size.x as u32).div_ceil(64) * 64, size.y as u32);
            self.update_backbuffer(size, frame);

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

            let dirty = response.dragged() || scrolled > 0.0;

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
            self.camera.fovx = 0.5;
            self.camera.fovy = 0.5 * (size.y as f32) / (size.x as f32);

            if let Some(backbuffer) = &self.backbuffer {
                if let Some(splats) = &self.splats {
                    let (img, _) = splats.render(&self.camera, size, glam::vec3(0.0, 0.0, 0.0));
                    copy_buffer_to_texture(img, &backbuffer.texture);
                }

                ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
                ui.painter().image(
                    backbuffer.id,
                    rect,
                    Rect {
                        min: pos2(0.0, 0.0),
                        max: pos2(1.0, 1.0),
                    },
                    Color32::WHITE,
                );
            }

            if dirty {
                ctx.request_repaint();
            }
            ctx.request_repaint();
        });
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 1.0]
    }
}

pub(crate) fn start() -> Result<()> {
    let native_options = NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(egui::Vec2::new(1920.0, 1080.0)),
        vsync: false,
        wgpu_options: WgpuConfiguration {
            device_descriptor: Arc::new(|adapter| wgpu::DeviceDescriptor {
                label: Some("egui+burn wgpu device"),
                required_features: wgpu::Features::default(),
                required_limits: adapter.limits(),
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "My egui App",
        native_options,
        Box::new(move |cc| Box::new(Viewer::new(cc))),
    )
    .unwrap();

    Ok(())
}
