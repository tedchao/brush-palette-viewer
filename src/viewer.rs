use std::sync::Arc;

use crate::{
    camera::Camera,
    dataset_readers,
    gaussian_splats::{self, Splats},
    splat_render::BurnBack,
};
use anyhow::Result;
use burn::tensor::Tensor;
use eframe::{egui_wgpu::WgpuConfiguration, NativeOptions};
use egui::{pos2, Color32, Rect, TextureId};
use glam::{Mat3, Mat4, Quat, Vec2, Vec3};
use wgpu::ImageDataLayout;

struct RunningData {
    splats: Splats<BurnBack>,
    backbuffer: wgpu::Texture,
    texture_id: TextureId,
}

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

struct MyApp {
    camera: Camera,
    path: String,
    data: Option<RunningData>,
    controls: OrbitControls,
    start_transform: Mat4,
}

impl MyApp {
    fn new(path: &str, camera: Camera) -> Self {
        MyApp {
            camera: camera.clone(),
            path: path.to_owned(),
            data: None,
            controls: OrbitControls::new(15.0),
            start_transform: glam::Mat4::from_rotation_translation(
                camera.rotation,
                camera.position,
            ),
        }
    }
}

fn copy_buffer_to_texture(img: Tensor<BurnBack, 3>, texture: &wgpu::Texture) {
    let client = img.clone().into_primitive().client.clone();
    let [height, width, _] = img.shape().dims;

    client.run_custom_command(
        |server, resources| {
            // Put compute passes in encoder before copying the buffer.
            let img_res = &resources[0];
            let bytes_per_row = Some(4 * width as u32);
            let encoder = server.get_command_encoder();

            // Now copy the buffer to the texture.
            encoder.copy_buffer_to_texture(
                wgpu::ImageCopyBuffer {
                    buffer: &img_res.buffer,
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
        },
        &[&img.into_primitive().handle],
    );
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if self.data.is_none() {
            let state = frame.wgpu_render_state().unwrap();

            let device = burn::backend::wgpu::init_existing_device(
                0,
                state.adapter.clone(),
                state.device.clone(),
                state.queue.clone(),
                Default::default(),
            );

            let splats = gaussian_splats::create_from_ply::<BurnBack>(&self.path, &device).unwrap();

            let mut rendererer = state.renderer.write();
            let egui_device = state.device.clone();

            let backbuffer = egui_device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Splat backbuffer"),
                size: wgpu::Extent3d {
                    width: self.camera.width,
                    height: self.camera.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb, // Minspec for wgpu WebGL emulation is WebGL2, so this should always be supported.
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
            });
            let texture_id = rendererer.register_native_texture(
                &egui_device,
                &backbuffer.create_view(&wgpu::TextureViewDescriptor {
                    ..Default::default()
                }),
                wgpu::FilterMode::Linear,
            );
            self.data = Some(RunningData {
                splats,
                backbuffer,
                texture_id,
            });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Brush splat viewwer");

            let (rect, response) = ui.allocate_exact_size(
                egui::Vec2::new(self.camera.width as f32, self.camera.height as f32),
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
            let scrolled = ui.input(|r| r.scroll_delta).y;

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

            let data = self.data.as_ref().expect("Initialization failed?");
            let (img, _) = data.splats.render(&self.camera, glam::vec3(0.0, 0.0, 0.0));
            copy_buffer_to_texture(img, &data.backbuffer);

            ui.painter().rect_filled(rect, 0.0, Color32::BLACK);

            ui.painter().image(
                data.texture_id,
                rect,
                Rect {
                    min: pos2(0.0, 0.0),
                    max: pos2(1.0, 1.0),
                },
                Color32::WHITE,
            );

            if dirty {
                ctx.request_repaint();
            }
        });
    }
}

pub(crate) fn view(path: &'static str, viewpoints: &str) -> Result<()> {
    // Init burns backend.

    let cameras = dataset_readers::read_viewpoint_data(viewpoints)?;

    let native_options = NativeOptions {
        wgpu_options: WgpuConfiguration {
            device_descriptor: Arc::new(|adapter| wgpu::DeviceDescriptor {
                label: Some("egui+burn wgpu device"),
                features: wgpu::Features::default(),
                limits: adapter.limits(),
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    let start_camera = Camera::new(
        cameras[0].position,
        cameras[0].rotation,
        cameras[0].fovx,
        cameras[0].fovx,
        1024,
        1024,
    );

    eframe::run_native(
        "My egui App",
        native_options,
        Box::new(move |_cc| Box::new(MyApp::new(path, start_camera))),
    )
    .unwrap();

    Ok(())

    // let rec = rerun::RecordingStreamBuilder::new("visualize training").spawn()?;
    // let splats: Splats<B> = gaussian_splats::create_from_ply(path, device)?;
    // let cameras = dataset_readers::read_viewpoint_data(viewpoints)?;
    // splats.visualize(&rec)?;
    // for (i, camera) in cameras.iter().enumerate() {
    //     let _span = info_span!("Splats render, sync").entered();

    //     let (img, _) = splats.render(camera, glam::vec3(0.0, 0.0, 0.0));
    //     B::sync(device);
    //     drop(_span);

    //     let img = Array::from_shape_vec(img.dims(), img.to_data().convert::<f32>().value).unwrap();
    //     let img = img.map(|x| (*x * 255.0).clamp(0.0, 255.0) as u8);

    //     rec.log_timeless(
    //         format!("images/fixed_camera_render_{i}"),
    //         &rerun::Image::try_from(img).unwrap(),
    //     )?;

    //     let rerun_camera = rerun::Pinhole::from_focal_length_and_resolution(
    //         camera.focal(),
    //         glam::vec2(camera.width as f32, camera.height as f32),
    //     );
    //     // TODO: make a function.
    //     let cam_path = format!("world/camera_{i}");
    //     rec.log_timeless(cam_path.clone(), &rerun_camera)?;
    //     rec.log_timeless(
    //         cam_path.clone(),
    //         &rerun::Transform3D::from_translation_rotation(camera.position(), camera.rotation()),
    //     )?;
    // }
}
