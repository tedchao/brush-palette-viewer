use std::sync::Arc;

use crate::{
    camera::Camera,
    dataset_readers,
    gaussian_splats::{self, Splats},
    splat_render::BurnBack,
};
use anyhow::Result;
use burn_wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuData};
use eframe::{egui_wgpu::WgpuConfiguration, NativeOptions};
use egui::{pos2, Color32, Rect, TextureId};
use wgpu::ImageDataLayout;

struct RunningData {
    splats: Splats<BurnBack>,
    backbuffer: wgpu::Texture,
    texture_id: TextureId,
}

struct MyApp {
    camera: Camera,
    path: String,
    did_init: bool,
    data: Option<RunningData>,
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let device = burn_wgpu::WgpuDevice::Existing;

        if !self.did_init {
            let state = frame.wgpu_render_state().unwrap();

            burn::backend::wgpu::init_sync::<AutoGraphicsApi>(
                &device,
                RuntimeOptions {
                    wgpu_devices: Some(WgpuData {
                        adapter: state.adapter.clone(),
                        device: state.device.clone(),
                        queue: state.queue.clone(),
                    }),
                    ..Default::default()
                },
            );
            let splats = gaussian_splats::create_from_ply::<BurnBack>(&self.path, &device).unwrap();
            self.did_init = true;

            let state = frame.wgpu_render_state().unwrap();
            let mut rendererer = state.renderer.write();
            let egui_device = state.device.clone();

            let backbuffer = egui_device.create_texture(&wgpu::TextureDescriptor {
                label: Some("EGUI custom texture"),
                size: wgpu::Extent3d {
                    width: self.camera.width as u32,
                    height: self.camera.height as u32,
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

        let data = self.data.as_ref().expect("Initialization failed?");
        let (img, _) = data.splats.render(&self.camera, glam::vec3(0.0, 0.0, 0.0));
        <BurnBack as burn::prelude::Backend>::sync(&device);

        {
            let client = img.clone().into_primitive().client;
            let mut device_guard = client.channel.server.lock();

            // Put compute passes in encoder before copying the buffer.
            device_guard.register_tasks();

            let resource = device_guard.get_resource(&img.into_primitive().handle);
            let bytes_per_row = Some(4 * self.camera.width);

            // Now copy the buffer to the texture.
            device_guard.encoder.copy_buffer_to_texture(
                wgpu::ImageCopyBuffer {
                    buffer: &resource.buffer,
                    layout: ImageDataLayout {
                        offset: 0,
                        bytes_per_row,
                        rows_per_image: None,
                    },
                },
                wgpu::ImageCopyTexture {
                    texture: &data.backbuffer,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: data.backbuffer.width(),
                    height: data.backbuffer.height(),
                    depth_or_array_layers: 1,
                },
            );
            // Flush to device queue.
            device_guard.submit();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("My egui Application");

            let uv = Rect {
                min: pos2(0.0, 0.0),
                max: pos2(1.0, 1.0),
            };
            let (rect, response) = ui.allocate_exact_size(
                egui::Vec2::new(self.camera.width as f32, self.camera.height as f32),
                egui::Sense::drag(),
            );

            let mut translation = glam::Vec3::ZERO;
            let camera_forwarwd = self.camera.rotation * glam::vec3(0.0, 0.0, 1.0);
            let camera_right = self.camera.rotation * glam::vec3(1.0, 0.0, 0.0);

            let move_speed = 0.05;
            // TODO: Deltatime.
            if ui.input(|i| i.key_down(egui::Key::W)) {
                translation += camera_forwarwd * move_speed;
                ctx.request_repaint();
            }
            if ui.input(|i| i.key_down(egui::Key::A)) {
                translation += -camera_right * move_speed;
                ctx.request_repaint();
            }
            if ui.input(|i| i.key_down(egui::Key::S)) {
                translation += -camera_forwarwd * move_speed;
                ctx.request_repaint();
            }
            if ui.input(|i| i.key_down(egui::Key::D)) {
                translation += camera_right * move_speed;
                ctx.request_repaint();
            }

            let drag = response.drag_delta();

            let rot_x = drag.x * 0.005;
            let rot_y = drag.y * 0.005;
            let rotation = glam::Quat::from_euler(glam::EulerRot::XYZ, -rot_y, rot_x, 0.0);

            self.camera.position += translation;
            self.camera.rotation = self.camera.rotation.mul_quat(rotation).normalize();

            ui.painter()
                .image(data.texture_id, rect, uv, Color32::WHITE);
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
        cameras[0].fovy,
        1024,
        1024,
    );

    eframe::run_native(
        "My egui App",
        native_options,
        Box::new(move |_cc| {
            Box::new(MyApp {
                path: path.to_owned(),
                camera: start_camera,
                did_init: false,
                data: None,
            })
        }),
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
