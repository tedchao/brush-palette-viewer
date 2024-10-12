use brush_render::{camera::Camera, gaussian_splats::Splats};
use egui::{pos2, CollapsingHeader, Color32, Rect};
use glam::{Quat, Vec2, Vec3};
use tracing::trace_span;
use web_time::Instant;
use wgpu::CommandEncoderDescriptor;

use crate::{burn_texture::BurnTexture, orbit_controls::OrbitControls};

// A simple window that can draw some splats.
pub(crate) struct SplatView {
    pub(crate) camera: Camera,
    pub(crate) controls: OrbitControls,
    pub(crate) backbuffer: Option<BurnTexture>,
    pub(crate) last_draw: Instant,
}

impl SplatView {
    pub(crate) fn new() -> Self {
        SplatView {
            camera: Camera::new(
                -Vec3::Z * 5.0,
                Quat::IDENTITY,
                glam::vec2(0.5, 0.5),
                glam::vec2(0.5, 0.5),
            ),
            backbuffer: None,
            controls: OrbitControls::new(),
            last_draw: Instant::now(),
        }
    }

    pub(crate) fn draw_splats(
        &mut self,
        splats: &Splats<brush_render::PrimaryBackend>,
        background: glam::Vec3,
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

                self.controls.pan_orbit_camera(
                    &mut self.camera,
                    pan * 5.0,
                    rotate * 5.0,
                    scrolled * 0.01,
                    glam::vec2(rect.size().x, rect.size().y),
                    delta_time.as_secs_f32(),
                );

                let base_fov = 0.75;
                self.camera.fov =
                    glam::vec2(base_fov, base_fov * (size.y as f32) / (size.x as f32));

                // If there's actual rendering to do, not just an imgui update.
                if ctx.has_requested_repaint() {
                    let _span = trace_span!("Render splats").entered();
                    let (img, _) = splats.render(&self.camera, size, background, true);

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

    pub(crate) fn is_animating(&self) -> bool {
        self.controls.is_animating()
    }
}
