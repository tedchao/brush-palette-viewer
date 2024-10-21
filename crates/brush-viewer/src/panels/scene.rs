use egui::epaint::mutex::RwLock as EguiRwLock;
use std::sync::Arc;

use brush_render::gaussian_splats::Splats;
use eframe::egui_wgpu::Renderer;
use egui::{Color32, Rect};
use glam::Vec2;
use tracing::trace_span;
use web_time::Instant;
use wgpu::CommandEncoderDescriptor;

use crate::{
    burn_texture::BurnTexture,
    viewer::{ViewerContext, ViewerMessage},
    ViewerPanel,
};

pub(crate) struct ScenePanel {
    pub(crate) backbuffer: BurnTexture,
    pub(crate) last_draw: Option<Instant>,
    pub(crate) last_message: Option<ViewerMessage>,

    queue: Arc<wgpu::Queue>,
    device: Arc<wgpu::Device>,
    renderer: Arc<EguiRwLock<Renderer>>,
}

impl ScenePanel {
    pub(crate) fn new(
        queue: Arc<wgpu::Queue>,
        device: Arc<wgpu::Device>,
        renderer: Arc<EguiRwLock<Renderer>>,
    ) -> Self {
        Self {
            backbuffer: BurnTexture::new(),
            last_draw: None,
            last_message: None,
            queue,
            device,
            renderer,
        }
    }

    pub(crate) fn draw_splats(
        &mut self,
        ui: &mut egui::Ui,
        context: &mut ViewerContext,
        splats: &Splats<brush_render::PrimaryBackend>,
        background: glam::Vec3,
    ) {
        let mut size = ui.available_size();
        let focal = context.camera.focal(glam::uvec2(1, 1));
        let aspect_ratio = focal.y / focal.x;
        if size.x / size.y > aspect_ratio {
            size.x = size.y * aspect_ratio;
        } else {
            size.y = size.x / aspect_ratio;
        }
        // Round to 64 pixels. Necesarry for buffer sizes to align.
        let size = glam::uvec2(size.x.round() as u32, size.y.round() as u32);

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

        if let Some(last_draw) = self.last_draw {
            let delta_time = cur_time - last_draw;

            context.controls.pan_orbit_camera(
                &mut context.camera,
                pan * 5.0,
                rotate * 5.0,
                scrolled * 0.01,
                glam::vec2(rect.size().x, rect.size().y),
                delta_time.as_secs_f32(),
            );
        }
        self.last_draw = Some(cur_time);

        // If this viewport is re-rendering.
        if ui.ctx().has_requested_repaint() {
            let _span = trace_span!("Render splats").entered();
            let (img, _) = splats.render(&context.camera, size, background, true);

            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("viewer encoder"),
                });
            self.backbuffer
                .update_texture(img, &self.device, self.renderer.clone(), &mut encoder);
            self.queue.submit([encoder.finish()]);
        }

        if let Some(id) = self.backbuffer.id() {
            ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
            ui.painter().image(
                id,
                rect,
                Rect {
                    min: egui::pos2(0.0, 0.0),
                    max: egui::pos2(1.0, 1.0),
                },
                Color32::WHITE,
            );
        }

        ui.label(format!("Resolution {}x{}", size.x, size.y));
    }
}

impl ViewerPanel for ScenePanel {
    fn title(&self) -> String {
        "Scene".to_owned()
    }

    fn on_message(&mut self, message: ViewerMessage, _: &mut ViewerContext) {
        match message.clone() {
            ViewerMessage::PickFile => {
                self.last_message = None;
            }
            ViewerMessage::Splats { iter: _, splats: _ } => {
                self.last_message = Some(message);
            }
            ViewerMessage::Error(_) => {
                self.last_message = Some(message);
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut ViewerContext) {
        let _span = trace_span!("Draw UI").entered();

        if let Some(message) = self.last_message.clone() {
            match message {
                ViewerMessage::PickFile => {
                    ui.label("Loading...");
                }
                ViewerMessage::Error(e) => {
                    ui.label("Error: ".to_owned() + &e.to_string());
                }
                ViewerMessage::Splats { iter: _, splats } => {
                    self.draw_splats(ui, context, &splats, context.dataset.train.background);
                }
                _ => {}
            }
        }

        if context.controls.is_animating() {
            ui.ctx().request_repaint();
        }
    }
}
