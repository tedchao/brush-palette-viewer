use brush_render::camera::Camera;
use glam::{Quat, Vec3};
use std::sync::Arc;

use anyhow::Context;
use async_channel::{Receiver, Sender};
use brush_render::gaussian_splats::Splats;
use brush_render::PrimaryBackend;
use burn_wgpu::{RuntimeOptions, Wgpu, WgpuDevice};
use egui::Hyperlink;
use egui_tiles::Tiles;
use futures_lite::StreamExt;
use web_time::Instant;

use brush_dataset::{self, Dataset, ZipData};

use crate::orbit_controls::OrbitControls;
use crate::panels::{LoadDataPanel, ScenePanel, ViewpointsPane};
use crate::train_loop::TrainArgs;
use crate::{splat_import, train_loop, PaneType, ViewerTree};

use eframe::egui;

#[derive(Clone)]
pub(crate) enum ViewerMessage {
    StartLoading,
    /// Some process errored out, and want to display this error
    /// to the user.
    Error(Arc<anyhow::Error>),
    /// Loaded a splat from a ply file.
    ///
    /// Nb: This includes all the intermediately loaded splats.
    SplatLoad {
        splats: Splats<PrimaryBackend>,
        total_count: usize,
    },
    /// Loaded a bunch of viewpoints to train on.
    Dataset(Dataset),
    /// Some number of training steps are done.
    TrainStep {
        splats: Splats<PrimaryBackend>,
        loss: f32,
        iter: u32,
        timestamp: Instant,
    },
}

pub struct Viewer {
    tree: egui_tiles::Tree<PaneType>,
    tree_ctx: ViewerTree,
}

// TODO: Bit too much random shared state here...
pub(crate) struct ViewerContext {
    pub dataset: Dataset,
    pub camera: Camera,
    pub controls: OrbitControls,

    device: WgpuDevice,
    ctx: egui::Context,
    receiver: Option<Receiver<ViewerMessage>>,
}

async fn process_loop(
    device: WgpuDevice,
    sender: Sender<ViewerMessage>,
    egui_ctx: egui::Context,
    train_args: TrainArgs,
) -> anyhow::Result<()> {
    let picked = rrfd::pick_file().await?;

    let _ = sender.send(ViewerMessage::StartLoading).await;

    if picked.file_name.contains(".ply") {
        load_ply_loop(&picked.data, device, sender, egui_ctx).await
    } else if picked.file_name.contains(".zip") {
        train_loop::train_loop(
            ZipData::from(picked.data),
            device,
            sender,
            egui_ctx,
            train_args,
        )
        .await
    } else {
        anyhow::bail!("Only .ply and .zip files are supported.")
    }
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

impl ViewerContext {
    fn new(device: WgpuDevice, ctx: egui::Context) -> Self {
        Self {
            camera: Camera::new(
                -Vec3::Z * 5.0,
                Quat::IDENTITY,
                glam::vec2(0.5, 0.5),
                glam::vec2(0.5, 0.5),
            ),
            controls: OrbitControls::new(),
            device,
            ctx,
            dataset: Dataset::empty(),
            receiver: None,
        }
    }

    pub fn focus_view(&mut self, cam: &Camera) {
        // TODO: How to control scene view.... ?
        self.camera = cam.clone();

        let scale = 0.5 / (2.0f32.sqrt());

        // TODO: Figure out a better focus.
        self.controls.focus = cam.position
            + cam.rotation * glam::Vec3::Z * self.dataset.train.bounds(0.0).extent.length() * scale;
    }

    pub(crate) fn start_data_load(&mut self, args: TrainArgs) {
        <Wgpu as burn::prelude::Backend>::seed(42);

        let device = self.device.clone();
        let ctx = self.ctx.clone();

        // create a channel for the train loop.
        let (sender, receiver) = async_channel::bounded(2);
        self.receiver = Some(receiver);
        self.dataset = Dataset::empty();

        let inner_process_loop = move || async move {
            if let Err(e) = process_loop(device, sender.clone(), ctx, args).await {
                let _ = sender.send(ViewerMessage::Error(Arc::new(e))).await;
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || futures_lite::future::block_on(inner_process_loop()));

        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(inner_process_loop());
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

        let mut tiles: Tiles<PaneType> = egui_tiles::Tiles::default();

        let context = ViewerContext::new(device, cc.egui_ctx.clone());

        let data_pane = LoadDataPanel::new();
        let viewpoints_pane = ViewpointsPane::new();

        let scene_pane = ScenePanel::new(
            state.queue.clone(),
            state.device.clone(),
            state.renderer.clone(),
        );

        let sides = vec![
            tiles.insert_pane(Box::new(data_pane)),
            tiles.insert_pane(Box::new(viewpoints_pane)),
            #[cfg(feature = "tracing")]
            tiles.insert_pane(Box::new(TracingPanel::default())),
        ];

        let side_panel = tiles.insert_vertical_tile(sides);

        let scene_pane_id = tiles.insert_pane(Box::new(scene_pane));

        let mut lin = egui_tiles::Linear::new(
            egui_tiles::LinearDir::Horizontal,
            vec![side_panel, scene_pane_id],
        );
        lin.shares.set_share(side_panel, 0.25);

        let root = tiles.insert_container(lin);
        let tree = egui_tiles::Tree::new("my_tree", root, tiles);

        let tree_ctx = ViewerTree { context };
        Viewer { tree, tree_ctx }
    }

    fn url_button(&mut self, label: &str, url: &str, ui: &mut egui::Ui) {
        ui.add(Hyperlink::from_label_and_url(label, url).open_in_new_tab(true));
    }
}

impl eframe::App for Viewer {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        if let Some(rec) = self.tree_ctx.context.receiver.clone() {
            while let Ok(message) = rec.try_recv() {
                for (_, pane) in self.tree.tiles.iter_mut() {
                    match pane {
                        egui_tiles::Tile::Pane(pane) => {
                            pane.on_message(message.clone(), &mut self.tree_ctx.context);
                        }
                        egui_tiles::Tile::Container(_) => {}
                    }
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // Close when pressing escape (in a native viewer anyway).
            if ui.input(|r| r.key_pressed(egui::Key::Escape)) {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
            self.tree.ui(&mut self.tree_ctx, ui);
        });
    }
}
