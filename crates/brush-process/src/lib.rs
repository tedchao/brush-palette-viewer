#![recursion_limit = "256"]

use burn_wgpu::{
    RuntimeOptions, WgpuDevice,
    graphics::{AutoGraphicsApi, GraphicsApi},
};
use wgpu::{Adapter, Device, Queue};

pub mod config;
pub mod message;

#[cfg(feature = "training")]
pub mod train_stream;

#[cfg(feature = "training")]
pub mod args_file;

pub mod slot;

use std::pin::Pin;

use anyhow::Error;
use async_fn_stream::try_fn_stream;
use brush_render::MainBackend;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_vfs::{DataSource, SendNotWasm};
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::WgpuRuntime;
use tokio_stream::{Stream, StreamExt};

fn burn_options() -> RuntimeOptions {
    RuntimeOptions {
        tasks_max: 64,
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
    }
}

pub async fn burn_init_setup() -> WgpuDevice {
    burn_wgpu::init_setup_async::<AutoGraphicsApi>(&WgpuDevice::DefaultDevice, burn_options())
        .await;
    connect_device(WgpuDevice::DefaultDevice);
    WgpuDevice::DefaultDevice
}

pub fn burn_init_device(adapter: Adapter, device: Device, queue: Queue) -> WgpuDevice {
    let setup = burn_wgpu::WgpuSetup {
        instance: wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()), // unused... need to fix this in Burn.
        adapter,
        device,
        queue,
        backend: AutoGraphicsApi::backend(),
    };
    let burn = burn_wgpu::init_device(setup, burn_options());
    connect_device(burn.clone());
    burn
}

use crate::{message::ProcessMessage, slot::Slot};

pub trait ProcessStream: Stream<Item = Result<ProcessMessage, Error>> + SendNotWasm {}
impl<T> ProcessStream for T where T: Stream<Item = Result<ProcessMessage, Error>> + SendNotWasm {}

pub struct RunningProcess {
    pub stream: Pin<Box<dyn ProcessStream>>,
    pub splat_view: Slot<Splats<MainBackend>>,
    pub palette_view: Slot<brush_palette::PaletteSplats<MainBackend>>,
}

use tokio::sync::SetOnce;

static DEVICE: SetOnce<WgpuDevice> = SetOnce::const_new();

pub(crate) fn connect_device(device: WgpuDevice) {
    DEVICE.set(device).unwrap();
}

/// Try to load and parse the .gswp sidecar associated with a .pply path.
/// The sidecar is first looked up inside the VFS (a picked directory or zip
/// containing both files; this is the only option on WASM). On native, it
/// falls back to the local filesystem next to `fs_pply_path`, which covers
/// opening a single .pply file whose VFS contains only that file.
/// Returns Ok(None) if the sidecar file simply doesn't exist; Err for any
/// I/O or parse failure.
async fn load_palette_sidecar(
    vfs: &brush_vfs::BrushVfs,
    vfs_pply_path: &std::path::Path,
    #[cfg_attr(target_family = "wasm", allow(unused_variables))] fs_pply_path: Option<
        &std::path::Path,
    >,
    expected_n_splats: u32,
) -> Result<Option<brush_palette::PaletteSidecar>, anyhow::Error> {
    use tokio::io::AsyncReadExt;

    if let Some(sidecar_path) = brush_palette::sidecar_path_for(vfs_pply_path)
        && let Ok(mut reader) = vfs.reader_at_path(&sidecar_path).await
    {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let sc = brush_palette::PaletteSidecar::parse(&bytes, expected_n_splats)?;
        return Ok(Some(sc));
    }

    #[cfg(not(target_family = "wasm"))]
    if let Some(sidecar_path) = fs_pply_path.and_then(brush_palette::sidecar_path_for)
        && tokio::fs::try_exists(&sidecar_path).await.unwrap_or(false)
    {
        let bytes = tokio::fs::read(&sidecar_path).await?;
        let sc = brush_palette::PaletteSidecar::parse(&bytes, expected_n_splats)?;
        return Ok(Some(sc));
    }

    Ok(None)
}

// On WASM, DynRead is not Send (single-threaded runtime; no OS threads).
// On native, DynRead is Send, so we require it for the stream to be Send.
#[cfg(not(target_family = "wasm"))]
fn pin_splat_stream(
    s: impl tokio_stream::Stream<Item = Result<brush_serde::SplatMessage, brush_serde::DeserializeError>>
        + Send
        + 'static,
) -> Pin<
    Box<
        dyn tokio_stream::Stream<
                Item = Result<brush_serde::SplatMessage, brush_serde::DeserializeError>,
            > + Send,
    >,
> {
    Box::pin(s)
}

#[cfg(target_family = "wasm")]
fn pin_splat_stream(
    s: impl tokio_stream::Stream<Item = Result<brush_serde::SplatMessage, brush_serde::DeserializeError>>
        + 'static,
) -> Pin<
    Box<
        dyn tokio_stream::Stream<
            Item = Result<brush_serde::SplatMessage, brush_serde::DeserializeError>,
        >,
    >,
> {
    Box::pin(s)
}

/// Replace the SH coefficients of `splats` with degree-0 only, set to the
/// provided baked DC values. Drops any existing higher-band coefficients.
/// `dc_colors` is length N*3, layout [r,g,b,r,g,b,...].
fn inject_dc_colors(
    mut splats: Splats<MainBackend>,
    dc_colors: Vec<f32>,
    device: &WgpuDevice,
) -> Splats<MainBackend> {
    use burn::module::{Param, ParamId};
    use burn::prelude::*;

    let n_splats = splats.num_splats() as usize;
    debug_assert_eq!(dc_colors.len(), n_splats * 3);

    // Build (N, 1, 3) tensor: degree-0 SH only, RGB.
    let new_sh = Tensor::<MainBackend, 3>::from_data(
        burn::tensor::TensorData::new(dc_colors, [n_splats, 1, 3]),
        device,
    );

    splats.sh_coeffs = Param::initialized(ParamId::new(), new_sh.detach().require_grad());
    splats
}

/// Create a running process from a datasource and args.
///
/// The `config_fn` callback receives the initial config (loaded from args.txt if present,
/// otherwise defaults) and returns the final config to use. This allows the caller to
/// modify or override settings as needed.
pub fn create_process<
    #[cfg(feature = "training")] Fun: FnOnce(crate::config::TrainStreamConfig) -> Fut + Send + 'static,
    #[cfg(feature = "training")] Fut: std::future::Future<Output = crate::config::TrainStreamConfig> + Send,
>(
    source: DataSource,
    #[cfg(feature = "training")] config_fn: Fun,
) -> RunningProcess {
    let splat_view = Slot::default();
    let splat_state_cl = splat_view.clone();
    let palette_view: Slot<brush_palette::PaletteSplats<MainBackend>> = Slot::default();
    let palette_view_cl = palette_view.clone();

    let stream = try_fn_stream(|emitter| async move {
        log::info!("Starting process with source {source:?}");
        emitter.emit(ProcessMessage::NewProcess).await;

        // Wait until the devise is set.
        let device = DEVICE.wait().await.clone();

        let vfs = source.clone().into_vfs().await?;
        let vfs_counts = vfs.file_count();

        if vfs_counts == 0 {
            return Err(anyhow::anyhow!("No files found."));
        }

        let ply_count = vfs.files_with_extension("ply").count()
            + vfs.files_with_extension("pply").count();
        // .gswp palette sidecars accompany .pply files; they don't make this a dataset.
        let sidecar_count = vfs.files_with_extension("gswp").count();

        log::info!(
            "Mounted VFS with {} files. (plys: {})",
            vfs.file_count(),
            ply_count
        );

        let is_training = vfs_counts != ply_count + sidecar_count;

        // Emit source info - just the display name
        let paths: Vec<_> = vfs.file_paths().collect();
        let source_name = if let Some(base_path) = vfs.base_path() {
            base_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(if is_training { "dataset" } else { "file" })
                .to_owned()
        } else if paths.len() == 1 {
            paths[0]
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("input.ply")
                .to_owned()
        } else {
            format!("{} files", paths.len())
        };

        let base_path = vfs.base_path();

        // Load initial config from args.txt via VFS if present
        #[cfg(feature = "training")]
        let initial_config = crate::args_file::load_config_from_vfs(&vfs).await;
        
        // Capture original source path for sidecar resolution (Stage B), as a
        // filesystem fallback when the sidecar isn't inside the VFS.
        let source_path: Option<std::path::PathBuf> = match &source {
            brush_vfs::DataSource::Path(s) => Some(std::path::PathBuf::from(s)),
            _ => None,
        };
        
        emitter
            .emit(ProcessMessage::StartLoading {
                name: source_name,
                source,
                training: is_training,
                base_path,
            })
            .await;

        if !is_training {
            let mut paths: Vec<_> = vfs
                .file_paths()
                .filter(|p| !p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gswp")))
                .collect();
            alphanumeric_sort::sort_path_slice(&mut paths);
            let client = WgpuRuntime::client(&device);
            let total_frames = paths.len() as u32;

            for (frame, path) in paths.iter().enumerate() {
                log::info!("Loading single ply file");

                let is_palette = brush_palette::is_pply_path(path);
                if is_palette {
                    log::info!("Detected palette-based .pply file");
                    if let Some(sidecar) = brush_palette::sidecar_path_for(path) {
                        log::info!("Sidecar expected at: {}", sidecar.display());
                    }
                }

                let mut splat_stream = if is_palette {
                    pin_splat_stream(brush_palette::stream_pply_as_geometry(
                        vfs.reader_at_path(path).await?,
                        None,
                        true,
                    ))
                } else {
                    pin_splat_stream(brush_serde::stream_splat_from_ply(
                        vfs.reader_at_path(path).await?,
                        None,
                        true,
                    ))
                };

                while let Some(message) = splat_stream.next().await {
                    
                    // For the first frame of a new file, clear existing frames
                    if frame == 0 {
                        splat_view.clear().await;
                        palette_view.clear().await;
                    }
                    
                    let message = message?;

                    let mode = message.meta.render_mode.unwrap_or(SplatRenderMode::Default);
                    let mut splats = message.data.into_splats(&device, mode);

                    // Stage B / C2.5b1: if this is a palette .pply, load the
                    // sidecar, build a PaletteSplats, probe-call render_palette
                    // (stub), then fall back to Stage B's bake-DC for display
                    // until C2.5c hooks render_palette into the per-frame loop.
                    if is_palette {
                        match load_palette_sidecar(
                            &vfs,
                            path,
                            source_path.as_deref(),
                            splats.num_splats(),
                        )
                        .await
                        {
                            Ok(Some(sidecar)) => {
                                log::info!(
                                    "Loaded sidecar: K_full={}, num_sh={}, palette[0]=({:.3},{:.3},{:.3})",
                                    sidecar.k_full,
                                    sidecar.num_sh,
                                    sidecar.palette[0],
                                    sidecar.palette[1],
                                    sidecar.palette[2],
                                );
                                
                                // Notify UI of palette colors for the edit panel.
                                let palette_colors_for_ui: Vec<[f32; 3]> = sidecar
                                    .palette
                                    .chunks_exact(3)
                                    .map(|c| [c[0], c[1], c[2]])
                                    .collect();
                                emitter
                                    .emit(ProcessMessage::PaletteLoaded { colors: palette_colors_for_ui })
                                    .await;
                                
                                // Stage B fallback DC for display.
                                let baked = sidecar.bake_dc_colors(true);

                                // C2.5a: build PaletteSplats with all tensors uploaded.
                                // We move the sidecar in here, so do this after `bake_dc_colors`.
                                let palette_splats = brush_palette::PaletteSplats::from_parts(
                                    splats.clone(),
                                    sidecar,
                                    &device,
                                )?;

                                // C2.5c1: store the PaletteSplats in the parallel slot.
                                // Display still uses Stage B's bake-DC fallback below.
                                palette_view.set_at(frame, palette_splats).await;
                                log::info!("PaletteSplats stored in palette_view");

                                // Stage B fallback for display path.
                                splats = inject_dc_colors(splats, baked, &device);
                            }
                            Ok(None) => {
                                log::warn!(
                                    "Palette .pply but no sidecar found alongside; rendering as gray geometry"
                                );
                            }
                            Err(e) => {
                                log::error!(
                                    "Sidecar parse failed: {}; rendering as gray geometry",
                                    e
                                );
                            }
                        }
                    }

                    // As loading concatenates splats each time, memory usage tends to accumulate a lot
                    // over time. Clear out memory after each step to prevent this buildup.
                    client.memory_cleanup();

                    // Capture stats before moving splats
                    let num_splats = splats.num_splats();
                    let sh_degree = splats.sh_degree();
                    splat_view.set_at(frame, splats).await;

                    emitter
                        .emit(ProcessMessage::SplatsUpdated {
                            up_axis: message.meta.up_axis,
                            frame: frame as u32,
                            total_frames,
                            num_splats,
                            sh_degree,
                        })
                        .await;
                }
            }
            
            emitter.emit(ProcessMessage::DoneLoading).await;
        } else {
            #[cfg(feature = "training")]
            {
                // Pass initial config (from args.txt or defaults) to the callback
                let base_config = initial_config.unwrap_or_default();
                let config = config_fn(base_config).await;
                crate::train_stream::train_stream(vfs, config, device, emitter, splat_view).await?;
            }

            #[cfg(not(feature = "training"))]
            anyhow::bail!("Training is not enabled in Brush, cannot load dataset.");
        };

        Ok(())
    });

    RunningProcess {
        stream: Box::pin(stream),
        splat_view: splat_state_cl,
        palette_view: palette_view_cl,
    }
}