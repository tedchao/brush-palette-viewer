# brush-palette-viewer

A fork of [Brush](https://github.com/ArthurBrussee/brush) extended with palette-based 3D Gaussian Splatting rendering, on top of a custom representation that splits per-Gaussian color into a small fixed palette plus per-pixel weights. Includes a live palette editor.

The base Brush viewer is preserved for vanilla `.ply` files. Palette-based scenes use a new `.pply` + `.gswp` pair; the viewer auto-detects and routes to the palette pipeline.

## What's new

- New `brush-palette` crate containing four WGSL shaders (`project_visible_weight`, `rasterize_weight`, `palette_remix`, `our_tile_offsets`, `populate_projected_for_map`) and the dispatch glue that wires them into Brush.
- Loaders for `.pply` (geometry-only PLY) and `.gswp` (sidecar with palette + per-Gaussian weight SH).
- A floating "Palette" window with live RGB color pickers — edit colors and the scene re-renders at full FPS.
- FPS counter overlay on the render area.

The vanilla Brush rendering pipeline is untouched: vanilla `.ply` files load and render exactly as upstream.

## Files you need on disk

To render a palette-based scene, you need both files together (same directory, same basename):

```
your_scene.pply       # geometry-only PLY (positions, scales, opacities, rotations)
your_scene.gswp       # sidecar: palette colors + per-Gaussian weight SH coefficients
```

The `.pply` is a binary little-endian PLY missing the standard `f_dc_*` / `f_rest_*` color fields. The `.gswp` is a custom binary format with a 32-byte header (magic `GSWP`, version, dimensions) followed by the palette and weight tensors. See `crates/brush-palette/src/lib.rs` for the exact layout.

These are produced by a Python script (`palette_npy_to_ply.py`) that converts a trained palette-based model from its native `.npy` form. The viewer itself only consumes the `.pply` + `.gswp` pair.

## Building

Requires Rust (rustup-installed; stable toolchain is fine). Tested on macOS (Apple Silicon).

Debug build (faster compile, slower runtime):

```
cargo build --bin brush
```

Release build (slower compile, full performance):

```
cargo build --release --bin brush
```

Incremental rebuilds after editing only WGSL or `brush-palette` are typically ~10-20 seconds. A full clean release build is ~3-5 minutes.

## Running

The CLI is unchanged from upstream Brush. Pass either a vanilla `.ply` or a palette `.pply` as the file argument:

```
./target/debug/brush --with-viewer path/to/scene.pply
```

For the palette path to work, `scene.gswp` must sit alongside `scene.pply`.

### Example

```
cargo build --bin brush && RUST_LOG=info ./target/debug/brush --with-viewer ../ColorGradedGaussians/re-submission/convert-npy-ply/truck_palette.pply
```

This builds (debug), then loads `truck_palette.pply` plus the auto-discovered `truck_palette.gswp`. The viewer opens and the truck renders with palette-based colors. The "Palette" window appears in the top-right of the scene area; clicking a color swatch opens a picker, and dragging it updates the rendered colors live.

`RUST_LOG=info` enables informational logs from the loader and renderer; omit for a quiet run.

### Loading vanilla 3DGS

Vanilla `.ply` files load through the original Brush path:

```
./target/debug/brush --with-viewer path/to/vanilla_scene.ply
```

No `.gswp` needed; rendering uses Brush's stock shaders.

## Repository layout

The new code lives in `crates/brush-palette/`. The most important pieces:

```
crates/brush-palette/
├── Cargo.toml
└── src/
    ├── lib.rs                    # PaletteSplats, sidecar parser, format helpers
    ├── render.rs                 # render_palette() — orchestrates the 12-step palette pipeline
    ├── shaders.rs                # registers WGSL shaders via brush-wgsl macro
    └── shaders/
        ├── project_visible_weight.wgsl   # per-Gaussian SH→K-vector weight evaluation
        ├── rasterize_weight.wgsl         # tile-binned K-channel weight accumulator
        ├── palette_remix.wgsl            # per-pixel K-vector → RGB
        ├── our_tile_offsets.wgsl         # tile-offset compute (replaces the upstream cube macro)
        ├── populate_projected_for_map.wgsl  # adapter: ProjectedWeightSplat → ProjectedSplat
        └── stub.wgsl                     # toolchain validation only
```

Modifications outside `brush-palette`:

- `crates/brush-process/src/lib.rs` — branch on `.pply` extension, build `PaletteSplats`, emit `PaletteLoaded` message; vanilla path untouched.
- `crates/brush-process/src/message.rs` — new `PaletteLoaded { colors }` message variant.
- `crates/brush-ui/src/ui_process.rs` — second `Slot<PaletteSplats>` parallel to the existing splat slot; palette state.
- `crates/brush-ui/src/splat_backbuffer.rs` — render-worker branches between vanilla and palette render paths; FPS counter.
- `crates/brush-ui/src/scene.rs` — floating palette editor window.

## License

Original Brush code is Apache-2.0; modifications follow the same license.