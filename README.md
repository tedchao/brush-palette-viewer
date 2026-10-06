# brush-palette-viewer

A real-time editor for palette-based 3D Gaussian Splatting, forked from [Brush](https://github.com/ArthurBrussee/brush). Edit palette colors and tone curves with the optimizer-driven solver running live in the loop.

**Tested on macOS Apple Silicon only.**

## Quickstart

### 1. Clone

```bash
git clone https://github.com/tedchao/brush-palette-viewer.git
cd brush-palette-viewer
```

### 2. Build

```bash
cargo build --bin brush
```

First build takes a few minutes. Incremental rebuilds are 10-20 seconds.

### 3. Run

Each scene lives in `models/<name>/` as a `name.pply` + `name.gswp` pair. Pass the `.pply` path; the `.gswp` is auto-discovered alongside it.

```bash
./target/debug/brush --with-viewer models/statue/statue.pply
```

The view recenters on the scene automatically once it has loaded. **Recenter view** in the settings (⚙) does the same on demand.

## Controls

- **Left drag** orbit, **right drag** look around, **middle drag** pan, **scroll** zoom
- **WASD** / **QE** fly, **shift** for faster movement
- **C (shift+c)** toggle pixel-click constraint mode → click in the viewport to add an image-space constraint
- **F** fullscreen

## UI overview

- **Palette window** (top right): click any palette swatch to edit colors. Check "Edit tone curves" to overlay and edit per-palette tone curves. Edits trigger the constraint optimizer in real time.
- **Image-space constraints window**: appears once you place pixel constraints. Each row shows the original → target color; click the target to edit, click the original to jump back to the saved view, 💾 to export the swatch pair, ✕ to remove.
- **Top bar buttons**: 📷 save current view as PNG, ⚖ save K weight images (RGBA, one per palette index) to a folder, ⚙ open settings (FOV, splat scale, fly speed, grid, background color, auto-rotate turntable, recenter, reset layout).
- **Save palette** in the palette window: exports the palette as PNG (horizontal or vertical, color-only or with curves).

## Vanilla `.ply` files

The original Brush rendering path is untouched. Vanilla `.ply` files (without an adjacent `.gswp`) render through the upstream Brush shaders.

```bash
./target/debug/brush --with-viewer path/to/vanilla.ply
```

## Web version

The viewer also runs in the browser via WebAssembly and WebGPU. The
`.github/workflows/pages.yml` workflow builds it and deploys it to GitHub Pages
(`https://<owner>.github.io/<repo>/`) on every push to the default branch; set
**Settings → Pages → Source** to **GitHub Actions** once to enable it.

**Browsers:** tested in Chrome, Firefox, and Safari 27. Safari 26 loads
scenes but renders them black.

**Loading scenes:** use **File**, **Directory**, or **URL**, or pass
`?url=<link>` in the page address. A `.pply` needs its `.gswp` to show colors:

- open a folder or `.zip` containing both files, or
- load the `.pply` from a URL; `foo.gswp` is fetched from the same location.
  Files on another server must allow cross-origin requests (CORS).

A single `.pply` picked with **File** renders as gray geometry, since the
browser can't see the file next to it.

**Differences from the desktop app:** the constraint optimizer runs
single-threaded, and saving weight images (⚖) isn't available.

**Building locally** needs the `wasm32-unknown-unknown` target from rustup (a
Homebrew `rust` install has none), [`wasm-pack`](https://github.com/drager/wasm-pack),
and Node.js:

```bash
rustup target add wasm32-unknown-unknown
npm install
cd brush_nextjs
npm run build:wasm-release   # WASM → brush_nextjs/pkg
npx next build --turbopack   # static site → brush_nextjs/out
python3 -m http.server -d out 8000
```

Then open <http://localhost:8000/>. `npm run dev` instead starts a dev server
with a debug WASM build.

## The constraint optimizer

Palette and tone-curve edits are solved by a native Rust optimizer in
`crates/brush-palette/src/optimizer.rs` — a coupled block-coordinate-descent
solver with IRLS, using [`faer`](https://faer-rs.github.io/) for the sparse and
dense linear solves. It runs in-process with no Python dependency, so building
and running the viewer needs nothing beyond the Rust toolchain.

The original reference implementation is kept at
`crates/brush-palette/python/constraint_optimizer.py`. A differential test runs
the Rust port and the Python reference on identical inputs and checks that the
`dP` (palette deltas) and `L` (tone curves) outputs agree:

```bash
cargo test -p brush-palette --test compare_python -- --nocapture
```

The test uses [`uv`](https://docs.astral.sh/uv/) to run the reference — numpy
and scipy are declared as PEP 723 inline dependencies in the driver script and
fetched automatically. If `uv` is not installed the test skips rather than
fails.

## License

Apache 2.0, following upstream Brush.