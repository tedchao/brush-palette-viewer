//! Shader registry for brush-palette.

use brush_wgsl::wgsl_kernel;

#[wgsl_kernel(source = "src/shaders/stub.wgsl")]
pub struct Stub;

#[wgsl_kernel(source = "src/shaders/palette_remix.wgsl")]
pub struct PaletteRemix;

#[wgsl_kernel(
    source = "src/shaders/project_visible_weight.wgsl",
    includes = ["../brush-render/src/shaders/helpers.wgsl"],
)]
pub struct ProjectVisibleWeight;

#[wgsl_kernel(
    source = "src/shaders/rasterize_weight.wgsl",
    includes = ["../brush-render/src/shaders/helpers.wgsl"],
)]
pub struct RasterizeWeight;

#[wgsl_kernel(source = "src/shaders/our_tile_offsets.wgsl")]
pub struct OurTileOffsets;

#[wgsl_kernel(
    source = "src/shaders/populate_projected_for_map.wgsl",
    includes = ["../brush-render/src/shaders/helpers.wgsl"],
)]
pub struct PopulateProjectedForMap;