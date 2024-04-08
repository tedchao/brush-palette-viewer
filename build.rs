use miette::{IntoDiagnostic, Result};
use wgsl_bindgen::{GlamWgslTypeMap, WgslBindgenOptionBuilder, WgslTypeSerializeStrategy};

fn main() -> Result<()> {
    WgslBindgenOptionBuilder::default()
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .emit_rerun_if_change(true)
        .short_constructor(0)
        .type_map(GlamWgslTypeMap)
        .workspace_root("shaders")
        .add_entry_point("shaders/project_forward.wgsl")
        .add_entry_point("shaders/map_gaussian_to_intersects.wgsl")
        .add_entry_point("shaders/get_tile_bin_edges.wgsl")
        .add_entry_point("shaders/rasterize.wgsl")
        .add_entry_point("shaders/rasterize_backwards.wgsl")
        .add_entry_point("shaders/project_backwards.wgsl")
        .add_entry_point("shaders/prefix_sum_scan.wgsl")
        .add_entry_point("shaders/prefix_sum_scan_sums.wgsl")
        .add_entry_point("shaders/prefix_sum_add_scanned_sums.wgsl")
        .output("src/splat_render/generated_bindings.rs")
        .build()?
        .generate()
        .into_diagnostic()
}
