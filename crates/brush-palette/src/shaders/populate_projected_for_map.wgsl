// populate_projected_for_map.wgsl
//
// MapGaussiansToIntersect reads `projected: array<ProjectedSplat>` and uses
// `xy`, `conic`, and `color_a` (which is opacity) to determine which tiles
// each splat hits. Our ProjectVisibleWeight wrote a `ProjectedWeightSplat`
// instead. This shader copies the matching geometry fields from our struct
// into the format MapGaussiansToIntersect expects, so we can reuse Brush's
// existing tile-binning kernel.
//
// We don't fill colors (sh evaluation isn't relevant for tile-binning); we
// only need `color_a` (opacity) to compute power_threshold.

#import helpers;

struct ProjectedWeightSplat {
    xy_x:    f32,
    xy_y:    f32,
    conic_x: f32,
    conic_y: f32,
    conic_z: f32,
    opacity: f32,
    depth:   f32,
    pad:     f32,
    w0: f32, w1: f32, w2: f32, w3: f32,
    w4: f32, w5: f32, w6: f32, w7: f32,
}

@group(0) @binding(0) var<storage, read>       weight_splats   : array<ProjectedWeightSplat>;
@group(0) @binding(1) var<storage, read_write> projected_splats: array<helpers::ProjectedSplat>;

struct Uniforms {
    num_visible: u32,
    pad_a: u32,
    pad_b: u32,
    pad_c: u32,
}
@group(0) @binding(2) var<storage, read> uniforms : Uniforms;

const WG_SIZE: u32 = 256u;

@compute
@workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id)             wid: vec3u,
    @builtin(num_workgroups)           num_wgs: vec3u,
    @builtin(local_invocation_index)   lid: u32,
) {
    let compact_gid = helpers::get_global_id(wid, num_wgs, lid, WG_SIZE);
    if compact_gid >= uniforms.num_visible {
        return;
    }
    let w = weight_splats[compact_gid];

    let mean2d       = vec2f(w.xy_x, w.xy_y);
    let conic_packed = vec3f(w.conic_x, w.conic_y, w.conic_z);
    let color_alpha  = vec4f(0.0, 0.0, 0.0, w.opacity);

    projected_splats[compact_gid] = helpers::create_projected_splat(
        mean2d, conic_packed, color_alpha
    );
}
