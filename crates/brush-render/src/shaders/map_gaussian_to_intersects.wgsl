#import helpers;

struct Uniforms {
    // Total reachable pixels (w, h)
    tile_bounds: vec2u,
}

@group(0) @binding(0) var<storage> uniforms: Uniforms;
@group(0) @binding(1) var<storage> compact_from_depthsort_gid: array<u32>;
@group(0) @binding(2) var<storage> xys: array<vec2f>;
@group(0) @binding(3) var<storage> conic_comp: array<vec4f>;
@group(0) @binding(4) var<storage> colors: array<vec4f>;

@group(0) @binding(5) var<storage> cum_tiles_hit: array<u32>;
@group(0) @binding(6) var<storage> num_visible: array<u32>;

@group(0) @binding(7) var<storage, read_write> tile_id_from_isect: array<u32>;
@group(0) @binding(8) var<storage, read_write> depthsort_gid_from_isect: array<u32>;

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let depthsort_gid = global_id.x;

    if depthsort_gid >= num_visible[0] {
        return;
    }

    let compact_gid = compact_from_depthsort_gid[depthsort_gid];
    let conic = conic_comp[compact_gid].xyz;
    let opac = colors[compact_gid].w;

    let radius = helpers::radius_from_conic(conic, opac);

    let tile_bounds = uniforms.tile_bounds;

    // get the tile bbox for gaussian
    let xy = xys[compact_gid];
    let tile_minmax = helpers::get_tile_bbox(xy, u32(radius), tile_bounds);
    let tile_min = tile_minmax.xy;
    let tile_max = tile_minmax.zw;

    // Get exclusive prefix sum of tiles hit.
    var isect_id = 0u;
    if depthsort_gid > 0u {
        isect_id = cum_tiles_hit[depthsort_gid - 1u];
    }

    for (var ty = tile_min.y; ty < tile_max.y; ty++) {
        for (var tx = tile_min.x; tx < tile_max.x; tx++) {
            if helpers::can_be_visible(vec2u(tx, ty), xy, conic, opac) {
                let tile_id = tx + ty * tile_bounds.x; // tile within image
                tile_id_from_isect[isect_id] = tile_id;
                depthsort_gid_from_isect[isect_id] = depthsort_gid;
                isect_id++; // handles gaussians that hit more than one tile
            }
        }
    }
}
