#import helpers;

struct Uniforms {
    // Total reachable pixels (w, h)
    tile_bounds: vec2u,
}

@group(0) @binding(0) var<storage> uniforms: Uniforms;
@group(0) @binding(1) var<storage> sorted_ids: array<u32>;
@group(0) @binding(2) var<storage> xys: array<vec2f>;
@group(0) @binding(3) var<storage> cov2ds: array<vec4f>;

@group(0) @binding(4) var<storage> radii: array<u32>;
@group(0) @binding(5) var<storage> cum_tiles_hit: array<u32>;
@group(0) @binding(6) var<storage> num_visible: array<u32>;

@group(0) @binding(7) var<storage, read_write> tile_ids: array<u32>;
@group(0) @binding(8) var<storage, read_write> gaussian_ids: array<u32>;


@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;

    if idx >= num_visible[0] {
        return;
    }

    let g_id = sorted_ids[idx];

    let radius = radii[g_id];
    let tile_bounds = uniforms.tile_bounds;

    // get the tile bbox for gaussian
    let xy = xys[g_id];
    let tile_minmax = helpers::get_tile_bbox(xy, radius, tile_bounds);
    let tile_min = tile_minmax.xy;
    let tile_max = tile_minmax.zw;

    let conic = helpers::cov2d_to_conic(cov2ds[g_id].xyz);

    // update the intersection info for all tiles this gaussian hits

    // Get exclusive prefix sum of tiles hit.
    var cur_idx = 0u;
    if idx > 0u {
        cur_idx = cum_tiles_hit[idx - 1u];
    }

    for (var ty = tile_min.y; ty < tile_max.y; ty++) {
        for (var tx = tile_min.x; tx < tile_max.x; tx++) {
            if helpers::can_be_visible(vec2u(tx, ty), xy, radius) {
                let tile_id = tx + ty * tile_bounds.x; // tile within image
                tile_ids[cur_idx] = tile_id;
                gaussian_ids[cur_idx] = g_id;
                cur_idx++; // handles gaussians that hit more than one tile
            }
        }
    }
}
