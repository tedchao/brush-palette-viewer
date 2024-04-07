#import helpers;

@group(0) @binding(0) var<storage, read> xys: array<vec2f>;
@group(0) @binding(1) var<storage, read> radii: array<u32>;
@group(0) @binding(2) var<storage, read> cum_tiles_hit: array<u32>;

@group(0) @binding(3) var<storage, read_write> isect_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> gaussian_ids: array<u32>;

@group(0) @binding(5) var<storage, read> info_array: array<Uniforms>;

struct Uniforms {
    // Total reachable pixels (w, h)
    tile_bounds: vec2u,
}

// kernel to map each intersection from tile ID and depth to a gaussian
// writes output to isect_ids and gaussian_ids
@compute
@workgroup_size(helpers::SPLATS_PER_GROUP, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;

    if idx >= arrayLength(&radii) {
        return;
    }

    // Check if gaussian is visible.
    let radius = radii[idx];

    if radius == 0 {
        return;
    }

    let info = info_array[0];
    let tile_bounds = info.tile_bounds;

    // get the tile bbox for gaussian
    let center = xys[idx];
    let tile_minmax = helpers::get_tile_bbox(center, radius, tile_bounds);
    let tile_min = tile_minmax.xy;
    let tile_max = tile_minmax.zw;

    // update the intersection info for all tiles this gaussian hits
    var cur_idx = 0u;
    if idx > 0 {
        cur_idx = cum_tiles_hit[idx - 1];
    }

    for (var ty = tile_min.y; ty < tile_max.y; ty++) {
        for (var tx = tile_min.x; tx < tile_max.x; tx++) {
            let tile_id = tx + ty * tile_bounds.x; // tile within image
            isect_ids[cur_idx] = tile_id;
            gaussian_ids[cur_idx] = idx; // 3D gaussian id
            cur_idx++; // handles gaussians that hit more than one tile
        }
    }
}
