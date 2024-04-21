#import helpers;

struct Uniforms {
    // Total reachable pixels (w, h)
    tile_bounds: vec2u,
}

@group(0) @binding(0) var<storage> uniforms: Uniforms;
@group(0) @binding(1) var<storage> xys: array<vec2f>;
@group(0) @binding(2) var<storage> radii: array<u32>;
@group(0) @binding(3) var<storage> cum_tiles_hit: array<u32>;
@group(0) @binding(4) var<storage> depths: array<f32>;
@group(0) @binding(5) var<storage, read_write> isect_ids: array<u32>;
@group(0) @binding(6) var<storage, read_write> gaussian_ids: array<u32>;

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

    if radius == 0u {
        return;
    }

    let tile_bounds = uniforms.tile_bounds;

    // get the tile bbox for gaussian
    let center = xys[idx];
    let tile_minmax = helpers::get_tile_bbox(center, radius, tile_bounds);
    let tile_min = tile_minmax.xy;
    let tile_max = tile_minmax.zw;
    let depth = depths[idx];

    // update the intersection info for all tiles this gaussian hits
    var cur_idx = 0u;
    if idx > 0u {
        cur_idx = cum_tiles_hit[idx - 1u];
    }

    for (var ty = tile_min.y; ty < tile_max.y; ty++) {
        for (var tx = tile_min.x; tx < tile_max.x; tx++) {
            // assume tile_id is less than 65K (just about works out for a 4k screen).
            let tile_id = tx + ty * tile_bounds.x; // tile within image

            // Sort by, unfortunately low precision, depth.
            // Nb: Radix sort on floats is only correct if the sign is positive
            // but depths are known to be.
            
            // sign | 8 bit exp | 23 bits mantissa.
            // assume sign is 0. Then we have space for 8 bits mantiassa, so,
            // remove 15 bits.
            let depth_id = bitcast<u32>(depth) >> 15u;
            let packed_val = (tile_id << 16u) | depth_id;
            isect_ids[cur_idx] = packed_val;
            gaussian_ids[cur_idx] = idx; // 3D gaussian id
            cur_idx++; // handles gaussians that hit more than one tile
        }
    }
}
