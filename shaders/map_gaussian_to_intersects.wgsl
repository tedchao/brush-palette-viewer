#import helpers;

@group(0) @binding(0) var<storage, read> xys: array<vec2f>;
@group(0) @binding(1) var<storage, read> depths: array<f32>;
@group(0) @binding(2) var<storage, read> radii: array<i32>;
@group(0) @binding(3) var<storage, read> cum_tiles_hit: array<u32>;

@group(0) @binding(4) var<storage, read_write> isect_ids: array<vec2u>;
@group(0) @binding(5) var<storage, read_write> gaussian_ids: array<u32>;

@group(0) @binding(6) var<storage, read> info_array: array<Uniforms>;

struct Uniforms {
    // Number of splats that exist.
    num_intersections: u32,
    // Total reachable pixels (w, h)
    tile_bounds: vec2u,
    // Width of blocks image is divided into.
    block_width: u32,
}

// kernel to map each intersection from tile ID and depth to a gaussian
// writes output to isect_ids and gaussian_ids
@compute
@workgroup_size(16, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    let info = info_array[0];
    let num_intersections = info.num_intersections;
    let tile_bounds = info.tile_bounds;
    let block_width = info.block_width;

    let idx = global_id.x;
    
    if idx >= num_intersections {
        return;
    }
    
    if radii[idx] <= 0 {
        return;
    }

    // get the tile bbox for gaussian
    let center = xys[idx];
    let tile_minmax = helpers::get_tile_bbox(center, f32(radii[idx]), tile_bounds, block_width);
    let tile_min = tile_minmax.xy;
    let tile_max = tile_minmax.zw;

    // update the intersection info for all tiles this gaussian hits
    var cur_idx = 0u;
    if idx != 0 {
        cur_idx = cum_tiles_hit[idx - 1];
    }
    
    let depth_id = bitcast<u32>(depths[idx]);

    for (var i = tile_min.y; i < tile_max.y; i++) {
        for (var j = tile_min.x; j < tile_max.x; j++) {
            // isect_id is tile ID and depth as int32
            let tile_id = u32(i * i32(tile_bounds.x) + j); // tile within image

            // TODO: Would need 64 bits to do this properly.
            isect_ids[cur_idx].x = tile_id;
            isect_ids[cur_idx].y = depth_id;
            
            gaussian_ids[cur_idx] = idx;                     // 3D gaussian id
            cur_idx++; // handles gaussians that hit more than one tile
        }
    }
}