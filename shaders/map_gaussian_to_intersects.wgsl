#import helpers;

@group(0) @binding(0) var<storage, read> xys: array<vec2f>;
@group(0) @binding(1) var<storage, read> depths: array<f32>;
@group(0) @binding(2) var<storage, read> radii: array<i32>;
@group(0) @binding(3) var<storage, read> cum_tiles_hit: array<u32>;

@group(0) @binding(4) var<storage, read_write> isect_ids: array<u32>;
@group(0) @binding(5) var<storage, read_write> gaussian_ids: array<u32>;

@group(0) @binding(6) var<storage, read_write> conics: array<vec3f>;
@group(0) @binding(7) var<storage, read_write> compensation: array<f32>;

@group(0) @binding(8) var<storage, read> info_array: array<helpers::InfoBinding>;

// kernel to map each intersection from tile ID and depth to a gaussian
// writes output to isect_ids and gaussian_ids
@compute
@workgroup_size(16, 1, 1)
fn map_gaussian_to_intersects(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    let info = info_array[0];
    let num_points = info.num_points;
    let tile_bounds = info.tile_bounds;
    let block_width = info.block_width;
    let idx = global_id.x;
    
    if idx >= num_points {
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
    
    // TODO: What in the ever loving god??
    // let depth_id = (int64_t) * (int32_t * )&(depths[idx]);
    let depth_id = 0u;

    for (var i = tile_min.y; i < tile_max.y; i++) {
        for (var j = tile_min.x; j < tile_max.x; j++) {
            // isect_id is tile ID and depth as int32
            let tile_id = u32(i * i32(tile_bounds.x) + j); // tile within image
            isect_ids[cur_idx] = (tile_id << 32) | depth_id; // tile | depth id
            gaussian_ids[cur_idx] = idx;                     // 3D gaussian id
            cur_idx++; // handles gaussians that hit more than one tile
        }
    }
}