#import helpers

@group(0) @binding(0) var<storage, read> isect_ids_sorted: array<u32>;
@group(0) @binding(1) var<storage, read_write> tile_bins: array<vec2u>;
@group(0) @binding(2) var<storage, read> info_array: array<Uniforms>;

struct Uniforms {
    // Number of intersections.
    num_intersects: u32,
}

// kernel to map sorted intersection IDs to tile bins
// expect that intersection IDs are sorted by increasing tile ID
// i.e. intersections of a tile are in contiguous chunks
@compute
@workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let info = info_array[0];
    let num_intersects = info.num_intersects;
    let idx = global_id.x;

    if idx >= num_intersects {
        return;
    }

    // Save the indices where the tile_id changes
    let cur_tile_idx = isect_ids_sorted[idx];

    // handle edge cases.
    if idx == num_intersects - 1 {
        tile_bins[cur_tile_idx].y = num_intersects;
    }

    if idx == 0 {
        tile_bins[cur_tile_idx].x = 0u;
    } else {
        let prev_tile_idx = isect_ids_sorted[idx - 1];

        if prev_tile_idx != cur_tile_idx {
            tile_bins[prev_tile_idx].y = idx;
            tile_bins[cur_tile_idx].x = idx;
        }
    }
}
