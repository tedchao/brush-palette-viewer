#import helpers

@group(0) @binding(0) var<storage> sorted_tiled_tile_ids: array<u32>;
@group(0) @binding(1) var<storage> num_intersections: array<u32>;

@group(0) @binding(2) var<storage, read_write> tile_bins: array<vec2u>;

const VERTICAL_GROUPS: u32 = 64;

// kernel to map sorted intersection IDs to tile bins
// expect that intersection IDs are sorted by increasing tile ID
// i.e. intersections of a tile are in contiguous chunks
@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let isect_id = global_id.x * VERTICAL_GROUPS + global_id.y;
    let num_intersects = num_intersections[0];

    if isect_id >= num_intersects {
        return;
    }

    // Save the indices where the tile_id changes
    let cur_tile_idx = sorted_tiled_tile_ids[isect_id];

    // handle edge cases.
    if isect_id == num_intersects - 1u {
        tile_bins[cur_tile_idx].y = num_intersects;
    }

    if isect_id == 0u {
        tile_bins[cur_tile_idx].x = 0u;
    } else {
        let prev_tile_idx = sorted_tiled_tile_ids[isect_id - 1u];

        if prev_tile_idx != cur_tile_idx {
            tile_bins[prev_tile_idx].y = isect_id;
            tile_bins[cur_tile_idx].x = isect_id;
        }
    }
}
