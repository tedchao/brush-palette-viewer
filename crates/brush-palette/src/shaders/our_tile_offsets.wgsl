// our_tile_offsets.wgsl
//
// Given a sorted array of tile_ids (one per intersection), compute per-tile
// [start, end) ranges. Drop-in replacement for brush-render's get_tile_offsets;
// avoids the cubecl macro signature mismatch we hit on macOS.
//
// Output layout: tile_offsets[ tile_id * 2 + 0 ] = start
//                tile_offsets[ tile_id * 2 + 1 ] = end
//
// One thread per intersection. Each thread compares its tile_id with the
// previous intersection's tile_id; transitions write the boundary.

@group(0) @binding(0) var<storage, read>       tile_id_from_isect : array<u32>;
@group(0) @binding(1) var<storage, read_write> tile_offsets       : array<u32>;

struct Uniforms {
    num_intersections : u32,
    pad_a: u32, pad_b: u32, pad_c: u32,
}
@group(0) @binding(2) var<storage, read> uniforms : Uniforms;

const WG_SIZE: u32 = 256u;

@compute
@workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let i = gid.x;
    let n = uniforms.num_intersections;
    if i >= n { return; }

    let tid = tile_id_from_isect[i];

    // First intersection: it's the start of its tile.
    if i == 0u {
        tile_offsets[tid * 2u + 0u] = 0u;
    }

    // Last intersection: it's the end of its tile.
    if i == n - 1u {
        tile_offsets[tid * 2u + 1u] = n;
    }

    // Boundary between tiles: write end of previous tile, start of current tile.
    if i > 0u {
        let prev_tid = tile_id_from_isect[i - 1u];
        if tid != prev_tid {
            tile_offsets[prev_tid * 2u + 1u] = i;
            tile_offsets[tid * 2u + 0u]      = i;
        }
    }
}
