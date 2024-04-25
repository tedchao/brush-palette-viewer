#import helpers;

struct Uniforms {
    // Total reachable pixels (w, h)
    tile_bounds: vec2u,
}


@group(0) @binding(0) var<storage> uniforms: Uniforms;
@group(0) @binding(1) var<storage> xys: array<vec2f>;
@group(0) @binding(2) var<storage> cov2ds: array<vec4f>;
@group(0) @binding(3) var<storage> opacities: array<f32>;
@group(0) @binding(4) var<storage> radii: array<u32>;
@group(0) @binding(5) var<storage> gassuan_ids_sorted: array<u32>;


@group(0) @binding(6) var<storage, read_write> tile_min_gid: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> tile_max_gid: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> tile_counts: array<atomic<u32>>;

// kernel to map each intersection from tile ID and depth to a gaussian
// writes output to isect_ids and gaussian_ids
@compute
@workgroup_size(helpers::SPLATS_PER_GROUP, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;
    let num_gaussians = arrayLength(&radii);

    if idx >= num_gaussians {
        return;
    }

    let g_id = gassuan_ids_sorted[idx];

    // Check if gaussian is visible.
    let radius = radii[g_id];
    if radius == 0u {
        return;
    }

    let tile_bounds = uniforms.tile_bounds;

    // get the tile bbox for gaussian
    let center = xys[g_id];
    let tile_minmax = helpers::get_tile_bbox(center, radius, tile_bounds);
    let tile_min = tile_minmax.xy;
    let tile_max = tile_minmax.zw;

    let cov2d = cov2ds[g_id].xyz;
    let conic = helpers::cov2d_to_conic(cov2d);
    let xy = xys[g_id];
    let opac = opacities[g_id];

    let bucket_size = (num_gaussians + helpers::BUCKET_COUNT - 1u) / helpers::BUCKET_COUNT;

    for (var ty = tile_min.y; ty < tile_max.y; ty++) {
        for (var tx = tile_min.x; tx < tile_max.x; tx++) {
            // assume tile_id is less than 65K (just about works out for a 4k screen).
            let tile_id = tx + ty * tile_bounds.x; // tile within image
            let tile_start = vec2f(vec2u(tx, ty) * helpers::TILE_WIDTH);

            if helpers::can_be_visible(tile_start, conic, xy, opac) {
                let bucket_id = tile_id * helpers::BUCKET_COUNT + idx / bucket_size;

                atomicMin(&tile_min_gid[bucket_id], idx);
                // Exclusive maximum index.
                atomicMax(&tile_max_gid[bucket_id], idx + 1u);

                // Just for debug. Keep per tile.
                atomicAdd(&tile_counts[tile_id], 1u);
            }
        }
    }
}
