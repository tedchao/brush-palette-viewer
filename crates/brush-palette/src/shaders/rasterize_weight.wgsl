// rasterize_weight.wgsl
//
// Tile-binned alpha-blend rasterizer for K_full-channel weight accumulation.
// Mirrors the FORWARD path of CUDA `draw_weight` but uses Brush's tile/work-
// group scaffolding (which is already wired into the rest of the pipeline).
//
// Per pixel, accumulate:
//   final_weight[k] = sum over splats t of  T(t) * alpha(t) * splat[t].weight[k]
//   T(t+1) = T(t) * (1 - alpha(t))
// terminating when T <= 1e-4.
//
// Output buffer is pixel-major to match palette_remix.wgsl:
//   out[(y*W + x)*MAX_K_FULL + k]   for k in 0..MAX_K_FULL
// Tail (k >= k_full) is written as 0.

#import helpers

// Static caps. Must match project_visible_weight.wgsl.
const MAX_K_FULL: u32 = 8u;

// Mirror of the struct laid out by project_visible_weight.wgsl.
// Same field order, same alignment behavior (no nested arrays).
struct ProjectedWeightSplat {
    xy_x:    f32,
    xy_y:    f32,
    conic_x: f32,
    conic_y: f32,
    conic_z: f32,
    opacity: f32,
    depth:   f32,
    pad:     f32,
    w0: f32, w1: f32, w2: f32, w3: f32,
    w4: f32, w5: f32, w6: f32, w7: f32,
}

@group(0) @binding(0) var<storage, read>       compact_gid_from_isect : array<u32>;
@group(0) @binding(1) var<storage, read>       tile_offsets           : array<u32>;
@group(0) @binding(2) var<storage, read>       projected              : array<ProjectedWeightSplat>;
@group(0) @binding(3) var<storage, read_write> out_weights            : array<f32>;
@group(0) @binding(4) var<storage, read_write> out_alpha              : array<f32>;
struct RasterWeightUniforms {
    tile_bounds: vec2u,
    img_size:    vec2u,
    k_full:      u32,
    pad_a:       u32,
    pad_b:       u32,
    pad_c:       u32,
}
@group(0) @binding(5) var<storage, read> uniforms : RasterWeightUniforms;

var<workgroup> range_uniform: vec2u;
var<workgroup> local_batch: array<ProjectedWeightSplat, helpers::TILE_SIZE>;
var<workgroup> num_done_atomic: atomic<u32>;

@compute
@workgroup_size(helpers::TILE_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3u,
    @builtin(num_workgroups) num_wgs: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let global_id   = helpers::get_global_id(wg_id, num_wgs, local_idx, helpers::TILE_SIZE);
    let pix_loc     = helpers::map_1d_to_2d(global_id, uniforms.tile_bounds.x);
    let pix_id      = pix_loc.x + pix_loc.y * uniforms.img_size.x;
    let pixel_coord = vec2f(pix_loc) + 0.5f;
    let tile_loc    = vec2u(pix_loc.x / helpers::TILE_WIDTH,
                            pix_loc.y / helpers::TILE_WIDTH);
    let tile_id     = tile_loc.x + tile_loc.y * uniforms.tile_bounds.x;
    let inside      = pix_loc.x < uniforms.img_size.x && pix_loc.y < uniforms.img_size.y;

    range_uniform = vec2u(
        tile_offsets[tile_id * 2],
        tile_offsets[tile_id * 2 + 1],
    );
    let range = workgroupUniformLoad(&range_uniform);

    var T = 1.0;
    // K accumulators in registers. WGSL doesn't let us index with a runtime
    // var into an array easily inside a hot loop, so we keep them as scalars
    // and write to the output buffer at the end.
    var w0_out: f32 = 0.0;
    var w1_out: f32 = 0.0;
    var w2_out: f32 = 0.0;
    var w3_out: f32 = 0.0;
    var w4_out: f32 = 0.0;
    var w5_out: f32 = 0.0;
    var w6_out: f32 = 0.0;
    var w7_out: f32 = 0.0;

    var done = !inside;

    if local_idx == 0u { atomicStore(&num_done_atomic, 0u); }
    workgroupBarrier();
    if done { atomicAdd(&num_done_atomic, 1u); }
    workgroupBarrier();

    for (var batch_start = range.x; batch_start < range.y; batch_start += helpers::TILE_SIZE) {
        if atomicLoad(&num_done_atomic) >= helpers::TILE_SIZE { break; }

        let remaining = min(helpers::TILE_SIZE, range.y - batch_start);

        let load_isect_id = batch_start + local_idx;
        var compact_gid = 0u;
        if local_idx < remaining {
            compact_gid = compact_gid_from_isect[load_isect_id];
        }

        workgroupBarrier();
        if local_idx < remaining {
            local_batch[local_idx] = projected[compact_gid];
        }
        workgroupBarrier();

        let was_done = done;
        for (var t = 0u; !done && t < remaining; t++) {
            let proj = local_batch[t];

            let xy    = vec2f(proj.xy_x, proj.xy_y);
            let conic = vec3f(proj.conic_x, proj.conic_y, proj.conic_z);
            let delta = xy - pixel_coord;
            let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y)
                       + conic.y * delta.x * delta.y;
            let alpha = min(0.999f, proj.opacity * exp(-sigma));

            if sigma >= 0.0f && alpha >= 1.0f / 255.0f {
                let next_T = T * (1.0 - alpha);
                if next_T <= 1e-4f {
                    done = true;
                    break;
                }
                let vis = alpha * T;
                w0_out = w0_out + vis * proj.w0;
                w1_out = w1_out + vis * proj.w1;
                w2_out = w2_out + vis * proj.w2;
                w3_out = w3_out + vis * proj.w3;
                w4_out = w4_out + vis * proj.w4;
                w5_out = w5_out + vis * proj.w5;
                w6_out = w6_out + vis * proj.w6;
                w7_out = w7_out + vis * proj.w7;
                T = next_T;
            }
        }
        if !was_done && done {
            atomicAdd(&num_done_atomic, 1u);
        }
    }

    if inside {
        let base = pix_id * MAX_K_FULL;
        out_weights[base + 0u] = w0_out;
        out_weights[base + 1u] = w1_out;
        out_weights[base + 2u] = w2_out;
        out_weights[base + 3u] = w3_out;
        out_weights[base + 4u] = w4_out;
        out_weights[base + 5u] = w5_out;
        out_weights[base + 6u] = w6_out;
        out_weights[base + 7u] = w7_out;
        out_alpha[pix_id] = 1.0 - T;
    }
}
