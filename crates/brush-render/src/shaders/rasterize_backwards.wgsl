#import helpers;

@group(0) @binding(0) var<storage, read_write> uniforms: helpers::RenderUniforms;

@group(0) @binding(1) var<storage, read_write> compact_gid_from_isect: array<u32>;
@group(0) @binding(2) var<storage, read_write> tile_bins: array<vec2u>;

@group(0) @binding(3) var<storage, read_write> projected_splats: array<helpers::ProjectedSplat>;

@group(0) @binding(4) var<storage, read_write> final_index: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> v_output: array<vec4f>;

@group(0) @binding(7) var<storage, read_write> scatter_grads: array<grads::ScatterGradient>;

@group(0) @binding(8) var<storage, read_write> cum_tiles_hit: array<u32>;

@group(0) @binding(9) var<storage, read_write> hit_ids: array<atomic<u32>>;


const GRAD_MEM_COUNT = helpers::TILE_SIZE / 2;

var<workgroup> v_conic_local: array<vec3f, GRAD_MEM_COUNT>;
var<workgroup> v_xy_local: array<vec2f, GRAD_MEM_COUNT>;
var<workgroup> v_colors_local: array<vec4f, GRAD_MEM_COUNT>;

var<workgroup> tile_bins_wg: vec2u;

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(helpers::TILE_WIDTH, helpers::TILE_WIDTH, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    let background = uniforms.background.xyz;
    let img_size = uniforms.img_size;

    let tile_loc = workgroup_id.xy;
    let tile_id = tile_loc.x + tile_loc.y * uniforms.tile_bounds.x;
    let pix_id = global_id.x + global_id.y * img_size.x;
    let pixel_coord = vec2f(global_id.xy) + 0.5;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    let inside = global_id.x < img_size.x && global_id.y < img_size.y;

    // this is the T AFTER the last gaussian in this pixel
    let T_final = 1.0 - output[pix_id].w;
    
    // the contribution from gaussians behind the current one

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between bin_start and bin_final in batches
    // which gaussians to look through in this tile
    if local_idx == 0u {
        tile_bins_wg = tile_bins[tile_id];
    }
    var range = workgroupUniformLoad(&tile_bins_wg);
    // var range = tile_bins[tile_id];
    let num_batches = helpers::ceil_div(range.y - range.x, helpers::TILE_SIZE);
    // current visibility left to render
    var T = T_final;

    var bin_final = 0u;
    var buffer = vec3f(0.0);

    if inside {
        bin_final = final_index[pix_id];
    }

    // df/d_out for this pixel
    let v_out = v_output[pix_id];


    for (var t = 0u; t < range.y - range.x; t++) {
        let isect_id = range.y - 1u - t;

        var v_xy = vec2f(0.0);
        var v_conic = vec3f(0.0);
        var v_colors = vec4f(0.0);

        if inside && isect_id <= bin_final {
            let projected = projected_splats[compact_gid_from_isect[isect_id]];
            let xy = vec2f(projected.x, projected.y);
            let conic = vec3f(projected.conic_x, projected.conic_y, projected.conic_z);
            let color = vec4f(projected.r, projected.g, projected.b, projected.a);

            let delta = xy - pixel_coord;
            let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            let vis = exp(-sigma);
            let alpha = min(0.99f, color.w * vis);

            // Nb: Don't continue; here - local_idx == 0 always
            // needs to write out gradients.
            // compute the current T for this gaussian
            if (sigma >= 0.0 && alpha >= 1.0 / 255.0) {
                let ra = 1.0 / (1.0 - alpha);
                T *= ra;
                // update v_colors for this gaussian
                let fac = alpha * T;
                var v_alpha = 0.0;

                // contribution from this pixel
                v_alpha += dot(color.xyz * T - buffer * ra, v_out.xyz);
                v_alpha += T_final * ra * v_out.w;
                // contribution from background pixel
                v_alpha -= dot(T_final * ra * background, v_out.xyz);

                // update the running sum
                buffer += color.xyz * fac;

                let v_sigma = -color.w * vis * v_alpha;
                
                v_xy = v_sigma * vec2f(
                    conic.x * delta.x + conic.y * delta.y, 
                    conic.y * delta.x + conic.z * delta.y
                );

                v_conic = vec3f(0.5f * v_sigma * delta.x * delta.x, 
                                                v_sigma * delta.x * delta.y,
                                                0.5f * v_sigma * delta.y * delta.y);

                v_colors = vec4f(fac * v_out.xyz, vis * v_alpha);
            }
        }

        // Parallel reduction
        var stride = helpers::TILE_SIZE / 2u;

        // Wait for all results to be written.
        workgroupBarrier();
        // Write the second half of the sum reduction to shared memory.
        if local_idx >= stride {
            v_xy_local[local_idx - stride] = v_xy;
            v_conic_local[local_idx - stride] = v_conic;
            v_colors_local[local_idx - stride] = v_colors;
        }
        // Wait for all results to be written.
        workgroupBarrier();

        // Now add in the first half of the sum reduction.
        if (local_idx < stride) {
            v_xy_local[local_idx] += v_xy;
            v_conic_local[local_idx] += v_conic;
            v_colors_local[local_idx] += v_colors;
        }
        workgroupBarrier();
        stride /= 2u;

        // Sum reduce to a final sum.
        while stride >= 16u {
            if (local_idx < stride) {
                v_colors_local[local_idx] += v_colors_local[local_idx + stride];
                v_conic_local[local_idx] += v_conic_local[local_idx + stride];
                v_xy_local[local_idx] += v_xy_local[local_idx + stride];
            }

            workgroupBarrier();
            stride = stride / 2u;
        }

        if local_idx == 0 {
            var sum_xy = v_xy_local[0];
            var sum_conic = v_conic_local[0];
            var sum_colors = v_colors_local[0];

            for(var i = 1u; i < stride * 2u; i++) {
                sum_xy += v_xy_local[i];
                sum_conic += v_conic_local[i];
                sum_colors += v_colors_local[i];
            }

            let compact_gid = compact_gid_from_isect[isect_id];

            var offset = 0u;
            if compact_gid > 0 {
                offset = cum_tiles_hit[compact_gid - 1];
            }
            let hit_id = atomicAdd(&hit_ids[compact_gid], 1u);
            let write_id = offset + hit_id;

            scatter_grads[write_id] = grads::ScatterGradient(
                sum_xy.x,
                sum_xy.y,
                sum_conic.x,
                sum_conic.y,
                sum_conic.z,
                sum_colors.x,
                sum_colors.y,
                sum_colors.z,
                sum_colors.w,
            );
        }
    }
}
