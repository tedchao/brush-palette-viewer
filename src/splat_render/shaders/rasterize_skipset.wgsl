#import helpers

struct Uniforms {
    // Img resolution (w, h)
    img_size: vec2u,
    tile_bounds: vec2u,
    // Background color behind splats.
    background: vec3f,
}

@group(0) @binding(0) var<storage> uniforms: Uniforms;
@group(0) @binding(1) var<storage> gaussian_ids_sorted: array<u32>;
@group(0) @binding(2) var<storage> xys: array<vec2f>;
@group(0) @binding(3) var<storage> conics: array<vec4f>;
@group(0) @binding(4) var<storage> colors: array<vec4f>;

@group(0) @binding(5) var<storage> tile_min_gid: array<u32>;
@group(0) @binding(6) var<storage> tile_max_gid: array<u32>;
@group(0) @binding(7) var<storage> tile_set_gid: array<u32>;

#ifdef FORWARD_ONLY
    @group(0) @binding(8) var<storage, read_write> out_img: array<u32>;
#else
    @group(0) @binding(8) var<storage, read_write> out_img: array<vec4f>;
    @group(0) @binding(9) var<storage, read_write> final_index : array<u32>;
#endif
 
// Workgroup variables.
const GAUSS_PER_BATCH: u32 = helpers::TILE_SIZE;
var<workgroup> num_gathered: atomic<u32>;
var<workgroup> visible: array<atomic<u32>, GAUSS_PER_BATCH>;

// const SHADE_PER_BATCH: u32 = 128u;
// const SHADER_PER_BATCH_OVERFLOW: u32 = SHADE_PER_BATCH + GAUSS_PER_BATCH;
var<workgroup> c_xy_batch: array<vec2f, GAUSS_PER_BATCH>;
var<workgroup> c_colors_batch: array<vec4f, GAUSS_PER_BATCH>;
var<workgroup> c_conic_comp_batch: array<vec4f, GAUSS_PER_BATCH>;

var<workgroup> num_viable_buckets: atomic<u32>;
var<workgroup> c_viable_bucket: array<u32, helpers::BUCKET_COUNT>;
var<workgroup> c_viable_bucket_set: array<u32, helpers::BUCKET_COUNT>;

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
    let background = uniforms.background;
    let img_size = uniforms.img_size;

    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    // Get index of tile being drawn.
    let tile_id = workgroup_id.x + workgroup_id.y * uniforms.tile_bounds.x;

    let tile_start = vec2f(workgroup_id.xy * helpers::TILE_WIDTH);

    let pix_id = global_id.x + global_id.y * img_size.x;
    let pixel_coord = vec2f(global_id.xy);

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    let inside = global_id.x < img_size.x && global_id.y < img_size.y;

    var done = false;
    if !inside {
        // this pixel is done
        done = true;
    }

    // current visibility left to render
    var T = 1.0;
    var pix_out = vec3f(0.0);

    #ifndef FORWARD_ONLY
        var final_idx = 0u;
    #endif
    
    let num_gaussians = arrayLength(&gaussian_ids_sorted);
    let bucket_size = (num_gaussians + helpers::BUCKET_COUNT - 1u) / helpers::BUCKET_COUNT;
    let bucket_bit_size = (bucket_size + 31u) / 32u;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel.
    let bucket_tile_offset = tile_id * helpers::BUCKET_COUNT;

    for(var b = 0u; b < helpers::BUCKET_COUNT; b += helpers::TILE_SIZE) {
        let base_offset = atomicLoad(&num_viable_buckets);
        workgroupBarrier();

        var vis_local = 0u;
        let bucket = b + local_idx;
        
        var bucket_set = 0u;
    
        if bucket < helpers::BUCKET_COUNT &&  tile_set_gid[bucket_tile_offset + bucket] > 0u {
            bucket_set = tile_set_gid[bucket_tile_offset + bucket];

            if bucket_set > 0u {
                vis_local = 1u;
                atomicAdd(&num_viable_buckets, 1u);
            }
        }
        visible[local_idx] = vis_local;
        workgroupBarrier();

        if vis_local == 1u {
            var exclusive_sum = base_offset;
            for(var i = 0u; i < local_idx; i++) {
                exclusive_sum += visible[i];
            }
            c_viable_bucket[exclusive_sum] = bucket;
            c_viable_bucket_set[exclusive_sum] = bucket_set;
        }
        workgroupBarrier();
    }

    workgroupBarrier();

    for(var b = 0u; b < num_viable_buckets; b++) {
        let bucket = c_viable_bucket[b];
        let bucket_set = c_viable_bucket_set[b];

        let bucket_index = bucket_tile_offset + bucket;
        
        let min_gid = tile_min_gid[bucket_index];
        let max_gid = tile_max_gid[bucket_index];

        for(var current_gauss_idx = min_gid; current_gauss_idx < max_gid; current_gauss_idx += GAUSS_PER_BATCH)  {
            var xy = vec2f(0.0);
            var color = vec4f(0.0);
            var conic_opac = vec4f(0.0);

            // Have some threads fetch & test some gaussians.

            let local_gauss_idx = current_gauss_idx + local_idx;
            var vis_local = 0u;
            let bit_index = (local_gauss_idx - bucket * bucket_size) / bucket_bit_size;

            if local_idx < GAUSS_PER_BATCH && local_gauss_idx < max_gid {
                if (bucket_set & (1u << bit_index)) > 0u {
                    // TODO: Test against some kind of sparse set.
                    let g_id = gaussian_ids_sorted[local_gauss_idx];
                    conic_opac = conics[g_id];
                    xy = xys[g_id];

                    if helpers::can_be_visible(tile_start, conic_opac.xyz, xy, conic_opac.w) {
                        color = vec4f(colors[g_id].xyz, conic_opac.w);
                        vis_local = 1u;
                        atomicAdd(&num_gathered, 1u);
                    }
                }

                visible[local_idx] = vis_local;
            }

            // wait for other threads to collect the gaussians in batch
            workgroupBarrier();

            if num_gathered >= 0u {
                // Compactify results.
                if vis_local == 1u {
                    var exclusiveSum = 0u;
                    for(var i = 0u; i < local_idx; i++) {
                        exclusiveSum += visible[i];
                    }
                    c_xy_batch[exclusiveSum] = xy;
                    c_colors_batch[exclusiveSum] = color;
                    c_conic_comp_batch[exclusiveSum] = conic_opac;
                }
                // Wait for the compacted results.
                workgroupBarrier();

                //  Shade pixels.
                if (!done) {
                    // Shade all the collected gaussians.
                    for (var t = 0u; t < num_gathered; t++) {
                        let xy = c_xy_batch[t].xy;
                        let c_opac = c_colors_batch[t];
                        let c = c_opac.xyz;
                        let opac = c_opac.w;
                        let conic_comp = c_conic_comp_batch[t];
                        let conic = conic_comp.xyz;
                        // TODO: Re-enable compensation.
                        let compensation = conic_comp.w;
                        let sigma = helpers::calc_sigma(conic, xy, pixel_coord);
                        let alpha = min(0.99f, opac * exp(-sigma));

                        if sigma < 0.0 || alpha < 1.0 / 255.0 {
                            continue;
                        }

                        let next_T = T * (1.0 - alpha);

                        if next_T <= 1e-4f { 
                            // this pixel is done
                            // we want to render the last gaussian that contributes and note
                            // that here idx > range.x so we don't underflow
                            done = true;
                            break;
                        }

                        let vis = alpha * T;

                        pix_out += c * vis;
                        T = next_T;
                    }
                }
            }

            // Wait for all the threads to be done shading.
            workgroupBarrier();
            // Clear num gathered flag.
            if local_idx == 0u {
                atomicStore(&num_gathered, 0u);
            }
            workgroupBarrier();
        }
    }

    if inside {
        // add background
        let final_color = pix_out + T * background;

        #ifdef FORWARD_ONLY
            let colors_u = vec4u(clamp(vec4f(final_color, 1.0 - T) * 255.0, vec4f(0.0), vec4f(255.0)));
            let packed = colors_u.x | (colors_u.y << 8u) | (colors_u.z << 16u) | (colors_u.w << 24u);
            out_img[pix_id] = packed;
        #else 
            final_index[pix_id] = final_idx; // index of in bin of last gaussian in this pixel
            out_img[pix_id] = vec4f(final_color, 1.0 - T);
        #endif
    }
}
