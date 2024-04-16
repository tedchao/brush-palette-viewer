#import helpers;

struct Uniforms {
    // Img resolution (w, h)
    img_size: vec2u,
    // Background color behind the splats.
    background: vec3f,
}

@group(0) @binding(0) var<storage> uniforms: Uniforms;

@group(0) @binding(1) var<storage> gaussian_ids_sorted: array<u32>;
@group(0) @binding(2) var<storage> tile_bins: array<vec2u>;
@group(0) @binding(3) var<storage> xys: array<vec2f>;
@group(0) @binding(4) var<storage> cov2ds: array<vec4f>;
@group(0) @binding(5) var<storage> colors: array<vec4f>;
@group(0) @binding(6) var<storage> opacities: array<f32>;
@group(0) @binding(7) var<storage> final_index: array<u32>;
@group(0) @binding(8) var<storage> output: array<vec4f>;
@group(0) @binding(9) var<storage> v_output: array<vec4f>;

@group(0) @binding(10) var<storage, read_write> v_xy: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> v_conic: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> v_colors: array<atomic<u32>>;
@group(0) @binding(13) var<storage, read_write> v_opacity: array<atomic<u32>>;

var<workgroup> id_batch: array<u32, helpers::TILE_SIZE>;
var<workgroup> xy_batch: array<vec2f, helpers::TILE_SIZE>;
var<workgroup> colors_batch: array<vec4f, helpers::TILE_SIZE>;
var<workgroup> conic_comp_batch: array<vec4f, helpers::TILE_SIZE>;

var<workgroup> v_opacity_local: array<f32, helpers::TILE_SIZE>;
var<workgroup> v_conic_local: array<vec3f, helpers::TILE_SIZE>;
var<workgroup> v_xy_local: array<vec2f, helpers::TILE_SIZE>;
var<workgroup> v_colors_local: array<vec3f, helpers::TILE_SIZE>;

fn bitAddFloat(cur: u32, add: f32) -> u32 {
    return bitcast<u32>(bitcast<f32>(cur) + add);
}

@compute
@workgroup_size(helpers::TILE_WIDTH, helpers::TILE_WIDTH, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    let background = uniforms.background;
    let img_size = uniforms.img_size;

    let tiles_xx = (img_size.x + helpers::TILE_WIDTH - 1) / helpers::TILE_WIDTH;
    let tile_id = workgroup_id.x + workgroup_id.y * tiles_xx;

    let pix_id = global_id.x + global_id.y * img_size.x;
    let pixel_coord = vec2f(global_id.xy);

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    let inside = global_id.x < img_size.x && global_id.y < img_size.y;

    // this is the T AFTER the last gaussian in this pixel
    let T_final = 1.0 - output[pix_id].w;
    
    var T = T_final;
    // the contribution from gaussians behind the current one
    var buffer = vec3f(0.0, 0.0, 0.0);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between bin_start and bin_final in batches
    // which gaussians to look through in this tile
    var range = tile_bins[tile_id];
    var bin_final = 0u;
    if inside {
        bin_final = final_index[pix_id];
    }

    // df/d_out for this pixel
    let v_out = v_output[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    var num_batches = (range.y - range.x + helpers::TILE_SIZE - 1) / helpers::TILE_SIZE;

    for(var batch = 0u; batch < num_batches; batch++) {
        let batch_end  = range.y - 1 - batch * helpers::TILE_SIZE;

        // resync all threads before writing next batch of shared mem
        workgroupBarrier();
        
        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        let fetch_idx = batch_end - local_idx;

        if fetch_idx >= range.x {
            let g_id = gaussian_ids_sorted[fetch_idx];
            id_batch[local_idx] = g_id;
            xy_batch[local_idx] = xys[g_id];

            colors_batch[local_idx] = vec4f(colors[g_id].xyz, opacities[g_id]);
            let cov2d = cov2ds[g_id].xyz;
            conic_comp_batch[local_idx] = vec4f(helpers::cov2d_to_conic(cov2d), helpers::cov_compensation(cov2d));
        }
    
        // wait for other threads to collect the gaussians in batch
        workgroupBarrier();

        // TODO: WGSL lacks warp intrinsics, quite a bit more contention here
        // than needed. Might be some other ways to reduce it?
        let batch_size = min(helpers::TILE_SIZE, batch_end + 1 - range.x);

        // reset local accumulations.
        for (var t = 0u; t < batch_size; t++) {
            
            // Don't overwrite data before all the gradients have been scattered to the gaussians.
            workgroupBarrier();

            v_opacity_local[local_idx] = 0.0;
            v_xy_local[local_idx] = vec2f(0.0);
            v_conic_local[local_idx] = vec3f(0.0);
            v_colors_local[local_idx] = vec3f(0.0);

            let batch_idx = batch_end - t;

            if inside && batch_idx >= range.x && batch_idx <= bin_final {
                let conic_comp = conic_comp_batch[t];
                let conic = conic_comp.xyz;
                let color = colors_batch[t];
                let xy = xy_batch[t];
                // TODO: Re-enable compensation.
                let compensation = conic_comp.w;
                let opac = color.w;
                let delta = xy - pixel_coord;
                var sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                let vis = exp(-sigma);

                let alpha = min(0.99f, opac * vis);

                if (sigma > 0.0 && alpha > 1.0 / 255.0) {
                    // compute the current T for this gaussian
                    let ra = 1.0 / (1.0 - alpha);
                    T *= ra;

                    // update v_colors for this gaussian
                    let fac = alpha * T;
                    v_colors_local[local_idx] = fac * v_out.xyz;

                    var v_alpha = 0.0;

                    let color = color.xyz;
                    // contribution from this pixel
                    v_alpha += dot(color * T - buffer * ra, v_out.xyz);
                    // TODO: Now that we store alpha instea of transmission, flip this?
                    // v_alpha += T_final * ra * v_out.w;
                    // contribution from background pixel
                    v_alpha -= dot(T_final * ra * background, v_out.xyz);

                    // update the running sum
                    buffer += color * fac;

                    let v_sigma = -opac * vis * v_alpha;

                    v_conic_local[local_idx] = vec3f(0.5f * v_sigma * delta.x * delta.x, 
                                                    v_sigma * delta.x * delta.y,
                                                    0.5f * v_sigma * delta.y * delta.y);
                    
                    v_xy_local[local_idx] = v_sigma * vec2f(
                        conic.x * delta.x + conic.y * delta.y, 
                        conic.y * delta.x + conic.z * delta.y
                    );

                    v_opacity_local[local_idx] = vis * v_alpha;
                }
            }

            // Make sure all threads have calculated their gradient.
            // This needs to be on a uniform path so can't be in the if above.
            workgroupBarrier();

            if local_idx == 0 {
                let g_id = id_batch[t];

                // Gather workgroup sums.
                var v_colors_sum = vec3f(0.0);
                var v_conic_sum = vec3f(0.0);
                var v_xy_sum = vec2f(0.0);
                var v_opacity_sum = 0.0;
                
                for(var i = 0u; i < helpers::TILE_SIZE; i++) {
                    v_colors_sum += v_colors_local[i];
                    v_conic_sum += v_conic_local[i];
                    v_xy_sum += v_xy_local[i];
                    v_opacity_sum += v_opacity_local[i];
                }

                // Write out results atomically. This is... truly awful :)
                // The basic problem is that WGSL doesn't have atomic floats. We need
                // to scatter gradients to the various gaussians. Some options were tried and 
                // didn't work out:
                // * Implement some kind of locking mechanism with atomic u32. This just seems to fundamentally
                //   not work, perhaps correctly if I understand the WGSL spec. While the lock can be 'consistent'
                //   between workgroups, the lock release can still be put before the write, as there are no memory
                //   guarantees between these.
                // * Transpose the problem, launch a thread per gaussian, and render all tiles for said gaussian in one
                //   thread. While I suspect this could work - it could have horrendous performance cliffs. Each gaussian 
                //   would need to load v_out (maybe once per workgroup to be fair), which could be the entire screen. It would also
                //   possibly involve a compaction pass.
                // In the end, "software emuation" of atomic floats seems easiest. It's somewhat TBD whether the performance of this is viable.
                // Code wise, it would be less awful if more of this could be functions, but as far as I can see I can't make a WGSL function
                // that takes a ptr<storage, atomic<u32>> sucesfully. I think calling with functions with pointers with storage
                // adress spaces is an extension or not yet properly supported anyway.
                // There is a lot of oppurtunity for optimization here still.

                // v_colors.
                loop {
                    let old = v_colors[g_id * 4 + 0];
                    if atomicCompareExchangeWeak(&v_colors[g_id * 4 + 0], old, bitAddFloat(old, v_colors_sum.x)).exchanged {
                        break;
                    }
                }

                loop {
                    let old = v_colors[g_id * 4 + 1];
                    if atomicCompareExchangeWeak(&v_colors[g_id * 4 + 1], old, bitAddFloat(old, v_colors_sum.y)).exchanged {
                        break;
                    }
                }

                loop {
                    let old = v_colors[g_id * 4 + 2];
                    if atomicCompareExchangeWeak(&v_colors[g_id * 4 + 2], old, bitAddFloat(old, v_colors_sum.z)).exchanged {
                        break;
                    }
                }

                // v_conic.
                loop {
                    let old = v_conic[g_id * 4 + 0];
                    if atomicCompareExchangeWeak(&v_conic[g_id * 4 + 0], old, bitAddFloat(old, v_conic_sum.x)).exchanged {
                        break;
                    }
                }

                loop {
                    let old = v_conic[g_id * 4 + 1];
                    if atomicCompareExchangeWeak(&v_conic[g_id * 4 + 1], old, bitAddFloat(old, v_conic_sum.y)).exchanged {
                        break;
                    }
                }

                loop {
                    let old = v_conic[g_id * 4 + 2];
                    if atomicCompareExchangeWeak(&v_conic[g_id * 4 + 2], old, bitAddFloat(old, v_conic_sum.z)).exchanged {
                        break;
                    }
                }

                // v_xy.
                loop {
                    let old = v_xy[g_id * 2 + 0];
                    if atomicCompareExchangeWeak(&v_xy[g_id * 2 + 0], old, bitAddFloat(old, v_xy_sum.x)).exchanged {
                        break;
                    }
                }

                loop {
                    let old = v_xy[g_id * 2 + 1];
                    if atomicCompareExchangeWeak(&v_xy[g_id * 2 + 1], old, bitAddFloat(old, v_xy_sum.y)).exchanged {
                        break;
                    }
                }

                // v_opacity
                loop {
                    let old = v_opacity[g_id];
                    if atomicCompareExchangeWeak(&v_opacity[g_id], old, bitAddFloat(old, v_opacity_sum)).exchanged {
                        break;
                    }
                }
            }
        }
    }
}
