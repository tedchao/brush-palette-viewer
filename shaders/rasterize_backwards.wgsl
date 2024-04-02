#import helpers;

@group(0) @binding(0) var<storage, read> gaussian_ids_sorted: array<u32>;
@group(0) @binding(1) var<storage, read> tile_bins: array<vec2u>;
@group(0) @binding(2) var<storage, read> xys: array<vec2f>;
@group(0) @binding(3) var<storage, read> conics: array<vec4f>;
@group(0) @binding(4) var<storage, read> colors: array<vec4f>;
@group(0) @binding(5) var<storage, read> opacities: array<f32>;
@group(0) @binding(6) var<storage, read> final_index: array<u32>;
@group(0) @binding(7) var<storage, read> output: array<vec4f>;
@group(0) @binding(8) var<storage, read> v_output: array<vec4f>;

// TODO: These all need to be atomic :/
@group(0) @binding(9) var<storage, read_write> v_opacity: array<f32>;
@group(0) @binding(10) var<storage, read_write> v_conic: array<vec4f>;
@group(0) @binding(11) var<storage, read_write> v_xy: array<vec2f>;
@group(0) @binding(12) var<storage, read_write> v_rgb: array<vec4f>;

@group(0) @binding(13) var<storage, read_write> locks: array<atomic<u32>>;

@group(0) @binding(14) var<storage, read> info_array: array<Uniforms>;



const BLOCK_WIDTH: u32 = 16u;
const BLOCK_SIZE: u32 = BLOCK_WIDTH * BLOCK_WIDTH;

var<workgroup> id_batch: array<u32, BLOCK_SIZE>;
var<workgroup> xy_batch: array<vec2f, BLOCK_SIZE>;
var<workgroup> opacity_batch: array<f32, BLOCK_SIZE>;
var<workgroup> conic_batch: array<vec4f, BLOCK_SIZE>;
var<workgroup> rgbs_batch: array<vec3f, BLOCK_SIZE>;

struct Uniforms {
    // Img resolution (w, h)
    img_size: vec2u,

    // Total reachable pixels (w, h)
    tile_bounds: vec2u,

    // Background color behind the splats.
    background: vec3f,
}

@compute
@workgroup_size(BLOCK_WIDTH, BLOCK_WIDTH, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    let info = info_array[0];
    let tile_bounds = info.tile_bounds;
    let background = info.background;
    let img_size = info.img_size;

    // TODO: Make a function to share this with the forward pass.
    let tile_id = workgroup_id.x + workgroup_id.y * tile_bounds.x;
    let px = f32(global_id.x) + 0.5;
    let py = f32(global_id.y) + 0.5;
    let pix_id = global_id.x + global_id.y * img_size.x;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    let inside = global_id.x < img_size.x && global_id.y < img_size.y;

    // this is the T AFTER the last gaussian in this pixel
    let T_final = output[pix_id].w;
    var T = T_final;
    // the contribution from gaussians behind the current one
    var buffer = vec3f(0.0, 0.0, 0.0);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between bin_start and bin_final in batches
    // which gaussians to look through in this tile
    let range = tile_bins[tile_id];

    let bin_start = range.x;
    var bin_final = range.y;
    if inside {
        bin_final = final_index[pix_id];
    }

    // df/d_out for this pixel
    let v_out = v_output[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    var num_batches = (bin_final - bin_start + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // num_batches = min(3u, num_batches);

    for(var batch = 0u; batch < num_batches; batch++) {
        let gauss_idx_start = bin_final - 1 - batch * BLOCK_SIZE;
        // resync all threads before writing next batch of shared mem
        // workgroupBarrier();
        
        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        let gauss_idx = gauss_idx_start - local_idx;

        if gauss_idx >= range.x {
            let g_id = gaussian_ids_sorted[gauss_idx];
            
            id_batch[local_idx] = g_id;
            xy_batch[local_idx] = xys[g_id];
            conic_batch[local_idx] = conics[g_id];
            opacity_batch[local_idx] = opacities[g_id];
            rgbs_batch[local_idx] = colors[g_id].xyz;
        }
    
        // wait for other threads to collect the gaussians in batch
        workgroupBarrier();

        // TODO: WGSL lacks warp intrinsics, quite a bit more contention here
        // than needed. Might be some other ways to reduce it?
        let remaining = min(BLOCK_SIZE, gauss_idx_start + 1 - range.x);

        if inside {
            for (var t = 0u; t < remaining; t++) {
                let conic = conic_batch[t];
                let delta =  xy_batch[t] - vec2f(px, py);

                // TODO: Make this a function to share with the forward pass.
                let sigma = 0.5f * (conic.x * delta.x * delta.x +
                                            conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;

                let opac = opacity_batch[t];
                let vis = exp(-sigma);
                let alpha = min(0.99f, opac * vis);

                if (sigma < 0.0f || alpha < 1.0f / 255.f) {
                    continue;
                }

                // compute the current T for this gaussian
                let ra = 1.0f / (1.0f - alpha);
                T *= ra;

                // update v_rgb for this gaussian
                let fac = alpha * T;
                let v_rgb_local = fac * v_out;

                var v_alpha = 0.0;

                let rgb = rgbs_batch[t];
                // contribution from this pixel
                v_alpha += (rgb.x * T - buffer.x * ra) * v_out.x;
                v_alpha += (rgb.y * T - buffer.y * ra) * v_out.y;
                v_alpha += (rgb.z * T - buffer.z * ra) * v_out.z;
                // contribution from background pixel
                v_alpha += -T_final * ra * background.x * v_out.x;
                v_alpha += -T_final * ra * background.y * v_out.y;
                v_alpha += -T_final * ra * background.z * v_out.z;

                // update the running sum
                buffer += rgb * fac;

                // TODO: With WGSL lacking atomic floats (urgh), maybe we could
                // take a lock here for one gaussian, do all the writes, release the lock.
                let v_sigma = -opac * vis * v_alpha;

                let v_conic_local = vec3f(0.5f * v_sigma * delta.x * delta.x, 
                                    v_sigma * delta.x * delta.y,
                                    0.5f * v_sigma * delta.y * delta.y);

                let v_xy_local = v_sigma * vec2f(
                    conic.x * delta.x + conic.y * delta.y, 
                    conic.y * delta.x + conic.z * delta.y
                );
                let v_opacity_local = vis * v_alpha;

                // This is NOT a correct spin lock, but a lossy one.
                // Between loading the lock and locking it, someone else might lock!!
                // However, a 'correct' spin lock seems to crash quite a lot... gpus
                // really don't like to busy wait :/
                // let lock = &locks[g];
                // while atomicLoad(lock) == 1u {
                // }
                let g_id = id_batch[t];

                //while !atomicCompareExchangeWeak(&locks[g_id], 0u, 1u).exchanged {
                //}
                // atomicCompareExchangeWeak(lock, 0u, 1u);

                v_opacity[g_id] += v_opacity_local;
                v_rgb[g_id] += v_rgb_local;
                v_conic[g_id] += vec4f(v_conic_local, 0.0f);
                v_xy[g_id] += v_xy_local;

                atomicStore(&locks[g_id], 0u);
            }

            // Sum up gradients for all pixels (which remember share the same gaussian list).

            if local_idx == 0 {
                // "atomic" add to gradients.
                // TODO: Atomic float emulation seems hard :/
                let g_id = id_batch[0];
            }
        }
    }
}