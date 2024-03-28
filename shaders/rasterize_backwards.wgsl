#import helpers;

@group(0) @binding(0) var<storage, read> gaussian_ids_sorted: array<u32>;
@group(0) @binding(1) var<storage, read> tile_bins: array<vec2u>;
@group(0) @binding(2) var<storage, read> xys: array<vec2f>;
@group(0) @binding(3) var<storage, read> conics: array<vec4f>;
@group(0) @binding(4) var<storage, read> colors: array<f32>;
@group(0) @binding(5) var<storage, read> opacities: array<f32>;
@group(0) @binding(6) var<storage, read> final_index: array<u32>;
@group(0) @binding(7) var<storage, read> output: array<vec4f>;
@group(0) @binding(8) var<storage, read> v_output: array<vec4f>;

// TODO: These all need to be atomic :/
@group(0) @binding(9) var<storage, read_write> v_opacity: array<f32>;
@group(0) @binding(10) var<storage, read_write> v_conic: array<vec4f>;
@group(0) @binding(11) var<storage, read_write> v_xy: array<vec2f>;
@group(0) @binding(12) var<storage, read_write> v_rgb: array<vec4f>;

@group(0) @binding(13) var<storage, read> info_array: array<Uniforms>;

const BLOCK_SIZE: u32 = 16 * 16;

var<workgroup> id_batch: array<u32, BLOCK_SIZE>;
var<workgroup> xy_batch: array<vec2f, BLOCK_SIZE>;
var<workgroup> opacity_batch: array<f32, BLOCK_SIZE>;
var<workgroup> conic_batch: array<vec4f, BLOCK_SIZE>;
var<workgroup> rgbs_batch: array<vec4f, BLOCK_SIZE>;

struct Uniforms {
    // Img resolution (w, h)
    img_size: vec2u,

    // Total reachable pixels (w, h)
    tile_bounds: vec2u,

    // Background color behind the splats.
    background: vec3f,
}

@compute
@workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(local_invocation_index) tr: u32,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    let info = info_array[0];
    let tile_bounds = info.tile_bounds;
    let background = info.background;
    let img_size = info.img_size;

    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile.

    // TODO: This is shared with forward?
    let tile_id = workgroup_id.x + workgroup_id.y * tile_bounds.x;
    let j = global_id.x;
    let i = global_id.y;

    let px = f32(j);
    let py = f32(i);
    let pix_id = global_id.x + global_id.y * img_size.x;

    // keep not rasterizing threads around for reading data
    let inside = i < img_size.y && j < img_size.x;

    var done = false;

    if !inside {
        done = true;
    }

    // this is the T AFTER the last gaussian in this pixel
    let T_final = output[pix_id].w;
    var T = T_final;
    // the contribution from gaussians behind the current one
    var buffer = vec3f(0.0, 0.0, 0.0);

    // index of last gaussian to contribute to this pixel
    var bin_final = 0u;
    if inside {
        bin_final = final_index[pix_id];
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    let range = tile_bins[tile_id];
    let num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // df/d_out for this pixel
    let v_out = v_output[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    for (var b = 0u; b < num_batches; b++) {
        // resync all threads before writing next batch of shared mem
        workgroupBarrier();
        
        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        let batch_end = range.y - 1 - BLOCK_SIZE * b;
        let idx = batch_end - tr;

        if (idx >= range.x) {
            let g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = xys[g_id];
            conic_batch[tr] = conics[g_id];
            opacity_batch[tr] = opacities[g_id];

            let c = vec3f(colors[g_id * 3 + 0], colors[g_id  * 3 + 1], colors[g_id  * 3 + 2]);
            rgbs_batch[tr] = vec4f(c, 0.0);
        }
    
        // wait for other threads to collect the gaussians in batch
        workgroupBarrier();

        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        let batch_size = min(BLOCK_SIZE, batch_end + 1 - range.x);

        // TODO: WGSL lacks warp intrinsics, quite a bit more contention here
        // than needed. Might be some other ways to reduce it?
        for (var t = 0u; (t < batch_size) && inside; t++) {
            if (batch_end - t > bin_final) {
                continue;
            }

            let conic = conic_batch[t];
            let center = xy_batch[t];
            let delta = center - vec2f(px, py);
            // TODO: Make this a function.
            let sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;

            let opac = opacity_batch[t];
            let vis = exp(-sigma);
            let alpha = min(0.99f, opac * vis);
            if (sigma < 0.0f || alpha < 1.0f / 255.f) {
                continue;
            }

            let g = id_batch[t];
            // compute the current T for this gaussian
            let ra = 1.0f / (1.f - alpha);
            T *= ra;

            // update v_rgb for this gaussian
            let fac = alpha * T;
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

            // gradient wrt rgb

            // update the running sum
            buffer.x += rgb.x * fac;
            buffer.y += rgb.y * fac;
            buffer.z += rgb.z * fac;

            // TODO: With WGSL lacking atomic floats (urgh), maybe we could
            // take a lock here for one gaussian, do all the writes, release the lock.

            let v_sigma = -opac * vis * v_alpha;

            v_opacity[g] += vis * v_alpha;
            v_rgb[g] += fac * v_out;
            v_conic[g] += vec4f(0.5f * v_sigma * vec3f(delta.x * delta.x, delta.x * delta.y, delta.y * delta.y), 0.0);
            v_xy[g] += v_sigma * (conic.x * delta.x + conic.y * delta.y);
        }
    }
}