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
@group(0) @binding(3) var<storage> cov2ds: array<vec4f>;
@group(0) @binding(4) var<storage> colors: array<vec4f>;
@group(0) @binding(5) var<storage> opacities: array<f32>;

@group(0) @binding(6) var<storage> tile_min_gid: array<atomic<u32>>;
@group(0) @binding(7) var<storage> tile_max_gid: array<atomic<u32>>;
// @group(0) @binding(8) var<storage> skipbits: array<atomic<u32>>;

#ifdef FORWARD_ONLY
    @group(0) @binding(8) var<storage, read_write> out_img: array<u32>;
#else
    @group(0) @binding(8) var<storage, read_write> out_img: array<vec4f>;
    @group(0) @binding(9) var<storage, read_write> final_index : array<u32>;
#endif
 
// Workgroup variables.
const GAUSS_PER_BATCH: u32 = 32u;
var<workgroup> visible: array<u32, GAUSS_PER_BATCH>;
var<workgroup> xy_batch: array<vec2f, GAUSS_PER_BATCH>;
var<workgroup> colors_batch: array<vec4f, GAUSS_PER_BATCH>;
var<workgroup> conic_comp_batch: array<vec4f, GAUSS_PER_BATCH>;

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

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    for(var bucket = 0u; bucket < helpers::BUCKET_COUNT; bucket++) {
        let min_gid = tile_min_gid[tile_id * helpers::BUCKET_COUNT + bucket];
        let max_gid = tile_max_gid[tile_id * helpers::BUCKET_COUNT + bucket];

        for(var current_gauss_idx = min_gid; current_gauss_idx < max_gid; current_gauss_idx += GAUSS_PER_BATCH)  {
            // each thread fetch 1 gaussian from front to back
            // index of gaussian to load
            let local_gauss_idx = current_gauss_idx + local_idx;

            // Fetch and shade one batch.
            workgroupBarrier();

            // Mark invisible.
            visible[local_idx] = 0u;

            if local_idx < GAUSS_PER_BATCH && local_gauss_idx < max_gid {
                // TODO: Test against some kind of sparse set.
                let g_id = gaussian_ids_sorted[local_gauss_idx];

                let cov2d = cov2ds[g_id].xyz;
                let conic = helpers::cov2d_to_conic(cov2d);
                let xy = xys[g_id];
                let opac = opacities[g_id];

                if helpers::can_be_visible(tile_start, conic, xy, opac) {
                    colors_batch[local_idx] = vec4f(colors[g_id].xyz, opac);
                    xy_batch[local_idx] = xy;
                    conic_comp_batch[local_idx] = vec4f(conic, helpers::cov_compensation(cov2d));
                    visible[local_idx] = 1u;
                }
            }
            // wait for other threads to collect the gaussians in batch
            workgroupBarrier();

            if (!done) {
                // Shade all the collected gaussians.
                for (var t = 0u; t < GAUSS_PER_BATCH; t++) {
                    if visible[t] == 0u {
                        continue;
                    }

                    let opac = colors_batch[t].w;
                    let conic_comp = conic_comp_batch[t];
                    let conic = conic_comp.xyz;
                    // TODO: Re-enable compensation.
                    let compensation = conic_comp.w;
                    let xy = xy_batch[t];
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

                    let c = colors_batch[t].xyz;
                    pix_out += c * vis;
                    T = next_T;

                    #ifndef FORWARD_ONLY
                        final_idx = current_gauss_idx + t;
                    #endif
                }
            }
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
