#import helpers

struct Uniforms {
    // Img resolution (w, h)
    img_size: vec2u,
    // Background color behind splats.
    background: vec3f,
}

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;

@group(0) @binding(1) var<storage, read> gaussian_ids_sorted: array<u32>;
@group(0) @binding(2) var<storage, read> tile_bins: array<vec2u>;
@group(0) @binding(3) var<storage, read> xys: array<vec2f>;
@group(0) @binding(4) var<storage, read> cov2ds: array<vec4f>;
@group(0) @binding(5) var<storage, read> colors: array<vec4f>;

#ifdef FORWARD_ONLY
    @group(0) @binding(6) var<storage, read_write> out_img: array<u32>;
#else
    @group(0) @binding(6) var<storage, read_write> out_img: array<vec4f>;
    @group(0) @binding(7) var<storage, read_write> final_index : array<u32>;
#endif

// Workgroup variables.
const GATHER_PER_ITERATION: u32 = 128u;

var<workgroup> xy_batch: array<vec2f, GATHER_PER_ITERATION>;
var<workgroup> colors_batch: array<vec4f, GATHER_PER_ITERATION>;
var<workgroup> conic_comp_batch: array<vec4f, GATHER_PER_ITERATION>;

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(16, 16, 1)
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
    let tiles_xx = (img_size.x + helpers::TILE_WIDTH - 1u) / helpers::TILE_WIDTH;

    let tile_loc = global_id.xy / helpers::TILE_WIDTH;
    let tile_id = tile_loc.x + tile_loc.y * tiles_xx;

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

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    var range = tile_bins[tile_id];

    let num_batches = helpers::ceil_div(range.y - range.x, GATHER_PER_ITERATION);
    // current visibility left to render
    var T = 1.0;

    var pix_out = vec3f(0.0);
    var final_idx = range.y;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    for (var b = 0u; b < num_batches; b++) {
        // resync all threads before beginning next batch
        // end early out if entire tile is done
        workgroupBarrier();

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        let batch_start = range.x + b * GATHER_PER_ITERATION;
        let idx = batch_start + local_idx;

        if idx < range.y && local_idx < GATHER_PER_ITERATION {
            let g_id = gaussian_ids_sorted[idx];
            xy_batch[local_idx] = xys[g_id];
            let cov2d = cov2ds[g_id].xyz;
            conic_comp_batch[local_idx] = vec4f(helpers::cov2d_to_conic(cov2d), helpers::cov_compensation(cov2d));
            colors_batch[local_idx] = colors[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        workgroupBarrier();

        // process gaussians in the current batch for this pixel
        let remaining = min(GATHER_PER_ITERATION, range.y - batch_start);

        workgroupBarrier();
    
        var t = 0u;
        for (; t < remaining && !done; t++) {
            let conic_comp = conic_comp_batch[t];
            let conic = conic_comp.xyz;
            // TODO: Re-enable compensation.
            let compensation = conic_comp.w;
            let xy = xy_batch[t];
            let opac = colors_batch[t].w;
            let delta = xy - pixel_coord;
            let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
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
            final_idx = batch_start + t;
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
