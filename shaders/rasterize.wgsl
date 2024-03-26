#import helpers

@group(0) @binding(0) var<storage, read> gaussian_ids_sorted: array<u32>;
@group(0) @binding(1) var<storage, read> tile_bins: array<vec2u>;
@group(0) @binding(2) var<storage, read> xys: array<vec2f>;
@group(0) @binding(3) var<storage, read> conics: array<vec3f>;
@group(0) @binding(4) var<storage, read> colors: array<vec3f>;
@group(0) @binding(5) var<storage, read> opacities: array<f32>;

@group(0) @binding(6) var<storage, read_write> final_index: array<u32>;
@group(0) @binding(7) var<storage, read_write> out_img: array<vec4f>;

@group(0) @binding(8) var<storage, read> info_array: array<helpers::InfoBinding>;

const MAX_BLOCK_SIZE: u32 = 16u * 16u;
const TILE_SIZE: u32 = 16u;
const GROUP_DIM: u32 = 16u;

// Workgroup variables.
var<workgroup> id_batch: array<u32, MAX_BLOCK_SIZE>;
var<workgroup> xy_opacity_batch: array<vec3f, MAX_BLOCK_SIZE>;
var<workgroup> conic_batch: array<vec3f, MAX_BLOCK_SIZE>;
var<workgroup> count_done: atomic<u32>;

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile

// TODO: Is this workgroup the size of block_width and co?
@compute
@workgroup_size(GROUP_DIM, GROUP_DIM, 1)
fn rasterize(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    let info = info_array[0];

    let tile_size = vec2u(TILE_SIZE, TILE_SIZE);
    let tile_bounds = info.tile_bounds;
    let background = info.background;
    let img_size = info.img_size;

    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile
    let tile_id = workgroup_id.y * tile_bounds.x + workgroup_id.x;
    let i = workgroup_id.y * GROUP_DIM + local_id.y;
    let j = workgroup_id.x * GROUP_DIM + local_id.x;

    let px = f32(j);
    let py = f32(i);
    let pix_id = i * img_size.x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    let inside = (i < img_size.y && j < img_size.x);
    var done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    let range = tile_bins[tile_id];
    let block_size = GROUP_DIM * GROUP_DIM;
    let num_batches = (range.y - range.x + block_size - 1) / block_size;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel

    // current visibility left to render
    var T = 1.0;
    // index of most recent gaussian to write to this thread's pixel
    var cur_idx = 0u;

    // TODO: Is this indeed global_idx?
    let tr = local_idx;
    var pix_out = vec3f(0.0);

    for (var b = 0u; b < num_batches; b++) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if count_done >= block_size {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        let batch_start = range.x + block_size * b;
        let idx = batch_start + tr;

        if idx < range.y {
            let g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_opacity_batch[tr] = vec3f(xys[g_id], opacities[g_id]);
            conic_batch[tr] = conics[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        workgroupBarrier();

        // process gaussians in the current batch for this pixel
        let batch_size = min(block_size, range.y - batch_start);
        
        for (var t = 0u; (t < batch_size) && !done; t++) {
            let conic = conic_batch[t];
            let xy_opac = xy_opacity_batch[t];
            let opac = xy_opac.z;
            let delta = xy_opac.xy - vec2f(px, py);
            let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            let alpha = min(0.999f, opac * exp(-sigma));
            
            if sigma < 0.0 || alpha < 1.0 / 255.0 {
                continue;
            }

            let next_T = T * (1.0 - alpha);
            
            if next_T <= 1e-4f { 
                // this pixel is done
                // we want to render the last gaussian that contributes and note
                // that here idx > range.x so we don't underflow
                atomicAdd(&count_done, 1u);
                done = true;
                break;
            }

            let g = id_batch[t];
            let vis = alpha * T;
            let c = colors[g];
            pix_out += c * vis;
            T = next_T;
            cur_idx = batch_start + t;
        }
    }

    if inside {
        // add background
        final_index[pix_id] = cur_idx; // index of in bin of last gaussian in this pixel
        out_img[pix_id] = vec4f(pix_out, T) + vec4f(T * background, 0.0);
    }
}
