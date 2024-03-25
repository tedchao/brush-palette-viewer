const GROUP_DIM_X: u32 = 16;
const GROUP_DIM_Y: u32 = 16;

const MAX_BLOCK_SIZE: u32 = GROUP_DIM_X * GROUP_DIM_Y;

var<workgroup> id_batch: array<i32, MAX_BLOCK_SIZE>;
var<workgroup> xy_opacity_batch: array<vec2f, MAX_BLOCK_SIZE>;
var<workgroup> conic_batch: array<vec4f, MAX_BLOCK_SIZE>;

var<workgroup> count_done: atomic<u32>;

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
fn rasterize_forward(
    tile_bounds: vec3u,
    img_size: vec3u,
    gaussian_ids_sorted: array<i32>,
    tile_bins: array<vec2i>,
    xys: array<vec2f>,
    conics: array<vec3f>,
    colors: array<vec3f>,
    opacities: array<f32>,
    final_Ts: array<f32>,
    final_index: array<i32>,
    out_img: array<vec3f>,
    background: vec3f,
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile
    let tile_id = workgroup_id.y * tile_bounds.x + workgroup_id.x;
    let i = workgroup_id.y * GROUP_DIM_Y + local_id.y;
    let j = workgroup_id.x * GROUP_DIM_X + local_id.x;

    let px = f32(j);
    let py = f32(i);
    let pix_id = i * img_size.x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    let inside = (i < img_size.y && j < img_size.x);
    let done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    let range = tile_bins[tile_id];
    let block_size = vec3u(GROUP_DIM_X, GROUP_DIM_Y, 1);
    let num_batches = (range.y - range.x + block_size - 1) / block_size;

    // current visibility left to render
    let T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    let cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    let tr = local_idx;
    let pix_out = vec3f(0.f, 0.f, 0.f);

    for (let b = 0; b < num_batches; b++) {
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
            let xy = xys[g_id];
            let opac = opacities[g_id];
            xy_opacity_batch[tr] = vec3f(xy.x, xy.y, opac);
            conic_batch[tr] = conics[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        workgroupBarrier();

        // process gaussians in the current batch for this pixel
        let batch_size = min(block_size, range.y - batch_start);
        
        for (let t = 0; (t < batch_size) && !done; t++) {
            let conic = conic_batch[t];
            let xy_opac = xy_opacity_batch[t];
            let opac = xy_opac.z;
            let delta = xy_opac.xy - vec2f(px, py);
            let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            let alpha = min(0.999f, opac * __expf(-sigma));
            
            if sigma < 0.f || alpha < 1.f / 255.f {
                continue;
            }

            let next_T = T * (1.f - alpha);
            
            if next_T <= 1e-4f { 
                // this pixel is done
                // we want to render the last gaussian that contributes and note
                // that here idx > range.x so we don't underflow
                atomicAdd(&count_done);
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
        final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
        final_index[pix_id] = cur_idx; // index of in bin of last gaussian in this pixel
        out_img[pix_id] = pix_out + T * background;
    }
}
