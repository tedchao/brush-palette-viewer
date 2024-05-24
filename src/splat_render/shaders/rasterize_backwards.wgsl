#import helpers;

struct Uniforms {
    // Img resolution (w, h)
    img_size: vec2u,
    // Background color behind the splats.
    background: vec3f,
}

@group(0) @binding(0) var<storage> uniforms: Uniforms;

@group(0) @binding(1) var<storage, read> depthsort_gid_from_isect: array<u32>;
@group(0) @binding(2) var<storage> compact_from_depthsort_gid: array<u32>;
@group(0) @binding(3) var<storage> tile_bins: array<vec2u>;
@group(0) @binding(4) var<storage> xys: array<vec2f>;
@group(0) @binding(5) var<storage> cum_tiles_hit: array<u32>;
@group(0) @binding(6) var<storage> conic_comps: array<vec4f>;
@group(0) @binding(7) var<storage> colors: array<vec4f>;

@group(0) @binding(8) var<storage> final_index: array<u32>;

@group(0) @binding(9) var<storage> output: array<vec4f>;
@group(0) @binding(10) var<storage> v_output: array<vec4f>;

@group(0) @binding(11) var<storage, read_write> v_xy: array<vec2f>;
@group(0) @binding(12) var<storage, read_write> v_conic: array<vec4f>;
@group(0) @binding(13) var<storage, read_write> v_colors: array<vec4f>;

@group(0) @binding(14) var<storage, read_write> hit_ids: array<atomic<u32>>;

var<workgroup> xy_batch: array<vec2f, helpers::TILE_SIZE>;
var<workgroup> colors_batch: array<vec4f, helpers::TILE_SIZE>;
var<workgroup> conic_comp_batch: array<vec4f, helpers::TILE_SIZE>;

var<workgroup> v_conic_local: array<vec3f, helpers::TILE_SIZE>;
var<workgroup> v_xy_local: array<vec2f, helpers::TILE_SIZE>;
var<workgroup> v_colors_local: array<vec4f, helpers::TILE_SIZE>;

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(helpers::TILE_WIDTH, helpers::TILE_WIDTH, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    let background = uniforms.background;
    let img_size = uniforms.img_size;

    let tile_bounds = vec2u(helpers::ceil_div(img_size.x, helpers::TILE_WIDTH),  
                            helpers::ceil_div(img_size.y, helpers::TILE_WIDTH));

    let tile_loc = workgroup_id.xy;
    let tile_id = tile_loc.x + tile_loc.y * tile_bounds.x;
    let pix_id = global_id.x + global_id.y * img_size.x;
    let pixel_coord = vec2f(global_id.xy) + 0.5;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    let inside = global_id.x < img_size.x && global_id.y < img_size.y;

    // this is the T AFTER the last gaussian in this pixel
    // TODO: Is this 1-x ? x?
    let T_final = 1.0 - output[pix_id].w;
    
    // the contribution from gaussians behind the current one

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between bin_start and bin_final in batches
    // which gaussians to look through in this tile
    var range = tile_bins[tile_id];
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

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    for (var b = 0u; b < num_batches; b++) {
        // resync all threads before beginning next batch
        workgroupBarrier();
        
        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        let batch_end  = range.y - 1u - b * helpers::TILE_SIZE;
        let isect_id = batch_end - local_idx;

        if isect_id >= range.x {
            let depthsort_gid = depthsort_gid_from_isect[isect_id];
            let compact_gid = compact_from_depthsort_gid[depthsort_gid];

            xy_batch[local_idx] = xys[compact_gid];
            conic_comp_batch[local_idx] = conic_comps[compact_gid];
            colors_batch[local_idx] = colors[compact_gid];
        }

        // wait for other threads to collect the gaussians in batch
        workgroupBarrier();

        // process gaussians in the current batch for this pixel
        let remaining = min(helpers::TILE_SIZE, batch_end + 1u - range.x);

        workgroupBarrier();

        for (var t = 0u; t < remaining; t++) {
            workgroupBarrier();

            v_xy_local[local_idx] = vec2f(0.0);
            v_conic_local[local_idx] = vec3f(0.0);
            v_colors_local[local_idx] = vec4f(0.0);

            let isect_id = batch_end - t;

            if inside && isect_id >= range.x && isect_id <= bin_final {
                let conic_comp = conic_comp_batch[t];
                let conic = conic_comp.xyz;
                // TODO: Re-enable compensation.
                // let compensation = conic_comp.w;
                let xy = xy_batch[t];
                let opac = colors_batch[t].w;
                let delta = xy - pixel_coord;
                let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                let vis = exp(-sigma);
                let alpha = min(0.99f, opac * vis);

                // Nb: Don't continue; here - local_idx == 0 always
                // needs to write out gradients.
                // compute the current T for this gaussian
                if (sigma >= 0.0 && alpha >= 1.0 / 255.0) {
                    let ra = 1.0 / (1.0 - alpha);
                    T *= ra;
                    // update v_colors for this gaussian
                    let fac = alpha * T;
                    var v_alpha = 0.0;

                    let rgb = colors_batch[t].xyz;
                    // contribution from this pixel
                    v_alpha += dot(rgb * T - buffer * ra, v_out.xyz);
                    v_alpha += T_final * ra * v_out.w;
                    // contribution from background pixel
                    v_alpha -= dot(T_final * ra * background, v_out.xyz);

                    // update the running sum
                    buffer += rgb.xyz * fac;

                    let v_sigma = -opac * vis * v_alpha;

                    v_conic_local[local_idx] = vec3f(0.5f * v_sigma * delta.x * delta.x, 
                                                    v_sigma * delta.x * delta.y,
                                                    0.5f * v_sigma * delta.y * delta.y);
                    
                    v_xy_local[local_idx] = v_sigma * vec2f(
                        conic.x * delta.x + conic.y * delta.y, 
                        conic.y * delta.x + conic.z * delta.y
                    );
                    v_colors_local[local_idx] = vec4f(fac * v_out.xyz, vis * v_alpha);
                }
            }

            // Make sure all threads have calculated their gradients.
            workgroupBarrier();

            if local_idx == 0 {
                let depthsort_gid = depthsort_gid_from_isect[isect_id];
                let compact_gid = compact_from_depthsort_gid[depthsort_gid];

                // Gather workgroup sums.
                var v_colors_sum = vec4f(0.0);
                var v_conic_sum = vec3f(0.0);
                var v_xy_sum = vec2f(0.0);
                
                for(var i = 0u; i < helpers::TILE_SIZE; i++) {
                    v_colors_sum += v_colors_local[i];
                    v_conic_sum += v_conic_local[i];
                    v_xy_sum += v_xy_local[i];
                }

                var offset = 0u;
                if depthsort_gid > 0 {
                    offset = cum_tiles_hit[depthsort_gid - 1];
                }
                let hit_id = atomicAdd(&hit_ids[compact_gid], 1u);

                // Scatter the gradients to the gradient per intersection buffer.
                let write_id = offset + hit_id;
                v_xy[write_id] = v_xy_sum;
                v_conic[write_id] = vec4f(v_conic_sum, 0.0);
                v_colors[write_id] = v_colors_sum;
            }

            workgroupBarrier();
        }
    }
}
