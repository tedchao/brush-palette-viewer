#import helpers;


@group(0) @binding(0) var<storage, read> means3d: array<vec3f>;
@group(0) @binding(1) var<storage, read> scales: array<vec3f>;
@group(0) @binding(2) var<storage, read> quats: array<vec4f>;


@group(0) @binding(3) var<storage, read_write> covs3d: array<f32>;
@group(0) @binding(4) var<storage, read_write> xys: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> depths: array<f32>;
@group(0) @binding(6) var<storage, read_write> radii: array<i32>;
@group(0) @binding(7) var<storage, read_write> conics: array<vec3f>;
@group(0) @binding(8) var<storage, read_write> compensation: array<f32>;
@group(0) @binding(9) var<storage, read_write> num_tiles_hit: array<i32>;

@group(0) @binding(10) var<storage, read> info_array: array<InfoBinding>;

struct InfoBinding {
    // View matrix transform world to view position.
    viewmat: mat4x4f,
    
    projmat: mat4x4f,

    intrins: vec4f,

    img_size: vec2u,
    tile_bounds: vec2u,

    glob_scale: f32,
    num_points: u32,
    clip_thresh: f32,
    block_width: u32,
}

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
@compute
@workgroup_size(16, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    // Until burn supports adding in uniforms we read these from a tensor.
    let info = info_array[0];

    let idx = local_id.x;
    let num_points = info.num_points;

    if idx >= num_points {
        return;
    }

    let glob_scale = info.glob_scale;
    let viewmat = info.viewmat;
    let projmat = info.projmat;
    let intrins = info.intrins;

    let img_size = info.img_size;
    let tile_bounds = info.tile_bounds;
    let block_width = info.block_width;
    let clip_thresh = info.clip_thresh;

    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    let p_world = means3d[idx];

    // Project local space to
    let p_view = viewmat * vec4f(p_world, 1.0f);

    if p_view.z <= clip_thresh {
        return;
    }

    // compute the projected covariance
    let scale = scales[idx];
    let quat = quats[idx];

    let tmp = helpers::scale_rot_to_cov3d(scale, glob_scale, quat);

    // save upper right of matrix, as it's symmetric.
    // TODO: Does this match the original order? row vs column major?
    let covs0 = tmp[0][0];
    let covs1 = tmp[0][1];
    let covs2 = tmp[0][2];
    let covs3 = tmp[1][1];
    let covs4 = tmp[1][2];
    let covs5 = tmp[2][2];
    
    covs3d[6 * idx + 0] = covs0;
    covs3d[6 * idx + 1] = covs1;
    covs3d[6 * idx + 2] = covs2;
    covs3d[6 * idx + 3] = covs3;
    covs3d[6 * idx + 4] = covs4;
    covs3d[6 * idx + 5] = covs5;

    let fx = intrins.x;
    let fy = intrins.y;
    let cx = intrins.z;
    let cy = intrins.w;
    let tan_fovx = 0.5 * f32(img_size.x) / fx;
    let tan_fovy = 0.5 * f32(img_size.y) / fy;

    let W = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);

    // clip so that the covariance
    // TODO: What does that comment mean... :)
    let lims = 1.3f * vec2f(tan_fovx, tan_fovy);

    let t = p_world.z * clamp(p_world.xy / p_world.z, -lims, lims);
    let rz = 1.0f / p_world.z;
    let rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    let J = mat3x3f(
        vec3f(fx * rz, 0.0f, 0.0f,),
        vec3f(0.0f, fy * rz, 0.0f),
        vec3f(-fx * t.x * rz2, -fy * t.y * rz2, 0.0f)
    );

    let T = J * W;

    let V = mat3x3f(
        vec3f(covs0, covs1, covs2),
        vec3f(covs1, covs3, covs4),
        vec3f(covs2, covs4, covs5)
    );

    let cov = T * V * transpose(T);

    // add a little blur along axes and save upper triangular elements
    // and compute the density compensation factor due to the blurs
    let c00 = cov[0][0];
    let c11 = cov[1][1];
    let c01 = cov[0][1];
    let det_orig = c00 * c11 - c01 * c01;
    let cov2d = vec3f(c00 + 0.3f, c01, c11 + 0.3f);
    
    let det_blur = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    let comp = sqrt(max(0.f, det_orig / det_blur));

    let cov2d_bounds = helpers::compute_cov2d_bounds(cov2d);

    if !cov2d_bounds.valid {
        return; // zero determinant
    }
    conics[idx] = cov2d_bounds.conic;

    // compute the projected mean
    let center = helpers::project_pix(projmat, p_world, img_size, vec2f(cx, cy));

    let tile_minmax = helpers::get_tile_bbox(center, cov2d_bounds.radius, tile_bounds, block_width);
    let tile_min = tile_minmax.xy;
    let tile_max = tile_minmax.zw;
    let tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);

    if tile_area <= 0 {
        return;
    }

    num_tiles_hit[idx] = i32(tile_area);
    depths[idx] = p_view.z;
    radii[idx] = i32(cov2d_bounds.radius);
    xys[idx] = center;
    compensation[idx] = comp;
}


// // kernel to map each intersection from tile ID and depth to a gaussian
// // writes output to isect_ids and gaussian_ids
// fn map_gaussian_to_intersects(
//     num_points: i32,
//     xys: array<vec2f>,
//     depths: array<f32>,
//     radii: array<i32>,
//     cum_tiles_hit: array<i32>,
//     tile_bounds: vec3u,
//     block_width: u32,
//     isect_ids: array<u64>,
//     gaussian_ids: array<i32>,
//     @builtin(global_invocation_id) global_id: vec3u,
//     @builtin(local_invocation_id) local_id: vec3u,
//     @builtin(workgroup_id) workgroup_id: vec3u,
// ) {
//     let idx = local_id.x;
//     if idx >= num_points {
//         return;
//     }

//     if radii[idx] <= 0 {
//         return;
//     }
//     // get the tile bbox for gaussian
//     let center = xys[idx];
//     let tile_minmax = get_tile_bbox(center, radii[idx], tile_bounds, block_width);
    
//     // update the intersection info for all tiles this gaussian hits
//     let cur_idx = select(0, cum_tiles_hit[idx - 1], idx == 0);
    
//     // TODO: What in the ever loving god??
//     // let depth_id = (int64_t) * (int32_t * )&(depths[idx]);
//     let depth_id = 0;

//     for (let i = tile_min.y; i < tile_max.y; i++) {
//         for (let j = tile_min.x; j < tile_max.x; j++) {
//             // isect_id is tile ID and depth as int32
//             let tile_id = i * tile_bounds.x + j; // tile within image
//             isect_ids[cur_idx] = (tile_id << 32) | depth_id; // tile | depth id
//             gaussian_ids[cur_idx] = idx;                     // 3D gaussian id
//             cur_idx++; // handles gaussians that hit more than one tile
//         }
//     }
// }

// // kernel to map sorted intersection IDs to tile bins
// // expect that intersection IDs are sorted by increasing tile ID
// // i.e. intersections of a tile are in contiguous chunks
// fn get_tile_bin_edges(
//     num_intersects: i32, 
//     isect_ids_sorted: array<i64>, 
//     ile_bins: array<vec2i>,
//     @builtin(global_invocation_id) global_id: vec3u,
//     @builtin(local_invocation_id) local_id: vec3u,
//     @builtin(workgroup_id) workgroup_id: vec3u) {
//     let idx = local_id.x;

//     if idx >= num_intersects {
//         return;
//     }

//     // save the indices where the tile_id changes
//     let cur_tile_idx = i32(isect_ids_sorted[idx] >> 32);
//     if idx == 0 || idx == num_intersects - 1 {
//         if idx == 0 {
//             tile_bins[cur_tile_idx].x = 0;
//         }

//         if idx == num_intersects - 1 {
//             tile_bins[cur_tile_idx].y = num_intersects;
//         }
//     }

//     if idx == 0 {
//         return;
//     }

//     let prev_tile_idx = i32(isect_ids_sorted[idx - 1] >> 32);

//     if prev_tile_idx != cur_tile_idx {
//         tile_bins[prev_tile_idx].y = idx;
//         tile_bins[cur_tile_idx].x = idx;
//         return;
//     }
// }


// given v_xy_pix, get v_xyz
// fn project_pix_vjp(transform: mat4x4f, p: vec3f, img_size: vec3u, v_xy: vec2f) -> vec3f {
//     let t = transform * p;
//     let rw = 1.f / (t.w + 1e-6f);

//     let v_ndc = vec3f(0.5f * img_size.x * v_xy.x, 0.5f * img_size.y * v_xy.y);
//     let v_t = vec4f(v_ndc.x * rw, v_ndc.y * rw, 0.0, -(v_ndc.x * t.x + v_ndc.y * t.y) * rw * rw);
    
//     // df / d_world = df / d_cam * d_cam / d_world
//     // = v_t * mat[:3, :4]

//     // TOOD: Row column major? Why just raw indexing it? Ugh.
//     return vec3f(
//         transform[0] * v_t.x + transform[4] * v_t.y + transform[8] * v_t.z + transform[12] * v_t.w,
//         transform[1] * v_t.x + transform[5] * v_t.y + transform[9] * v_t.z + transform[13] * v_t.w,
//         transform[2] * v_t.x + transform[6] * v_t.y + transform[10] * v_t.z + transform[14] * v_t.w,
//     );
// }


