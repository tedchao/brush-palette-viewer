
@group(0) @binding(0) var<storage, read> means3d: array<vec3f>;
@group(0) @binding(1) var<storage, read> scales: array<vec3f>;
@group(0) @binding(2) var<storage, read> quats: array<vec4f>;

@group(0) @binding(3) var<storage, read_write> radii: array<f32>;
@group(0) @binding(4) var<storage, read_write> num_tiles_hit: array<f32>;
@group(0) @binding(5) var<storage, read_write> covs3d: array<f32>;
@group(0) @binding(6) var<storage, read_write> conics: array<vec3f>;
@group(0) @binding(7) var<storage, read_write> depths: array<f32>;
@group(0) @binding(8) var<storage, read_write> xys: array<vec2f>;
@group(0) @binding(9) var<storage, read_write> compensation: array<f32>;

@group(0) @binding(10) var<storage, read> info_array: array<InfoBinding>;

struct InfoBinding {
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
    let idx = local_id.x;

    // Until burn supports adding in uniforms we read these from a tensor.
    let info = info_array[0];
    let num_points = info.num_points;
    let glob_scale = info.glob_scale;
    let viewmat = info.viewmat;
    let projmat = info.projmat;
    let intrins = info.intrins;
    let img_size = info.img_size;
    let tile_bounds = info.tile_bounds;
    let block_width = info.block_width;
    let clip_thresh = info.clip_thresh;

    if idx >= num_points {
        return;
    }

    radii[idx] = 0.0;
    num_tiles_hit[idx] = 0.0;

    let p_world = means3d[idx];

    let p_view = viewmat * vec4f(p_world, 1.0f);
    if p_view.z <= clip_thresh {
        return;
    }

    // compute the projected covariance
    let scale = scales[idx];
    let quat = quats[idx];

    let R = quat_to_rotmat(quat);
    let S = scale_to_mat(scale, glob_scale);
    let M = R * S;
    let tmp = M * transpose(M);

    // save upper right of matrix, as it's symmetric.
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

    // project to 2d with ewa approximation
    let fx = intrins.x;
    let fy = intrins.y;
    let cx = intrins.z;
    let cy = intrins.w;

    let tan_fovx = 0.5 * f32(img_size.x) / fx;
    let tan_fovy = 0.5 * f32(img_size.y) / fy;

    // Get the position from the view matrix.
    var t = viewmat * vec4f(p_world, 1.0f);

    // clip so that the covariance
    let lim_x = 1.3f * tan_fovx;
    let lim_y = 1.3f * tan_fovy;
    t.x = t.z * min(lim_x, max(-lim_x, t.x / t.z));
    t.y = t.z * min(lim_y, max(-lim_y, t.y / t.z));

    let rz = 1.0f / t.z;
    let rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    let J = mat3x3f(
        vec3f(
            fx * rz,
            0.0f,
            0.0f,
        ),
        vec3f(
            0.f,
            fy * rz,
            0.f
        ),
        vec3f(
            -fx * t.x * rz2,
            -fy * t.y * rz2,
            0.0f
        )
    );
    let T = J * mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);

    let V = mat3x3f(
        vec3f(
            covs0,
            covs1,
            covs2
        ),
        vec3f(
            covs1,
            covs3,
            covs4
        ),
        vec3f(
            covs2,
            covs4,
            covs5
        )
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

    let cov2d_bounds = compute_cov2d_bounds(cov2d);

    if !cov2d_bounds.valid {
        return; // zero determinant
    }

    conics[idx] = cov2d_bounds.conic;

    // compute the projected mean
    let center = project_pix(projmat, p_world, img_size, vec2f(cx, cy));

    let tile_minmax = get_tile_bbox(center, cov2d_bounds.radius, tile_bounds, block_width);
    let tile_area = (tile_minmax.z - tile_minmax.x) * (tile_minmax.w - tile_minmax.y);

    if tile_area <= 0 {
        return;
    }

    num_tiles_hit[idx] = f32(tile_area);
    depths[idx] = p_view.z;
    radii[idx] = f32(cov2d_bounds.radius);
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



fn ndc2pix(x: f32, W: f32, cx: f32) -> f32 {
    return 0.5f * W * x + cx - 0.5f;
}

fn get_bbox(center: vec2f, dims: vec2f, img_size: vec2u) -> vec4i {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    let bb_min_x = min(max(0, i32(center.x - dims.x)), i32(img_size.x));
    let bb_max_x = min(max(0, i32(center.x + dims.x + 1)), i32(img_size.x));

    let bb_min_y = min(max(0, i32(center.y - dims.y)), i32(img_size.y));
    let bb_max_y = min(max(0, i32(center.y + dims.y + 1)), i32(img_size.y));

    return vec4i(bb_min_x, bb_min_y, bb_max_y, bb_max_y);
}

fn get_tile_bbox(pix_center: vec2f, pix_radius: f32, tile_bounds: vec2u, block_size: u32) -> vec4i {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    let tile_center = pix_center / f32(block_size);
    let tile_radius = vec2f(pix_radius, pix_radius) / f32(block_size);

    return get_bbox(tile_center, tile_radius, tile_bounds);
}

struct ComputeCov2DBounds {
    conic: vec3f,
    radius: f32,
    valid: bool
}

fn compute_cov2d_bounds(cov2d: vec3f) -> ComputeCov2DBounds {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if det == 0.0f {
        return ComputeCov2DBounds(vec3f(0.0), 0.0, false);
    }
    let inv_det = 1.0f / det;

    // inverse of 2x2 cov2d matrix
    let conic = vec3f(cov2d.z * inv_det, -cov2d.y * inv_det, cov2d.x * inv_det);

    let b = 0.5f * (cov2d.x + cov2d.z);
    let v1 = b + sqrt(max(0.1f, b * b - det));
    let v2 = b - sqrt(max(0.1f, b * b - det));
    // take 3 sigma of covariance
    let radius = ceil(3.f * sqrt(max(v1, v2)));

    return ComputeCov2DBounds(vec3f(0.0), 0.0, true);
}

// compute vjp from df/d_conic to df/c_cov2d
fn cov2d_to_conic_vjp(conic: vec3f, v_conic: vec3f) -> vec3f {
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    let X = mat2x2f(conic.x, conic.y, conic.y, conic.z);
    let G = mat2x2f(v_conic.x, v_conic.y / 2.f, v_conic.y / 2.f, v_conic.z);
    let v_Sigma = -X * G * X;
    return vec3f(v_Sigma[0][0], v_Sigma[1][0] + v_Sigma[0][1], v_Sigma[1][1]);
}

fn cov2d_to_compensation_vjp(compensation: f32, conic: vec3f, v_compensation: f32, v_cov2d: vec3f) -> vec3f {
    // comp = sqrt(det(cov2d - 0.3 I) / det(cov2d))
    // conic = inverse(cov2d)
    // df / d_cov2d = df / d comp * 0.5 / comp * [ d comp^2 / d cov2d ]
    // d comp^2 / d cov2d = (1 - comp^2) * conic - 0.3 I * det(conic)
    let inv_det = conic.x * conic.z - conic.y * conic.y;
    let one_minus_sqr_comp = 1 - compensation * compensation;
    let v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
    var v_cov2d_ret = v_cov2d;
    v_cov2d_ret.x += v_sqr_comp * (one_minus_sqr_comp * conic.x - 0.3 * inv_det);
    v_cov2d_ret.y += 2 * v_sqr_comp * (one_minus_sqr_comp * conic.y);
    v_cov2d_ret.z += v_sqr_comp * (one_minus_sqr_comp * conic.z - 0.3 * inv_det);
    return v_cov2d;
}

fn project_pix(transform: mat4x4f, p: vec3f, img_size: vec2u, pp: vec2f) -> vec2f {
    let p_hom = transform * vec4f(p, 1.0f);
    let rw = 1.0f / (p_hom.w + 1e-6f);
    let p_proj = p_hom.xyz / (p_hom.w + 1e-6f);
    let img_f = vec2f(img_size);
    return vec2f(ndc2pix(p_proj.x, img_f.x, pp.x), ndc2pix(p_proj.y, img_f.y, pp.y));
}

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

fn quat_to_rotmat(quat: vec4f) -> mat3x3f {
    // quat to rotation matrix
    let s = inverseSqrt(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );

    let w = quat.x * s;
    let x = quat.y * s;
    let y = quat.z * s;
    let z = quat.w * s;

    return mat3x3(
        1.f - 2.f * (y * y + z * z),
        2.f * (x * y + w * z),
        2.f * (x * z - w * y),
        2.f * (x * y - w * z),
        1.f - 2.f * (x * x + z * z),
        2.f * (y * z + w * x),
        2.f * (x * z + w * y),
        2.f * (y * z - w * x),
        1.f - 2.f * (x * x + y * y)
    );
}

fn quat_to_rotmat_vjp(quat: vec4f, v_R: mat3x3f) -> vec4f {
    let s = inverseSqrt(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    let w = quat.x * s;
    let x = quat.y * s;
    let y = quat.z * s;
    let z = quat.w * s;

    var v_quat = vec4(0.0);
    // v_R is COLUMN MAJOR
    // w element stored in x field
    v_quat.x = 2.f * (// v_quat.w = 2.f * (
                  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) + z * (v_R[0][1] - v_R[1][0]));
    // x element in y field
    v_quat.y = 2.f * (// v_quat.x = 2.f * (
            -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) + z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1]));
    // y element in z field
    v_quat.z = 2.f * (// v_quat.y = 2.f * (
            x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) + z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2]));
    // z element in w field
    v_quat.w = 2.f * (// v_quat.z = 2.f * (
            x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) - 2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0]));
    return v_quat;
}

fn scale_to_mat(scale: vec3f, glob_scale: f32) -> mat3x3f {
    return mat3x3(
        vec3f(glob_scale * scale.x, 0, 0), 
        vec3f(0, glob_scale * scale.y, 0), 
        vec3f(0, 0, glob_scale * scale.z)
    );
}
