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

// device helper to get 3D covariance from scale and quat parameters
fn quat_to_rotmat(quat: vec4f) -> mat3x3f {
    // quat to rotation matrix
    let quat_norm = normalize(quat);

    let x = quat_norm.x;
    let y = quat_norm.y;
    let z = quat_norm.z;
    let w = quat_norm.w;
    // See https://www.songho.ca/opengl/gl_quaternion.html
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

fn scale_to_mat(scale: vec3f, glob_scale: f32) -> mat3x3f {
    return mat3x3(
        vec3f(glob_scale * scale.x, 0, 0), 
        vec3f(0, glob_scale * scale.y, 0), 
        vec3f(0, 0, glob_scale * scale.z)
    );
}

fn scale_rot_to_cov3d(scale: vec3f, glob_scale: f32, quat: vec4f) -> mat3x3f {
    let R = quat_to_rotmat(quat);
    let S = scale_to_mat(scale, glob_scale);
    let M = R * S;
    return M * transpose(M);
}

fn ndc2pix(x: vec2f, W: vec2f, cx: vec2f) -> vec2f {
    return 0.5f * W * x + cx - 0.5f;
}

fn project_pix(transform: mat4x4f, p: vec3f, img_size: vec2u, pp: vec2f) -> vec2f {
    let p_hom = transform * vec4f(p, 1.0f);
    let rw = 1.0f / (p_hom.w + 1e-6f);

    let p_proj = p_hom.xyz / (p_hom.w + 1e-6f);
    return ndc2pix(p_proj.xy, vec2f(img_size.xy), pp);
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

    // inverse of 2x2 cov2d matrix
    let conic = vec3f(cov2d.z, -cov2d.y, cov2d.x) / det;

    let b = 0.5f * (cov2d.x + cov2d.z);
    let v1 = b + sqrt(max(0.1f, b * b - det));
    let v2 = b - sqrt(max(0.1f, b * b - det));
    
    // take 3 sigma of covariance
    let radius = ceil(3.0f * sqrt(max(v1, v2)));

    return ComputeCov2DBounds(conic, radius, true);
}



// fn cov2d_to_conic_vjp(conic: vec3f, v_conic: vec3f) -> vec3f {
//     // conic = inverse cov2d
//     // df/d_cov2d = -conic * df/d_conic * conic
//     let X = mat2x2f(conic.x, conic.y, conic.y, conic.z);
//     let G = mat2x2f(v_conic.x, v_conic.y / 2.f, v_conic.y / 2.f, v_conic.z);
//     let v_Sigma = -X * G * X;
//     return vec3f(v_Sigma[0][0], v_Sigma[1][0] + v_Sigma[0][1], v_Sigma[1][1]);
// }
// fn cov2d_to_compensation_vjp(compensation: f32, conic: vec3f, v_compensation: f32, v_cov2d: vec3f) -> vec3f {
//     // comp = sqrt(det(cov2d - 0.3 I) / det(cov2d))
//     // conic = inverse(cov2d)
//     // df / d_cov2d = df / d comp * 0.5 / comp * [ d comp^2 / d cov2d ]
//     // d comp^2 / d cov2d = (1 - comp^2) * conic - 0.3 I * det(conic)
//     let inv_det = conic.x * conic.z - conic.y * conic.y;
//     let one_minus_sqr_comp = 1 - compensation * compensation;
//     let v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
//     var v_cov2d_ret = v_cov2d;
//     v_cov2d_ret.x += v_sqr_comp * (one_minus_sqr_comp * conic.x - 0.3 * inv_det);
//     v_cov2d_ret.y += 2 * v_sqr_comp * (one_minus_sqr_comp * conic.y);
//     v_cov2d_ret.z += v_sqr_comp * (one_minus_sqr_comp * conic.z - 0.3 * inv_det);
//     return v_cov2d;
// }
// fn quat_to_rotmat_vjp(quat: vec4f, v_R: mat3x3f) -> vec4f {
//     let s = inverseSqrt(
//         quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
//     );
//     let w = quat.x * s;
//     let x = quat.y * s;
//     let y = quat.z * s;
//     let z = quat.w * s;
//     var v_quat = vec4(0.0);
//     // v_R is COLUMN MAJOR
//     // w element stored in x field
//     v_quat.x = 2.f * (// v_quat.w = 2.f * (
//                   x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) + z * (v_R[0][1] - v_R[1][0]));
//     // x element in y field
//     v_quat.y = 2.f * (// v_quat.x = 2.f * (
//             -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) + z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1]));
//     // y element in z field
//     v_quat.z = 2.f * (// v_quat.y = 2.f * (
//             x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) + z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2]));
//     // z element in w field
//     v_quat.w = 2.f * (// v_quat.z = 2.f * (
//             x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) - 2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0]));
//     return v_quat;
// }

