fn get_bbox(center: vec2f, dims: vec2f, bounds: vec2u) -> vec4u {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    let min = vec2u(clamp(vec2i(center - dims), vec2i(0), vec2i(bounds)));
    let max = vec2u(clamp(vec2i(center + dims + 1), vec2i(0), vec2i(bounds)));
    return vec4u(min, max);
}

fn get_tile_bbox(pix_center: vec2f, pix_radius: f32, tile_bounds: vec2u, block_size: u32) -> vec4u {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    let tile_center = pix_center / f32(block_size);
    let tile_radius = vec2f(pix_radius, pix_radius) / f32(block_size);

    return get_bbox(tile_center, tile_radius, tile_bounds);
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
