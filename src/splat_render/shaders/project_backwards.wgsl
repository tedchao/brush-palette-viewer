#import helpers;

struct Uniforms {
    // View matrix transform world to view position.
    viewmat: mat4x4f,
    // Focal of camera (fx, fy)
    focal: vec2f,
    // Img resolution (w, h)
    img_size: vec2u,
    // Degree of sh coeffecients used.
    sh_degree: u32,
}
@group(0) @binding(0) var<storage> uniforms: Uniforms;

@group(0) @binding(1) var<storage> means: array<f32>; // packed vec3
@group(0) @binding(2) var<storage> log_scales: array<f32>; // packed vec3
@group(0) @binding(3) var<storage> quats: array<vec4f>;
@group(0) @binding(4) var<storage> raw_opacities: array<f32>;

@group(0) @binding(5) var<storage> conic_comps: array<vec4f>;

@group(0) @binding(6) var<storage> cum_tiles_hit: array<u32>;
@group(0) @binding(7) var<storage> v_xy: array<vec2f>;
@group(0) @binding(8) var<storage> v_conic: array<vec4f>;
@group(0) @binding(9) var<storage> v_colors: array<vec4f>;

@group(0) @binding(10) var<storage> num_visible: array<u32>;
@group(0) @binding(11) var<storage> global_from_compact_gid: array<u32>;
@group(0) @binding(12) var<storage> compact_from_depthsort_gid: array<u32>;

@group(0) @binding(13) var<storage, read_write> v_means_agg: array<f32>; // packed vec3
@group(0) @binding(14) var<storage, read_write> v_xys_agg: array<vec2f>;
@group(0) @binding(15) var<storage, read_write> v_scales_agg: array<f32>; // packed vec3

@group(0) @binding(16) var<storage, read_write> v_quats_agg: array<vec4f>;
@group(0) @binding(17) var<storage, read_write> v_coeffs_agg: array<f32>;
@group(0) @binding(18) var<storage, read_write> v_opac_agg: array<f32>;


struct ShCoeffs {
    b0_c0: vec3f,

    b1_c0: vec3f,
    b1_c1: vec3f,
    b1_c2: vec3f,

    b2_c0: vec3f,
    b2_c1: vec3f,
    b2_c2: vec3f,
    b2_c3: vec3f,
    b2_c4: vec3f,

    b3_c0: vec3f,
    b3_c1: vec3f,
    b3_c2: vec3f,
    b3_c3: vec3f,
    b3_c4: vec3f,
    b3_c5: vec3f,
    b3_c6: vec3f,

    b4_c0: vec3f,
    b4_c1: vec3f,
    b4_c2: vec3f,
    b4_c3: vec3f,
    b4_c4: vec3f,
    b4_c5: vec3f,
    b4_c6: vec3f,
    b4_c7: vec3f,
    b4_c8: vec3f,
}

fn num_sh_coeffs(degree: u32) -> u32 {
    return (degree + 1) * (degree + 1);
}

fn sh_coeffs_to_color_fast_vjp(
    degree: u32,
    viewdir: vec3f,
    v_colors: vec3f,
) -> ShCoeffs {
    var v_coeffs = ShCoeffs();

    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    v_coeffs.b0_c0 = 0.2820947917738781f * v_colors;

    if (degree < 1) {
        return v_coeffs;
    }
    let norm = normalize(viewdir);
    let x = viewdir.x;
    let y = viewdir.y;
    let z = viewdir.z;

    let fTmp0A = 0.48860251190292f;
    v_coeffs.b1_c0 = -fTmp0A * y * v_colors;
    v_coeffs.b1_c1 = fTmp0A * z * v_colors;
    v_coeffs.b1_c2 = -fTmp0A * x * v_colors;

    if (degree < 2) {
        return v_coeffs;
    }

    let z2 = z * z;
    let fTmp0B = -1.092548430592079f * z;
    let fTmp1A = 0.5462742152960395f;
    let fC1 = x * x - y * y;
    let fS1 = 2.f * x * y;
    let pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    let pSH7 = fTmp0B * x;
    let pSH5 = fTmp0B * y;
    let pSH8 = fTmp1A * fC1;
    let pSH4 = fTmp1A * fS1;
    v_coeffs.b2_c0 = pSH4 * v_colors;
    v_coeffs.b2_c1 = pSH5 * v_colors;
    v_coeffs.b2_c2 = pSH6 * v_colors;
    v_coeffs.b2_c3 = pSH7 * v_colors;
    v_coeffs.b2_c4 = pSH8 * v_colors;

    if (degree < 3) {
        return v_coeffs;
    }

    let fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    let fTmp1B = 1.445305721320277f * z;
    let fTmp2A = -0.5900435899266435f;
    let fC2 = x * fC1 - y * fS1;
    let fS2 = x * fS1 + y * fC1;
    let pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    let pSH13 = fTmp0C * x;
    let pSH11 = fTmp0C * y;
    let pSH14 = fTmp1B * fC1;
    let pSH10 = fTmp1B * fS1;
    let pSH15 = fTmp2A * fC2;
    let pSH9  = fTmp2A * fS2;
    v_coeffs.b3_c0 = pSH9 * v_colors;
    v_coeffs.b3_c1 = pSH10 * v_colors;
    v_coeffs.b3_c2 = pSH11 * v_colors;
    v_coeffs.b3_c3 = pSH12 * v_colors;
    v_coeffs.b3_c4 = pSH13 * v_colors;
    v_coeffs.b3_c5 = pSH14 * v_colors;
    v_coeffs.b3_c6 = pSH15 * v_colors;
    if (degree < 4) {
        return v_coeffs;
    }

    let fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    let fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    let fTmp2B = -1.770130769779931f * z;
    let fTmp3A = 0.6258357354491763f;
    let fC3 = x * fC2 - y * fS2;
    let fS3 = x * fS2 + y * fC2;
    let pSH20 = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
    let pSH21 = fTmp0D * x;
    let pSH19 = fTmp0D * y;
    let pSH22 = fTmp1C * fC1;
    let pSH18 = fTmp1C * fS1;
    let pSH23 = fTmp2B * fC2;
    let pSH17 = fTmp2B * fS2;
    let pSH24 = fTmp3A * fC3;
    let pSH16 = fTmp3A * fS3;
    v_coeffs.b4_c0 = pSH16 * v_colors;
    v_coeffs.b4_c1 = pSH17 * v_colors;
    v_coeffs.b4_c2 = pSH18 * v_colors;
    v_coeffs.b4_c3 = pSH19 * v_colors;
    v_coeffs.b4_c4 = pSH20 * v_colors;
    v_coeffs.b4_c5 = pSH21 * v_colors;
    v_coeffs.b4_c6 = pSH22 * v_colors;
    v_coeffs.b4_c7 = pSH23 * v_colors;
    v_coeffs.b4_c8 = pSH24 * v_colors;
    return v_coeffs;
}

fn write_coeffs(base_id: ptr<function, u32>, val: vec3f) {
    v_coeffs_agg[*base_id + 0] = val.x;
    v_coeffs_agg[*base_id + 1] = val.y;
    v_coeffs_agg[*base_id + 2] = val.z;
    *base_id += 3u;
}

fn project_pix_vjp(fxfy: vec2f, p_view: vec3f, v_xy: vec2f) -> vec3f {
    let rw = 1.0f / (p_view.z + 1e-6f);
    let v_proj = fxfy * v_xy;
    return vec3f(v_proj.x * rw, v_proj.y * rw, -(v_proj.x * p_view.x + v_proj.y * p_view.y) * rw * rw);
}

fn quat_to_rotmat_vjp(quat: vec4f, v_R: mat3x3f) -> vec4f {
    let w = quat.x;
    let x = quat.y;
    let y = quat.z;
    let z = quat.w;

    return vec4f(
        // w element stored in x field
        2.f * (
                  // v_quat.w = 2.f * (
                  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                  z * (v_R[0][1] - v_R[1][0])
              ),
        // x element in y field
        2.f *
        (
            // v_quat.x = 2.f * (
            -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
            z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
        ),
        // y element in z field
        2.f *
        (
            // v_quat.y = 2.f * (
            x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
            z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
        ),
        // z element in w field
        2.f *
        (
            // v_quat.z = 2.f * (
            x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
            2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
        )
    );
}

fn cov2d_to_conic_vjp(conic: vec3f, v_conic: vec3f) -> vec3f {
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    let X = mat2x2f(vec2f(conic.x, conic.y), vec2f(conic.y, conic.z));
    let G = mat2x2f(vec2f(v_conic.x, v_conic.y / 2.0), 
                    vec2f(v_conic.y / 2.0, v_conic.z));
    let v_Sigma = X * G * X;

    return -vec3f(
        v_Sigma[0][0],
        v_Sigma[1][0] + v_Sigma[0][1],
        v_Sigma[1][1]
    );
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn v_sigmoid(x: f32) -> f32 {
    return sigmoid(x) * (1.0 - sigmoid(x));
}

@compute
@workgroup_size(helpers::SPLATS_PER_GROUP, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let depthsort_gid = global_id.x;

    if depthsort_gid >= num_visible[0] {
        return;
    }

    let compact_gid = compact_from_depthsort_gid[depthsort_gid];
    let global_gid = global_from_compact_gid[compact_gid];

    // Aggregate the gradients that are written per tile.
    var v_xy_agg = vec2f(0.0);
    var v_conic_agg = vec3f(0.0);
    var v_colors_agg = vec4f(0.0);

    var grad_idx = 0u;
    if depthsort_gid > 0 {
        grad_idx = cum_tiles_hit[depthsort_gid - 1u];
    }
    for(; grad_idx < cum_tiles_hit[depthsort_gid]; grad_idx++) {
        v_xy_agg += v_xy[grad_idx];
        v_conic_agg += v_conic[grad_idx].xyz;
        v_colors_agg += v_colors[grad_idx];
    }

    v_xys_agg[global_gid] = v_xy_agg;

    let viewmat = uniforms.viewmat;
    let focal = uniforms.focal;
    
    let mean = vec3f(means[global_gid * 3 + 0], means[global_gid * 3 + 1], means[global_gid * 3 + 2]);
    let scale = exp(vec3f(log_scales[global_gid * 3 + 0], log_scales[global_gid * 3 + 1], log_scales[global_gid * 3 + 2]));
    let quat = quats[global_gid];

    let W = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let p_view = W * mean + viewmat[3].xyz;
    var v_mean = transpose(W) * project_pix_vjp(focal, p_view, v_xy_agg);

    // get z gradient contribution to mean3d gradient
    // TODO: Is this indeed only active if depth is supervised?
    // z = viemwat[8] * mean3d.x + viewmat[9] * mean3d.y + viewmat[10] *
    // mean3d.z + viewmat[11]
    // let v_z = v_depth[idx];
    // v_mean += viewmat[2].xyz * v_z;
    // get v_cov2d
    // compute vjp from df/d_conic to df/c_cov2d
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    let conic_comp = conic_comps[compact_gid];
    let conic = conic_comp.xyz;
    let compensation = conic_comp.w;
    var v_cov2d = cov2d_to_conic_vjp(conic, v_conic_agg);
    
    // // Compensation is applied as opac * comp
    // // so deriv is v_opac.
    // // TODO: Re-enable compensation.
    // let v_compensation = v_opacity[idx] * 0.0;
    // // comp = sqrt(det(cov2d - 0.3 I) / det(cov2d))
    // // conic = inverse(cov2d)
    // // df / d_cov2d = df / d comp * 0.5 / comp * [ d comp^2 / d cov2d ]
    // // d comp^2 / d cov2d = (1 - comp^2) * conic - 0.3 I * det(conic)
    // let inv_det = conic.x * conic.z - conic.y * conic.y;
    // let one_minus_sqr_comp = 1.0 - compensation * compensation;
    // let v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
    // v_cov2d += vec3f(
    //     v_sqr_comp * (one_minus_sqr_comp * conic.x - helpers::COV_BLUR * inv_det),
    //     2.0 * v_sqr_comp * (one_minus_sqr_comp * conic.y),
    //     v_sqr_comp * (one_minus_sqr_comp * conic.z - helpers::COV_BLUR * inv_det)
    // );

    // get v_cov3d (and v_mean3d contribution)
    let rz = 1.0 / p_view.z;
    let rz2 = rz * rz;

    let J = mat3x3f(
        vec3f(focal.x * rz, 0.0f, 0.0f),
        vec3f(0.0f, focal.y * rz, 0.0f),
        vec3f(-focal.x * p_view.x * rz2, -focal.y * p_view.y * rz2, 0.0f)
    );

    let R = helpers::quat_to_rotmat(quat);
    let S = helpers::scale_to_mat(scale);
    let M = R * S;
    let V = M * transpose(M);

    // cov = T * V * Tt; G = df/dcov = v_cov
    // -> d/dV = Tt * G * T
    // -> df/dT = G * T * Vt + Gt * T * V
    let v_cov = mat3x3f(
        vec3f(v_cov2d.x, 0.5 * v_cov2d.y, 0.0),
        vec3f(0.5 * v_cov2d.y, v_cov2d.z, 0.0),
        vec3f(0.0, 0.0, 0.0),
    );

    let T = J * W;
    let Tt = transpose(T);
    let Vt = transpose(V);

    let v_V = Tt * v_cov * T;
    let v_T = v_cov * T * Vt + transpose(v_cov) * T * V;

    // vjp of cov3d parameters
    // v_cov3d_i = v_V : dV/d_cov3d_i
    // where : is frobenius inner product
    let v_cov3d0 = v_V[0][0];
    let v_cov3d1 = v_V[0][1] + v_V[1][0];
    let v_cov3d2 = v_V[0][2] + v_V[2][0];
    let v_cov3d3 = v_V[1][1];
    let v_cov3d4 = v_V[1][2] + v_V[2][1];
    let v_cov3d5 = v_V[2][2];

    // compute df/d_mean3d
    // T = J * W
    let v_J = v_T * transpose(W);
    let rz3 = rz2 * rz;
    let v_t = vec3f(
        -focal.x * rz2 * v_J[2][0],
        -focal.y * rz2 * v_J[2][1],
        -focal.x * rz2 * v_J[0][0] + 2.0 * focal.x * p_view.x * rz3 * v_J[2][0] -
            focal.y * rz2 * v_J[1][1] + 2.0 * focal.y * p_view.y * rz3 * v_J[2][1]
    );

    v_mean += vec3f(
        dot(v_t, W[0]), 
        dot(v_t, W[1]), 
        dot(v_t, W[2])
    );

    // cov3d is upper triangular elements of matrix
    // off-diagonal elements count grads from both ij and ji elements,
    // must halve when expanding back into symmetric matrix
    // TODO: Is this definitely the same as above?
    let v_V_symm = mat3x3f(
        vec3f(
            v_cov3d0,
            0.5 * v_cov3d1,
            0.5 * v_cov3d2,
        ),
        vec3f(
            0.5 * v_cov3d1,
            v_cov3d3,
            0.5 * v_cov3d4,
        ),
        vec3f(
            0.5 * v_cov3d2,
            0.5 * v_cov3d4,
            v_cov3d5
        )
    );

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    let v_M = 2.0 * v_V_symm * M;

    let v_scale = vec3f(
        dot(R[0], v_M[0]),
        dot(R[1], v_M[1]),
        dot(R[2], v_M[2]),
    );
    let v_scale_exp = v_scale * scale;

    let v_R = v_M * S;
    let v_quat = quat_to_rotmat_vjp(quat, v_R);

    v_quats_agg[global_gid] = v_quat;

    // Write out components of scale/mean gradients.
    for(var i = 0u; i < 3; i++) {
        v_scales_agg[global_gid * 3 + i] = v_scale_exp[i];
    }
    for(var i = 0u; i < 3; i++) {
        v_means_agg[global_gid * 3 + i] = v_mean[i];
    }

    // Write SH gradients.
    // TODO: Get real viewdir.
    let v_coeff = sh_coeffs_to_color_fast_vjp(uniforms.sh_degree, vec3f(0.0, 0.0, 1.0), v_colors_agg.xyz);
    let num_coeffs = num_sh_coeffs(uniforms.sh_degree);
    var base_id = global_gid * num_coeffs * 3;

    write_coeffs(&base_id, v_coeff.b0_c0);
    if uniforms.sh_degree > 0 {
        write_coeffs(&base_id, v_coeff.b1_c0);
        write_coeffs(&base_id, v_coeff.b1_c1);
        write_coeffs(&base_id, v_coeff.b1_c2);
        if uniforms.sh_degree > 1 {
            write_coeffs(&base_id, v_coeff.b2_c0);
            write_coeffs(&base_id, v_coeff.b2_c1);
            write_coeffs(&base_id, v_coeff.b2_c2);
            write_coeffs(&base_id, v_coeff.b2_c3);
            write_coeffs(&base_id, v_coeff.b2_c4);
            if uniforms.sh_degree > 2 {
                write_coeffs(&base_id, v_coeff.b3_c0);
                write_coeffs(&base_id, v_coeff.b3_c1);
                write_coeffs(&base_id, v_coeff.b3_c2);
                write_coeffs(&base_id, v_coeff.b3_c3);
                write_coeffs(&base_id, v_coeff.b3_c4);
                write_coeffs(&base_id, v_coeff.b3_c5);
                write_coeffs(&base_id, v_coeff.b3_c6);
                if uniforms.sh_degree > 3 {
                    write_coeffs(&base_id, v_coeff.b4_c0);
                    write_coeffs(&base_id, v_coeff.b4_c1);
                    write_coeffs(&base_id, v_coeff.b4_c2);
                    write_coeffs(&base_id, v_coeff.b4_c3);
                    write_coeffs(&base_id, v_coeff.b4_c4);
                    write_coeffs(&base_id, v_coeff.b4_c5);
                    write_coeffs(&base_id, v_coeff.b4_c6);
                    write_coeffs(&base_id, v_coeff.b4_c7);
                    write_coeffs(&base_id, v_coeff.b4_c8);
                }
            }
        }
    }

    // TODO: Could use opacity activation? Doesn't really matter
    let raw_opac = raw_opacities[global_gid];
    v_opac_agg[global_gid] = v_colors_agg.w * v_sigmoid(raw_opac);
}