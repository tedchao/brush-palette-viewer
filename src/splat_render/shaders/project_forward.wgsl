#import helpers;

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;

@group(0) @binding(1) var<storage, read> means: array<vec4f>;
@group(0) @binding(2) var<storage, read> log_scales: array<vec4f>;
@group(0) @binding(3) var<storage, read> quats: array<vec4f>;
@group(0) @binding(4) var<storage, read> coeffs: array<f32>;
@group(0) @binding(5) var<storage, read> opacities: array<f32>;

@group(0) @binding(6) var<storage, read_write> compact_ids: array<u32>;
@group(0) @binding(7) var<storage, read_write> remap_ids: array<u32>;
@group(0) @binding(8) var<storage, read_write> xys: array<vec2f>;
@group(0) @binding(9) var<storage, read_write> depths: array<f32>;
@group(0) @binding(10) var<storage, read_write> colors: array<vec4f>;

@group(0) @binding(11) var<storage, read_write> radii: array<u32>;
@group(0) @binding(12) var<storage, read_write> cov2ds: array<vec4f>;
@group(0) @binding(13) var<storage, read_write> num_tiles_hit: array<u32>;
@group(0) @binding(14) var<storage, read_write> num_visible: array<atomic<u32>>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

struct Uniforms {
    // View matrix transform world to view position.
    viewmat: mat4x4f,
    // Focal (fx, fy).
    focal: vec2f,
    // Camera center (cx, cy).
    pixel_center: vec2f,
    // Img resolution (w, h)
    img_size: vec2u,
    // Nr. of tiles on each axis (w, h)
    tile_bounds: vec2u,
    // Width of blocks image is divided into.
    block_width: u32,
    // Near clip threshold.
    clip_thresh: f32,
    // num of sh coefficients per channel.
    num_sh_coeffs: u32,
    // World position of camera
    camera_point: vec3f,
}

fn project_pix(fxfy: vec2f, p_view: vec3f, pp: vec2f) -> vec2f {
    let p_proj = p_view.xy / max(p_view.z, 1e-6f);
    return p_proj * fxfy + pp;
}

struct Sh0 {
    c0: vec3f,
}

struct Sh1 {
    c0: vec3f,
    c1: vec3f,
    c2: vec3f,
}

struct Sh2 {
    c0: vec3f,
    c1: vec3f,
    c2: vec3f,
    c3: vec3f,
    c4: vec3f,
}

struct Sh3 {
    c0: vec3f,
    c1: vec3f,
    c2: vec3f,
    c3: vec3f,
    c4: vec3f,
    c5: vec3f,
    c6: vec3f,
}

struct Sh4 {
    c0: vec3f,
    c1: vec3f,
    c2: vec3f,
    c3: vec3f,
    c4: vec3f,
    c5: vec3f,
    c6: vec3f,
    c7: vec3f,
    c8: vec3f,
}

// Evaluate spherical harmonics bases at unit direction for high orders using approach described by
// Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
// See https://jcgt.org/published/0002/02/06/ for reference implementation
fn sh_coeffs_to_color(
    degree: u32,
    viewdir: vec3f,
    sh0: Sh0,
    sh1: Sh1,
    sh2: Sh2,
    sh3: Sh3,
    sh4: Sh4,
) -> vec3f {
    var colors = 0.2820947917738781f * sh0.c0;

    if (degree < 1) {
        return colors;
    }


    let viewdir_norm = normalize(viewdir);
    let x = viewdir_norm.x;
    let y = viewdir_norm.y;
    let z = viewdir_norm.z;

    let fTmp0A = 0.48860251190292f;
    colors += fTmp0A *
                    (-y * sh1.c0 +
                    z * sh1.c1 -
                    x * sh1.c2);

    if (degree < 2) {
        return colors;
    }
    let z2 = z * z;

    let fTmp0B = -1.092548430592079 * z;
    let fTmp1A = 0.5462742152960395;
    let fC1 = x * x - y * y;
    let fS1 = 2.f * x * y;
    let pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    let pSH7 = fTmp0B * x;
    let pSH5 = fTmp0B * y;
    let pSH8 = fTmp1A * fC1;
    let pSH4 = fTmp1A * fS1;


    colors +=
        pSH4 * sh2.c0 + 
        pSH5 * sh2.c1 +
        pSH6 * sh2.c2 + 
        pSH7 * sh2.c3 +
        pSH8 * sh2.c4;

    if (degree < 3) {
        return colors;
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
    colors +=   pSH9  * sh3.c0 +
                pSH10 * sh3.c1 +
                pSH11 * sh3.c2 +
                pSH12 * sh3.c3 +
                pSH13 * sh3.c4 +
                pSH14 * sh3.c5 +
                pSH15 * sh3.c6;
    
    if (degree < 4) {
        return colors;
    }

    let fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    let fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    let fTmp2B = -1.770130769779931f * z;
    let fTmp3A = 0.6258357354491763f;
    let fC3 = x * fC2 - y * fS2;
    let fS3 = x * fS2 + y * fC2;
    let pSH20 = (1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6);
    let pSH21 = fTmp0D * x;
    let pSH19 = fTmp0D * y;
    let pSH22 = fTmp1C * fC1;
    let pSH18 = fTmp1C * fS1;
    let pSH23 = fTmp2B * fC2;
    let pSH17 = fTmp2B * fS2;
    let pSH24 = fTmp3A * fC3;
    let pSH16 = fTmp3A * fS3;
    colors += pSH16 * sh4.c0 +
                pSH17 * sh4.c1 +
                pSH18 * sh4.c2 +
                pSH19 * sh4.c3 +
                pSH20 * sh4.c4 +
                pSH21 * sh4.c5 +
                pSH22 * sh4.c6 +
                pSH23 * sh4.c7 +
                pSH24 * sh4.c8;
    return colors;
}

fn read_coeffs(base_id: ptr<function, u32>) -> vec3f {
    let ret = vec3f(coeffs[*base_id + 0], coeffs[*base_id + 1], coeffs[*base_id + 2]);
    *base_id += 3u;
    return ret;
}

// Kernel function for projecting gaussians.
// Each thread processes one gaussian
@compute
@workgroup_size(helpers::SPLATS_PER_GROUP, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;
    let num_points = arrayLength(&means);

    if idx >= num_points {
        return;
    }

    // 0 buffer to mark gaussian as not visible.
    radii[idx] = 0u;
    // Zero out number of tiles hit before cumulative sum.
    num_tiles_hit[idx] = 0u;
    depths[idx] = 1e30;

    let viewmat = uniforms.viewmat;
    let focal = uniforms.focal;
    let pixel_center = uniforms.pixel_center;

    let clip_thresh = uniforms.clip_thresh;

    // Project world space to camera space.
    let p_world = means[idx].xyz;

    let W = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let p_view = W * p_world + viewmat[3].xyz;

    if p_view.z <= clip_thresh {
        return;
    }

    // compute the projected covariance
    let scale = exp(log_scales[idx].xyz);
    let quat = quats[idx];

    let R = helpers::quat_to_rotmat(quat);
    let S = helpers::scale_to_mat(scale);
    let M = R * S;
    let V = M * transpose(M);
    
    let tan_fov = 0.5 * vec2f(uniforms.img_size.xy) / focal;
    
    let lims = 1.3 * tan_fov;
    // Get ndc coords +- clipped to the frustum.
    let t = p_view.z * clamp(p_view.xy / p_view.z, -lims, lims);
    let rz = 1.0 / p_view.z;
    let rz2 = rz * rz;

    let J = mat3x3f(
        vec3f(focal.x * rz, 0.0, 0.0),
        vec3f(0.0, focal.y * rz, 0.0),
        vec3f(-focal.x * t.x * rz2, -focal.y * t.y * rz2, 0.0)
    );

    let T = J * W;
    let cov = T * V * transpose(T);

    let c00 = cov[0][0];
    let c11 = cov[1][1];
    let c01 = cov[0][1];

    // add a little blur along axes and save upper triangular elements
    let cov2d = vec3f(c00 + helpers::COV_BLUR, c01, c11 + helpers::COV_BLUR);
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;

    // inverse of 2x2 cov2d matrix
    let b = 0.5 * (cov2d.x + cov2d.z);
    let v1 = b + sqrt(max(0.1f, b * b - det));
    let v2 = b - sqrt(max(0.1f, b * b - det));

    // Render 3 sigma of covariance.
    // TODO: Make this a setting? Could save a good amount of pixels
    // that need to be rendered. Also: pick gaussians that REALLY clip to 0.
    // TODO: Is rounding down better? Eg. for gaussians <pixel size, just skip?
    let radius = u32(ceil(3.0 * sqrt(max(0.0, max(v1, v2)))));

    if radius == 0u {
        return;
    }

    // compute the projected mean
    let center = project_pix(focal, p_view, pixel_center);
    let tile_minmax = helpers::get_tile_bbox(center, radius, uniforms.tile_bounds);
    let tile_min = tile_minmax.xy;
    let tile_max = tile_minmax.zw;

    var tile_area = 0u;
    for (var ty = tile_min.y; ty < tile_max.y; ty++) {
        for (var tx = tile_min.x; tx < tile_max.x; tx++) {
            if helpers::can_be_visible(vec2u(tx, ty), center, radius) {
                tile_area += 1u;
            }
        }
    }

    if tile_area > 0u {
        // Now write all the data to the buffers.
        let write_id = atomicAdd(&num_visible[0], 1u);
        // TODO: Remove this when burn fixes int_arange...
        compact_ids[write_id] = write_id;
        remap_ids[write_id] = idx;

        var base_id = idx * uniforms.num_sh_coeffs * 3;

        var sh0 = Sh0();
        var sh1 = Sh1();
        var sh2 = Sh2();
        var sh3 = Sh3();
        var sh4 = Sh4();

        var degree = 0u;
        sh0.c0 = read_coeffs(&base_id);

        if uniforms.num_sh_coeffs > 1 {
            degree = 1u;
            sh1.c0 = read_coeffs(&base_id);
            sh1.c1 = read_coeffs(&base_id);
            sh1.c2 = read_coeffs(&base_id);

            if uniforms.num_sh_coeffs > 4 {
                degree = 2u;
                sh2.c0 = read_coeffs(&base_id);
                sh2.c1 = read_coeffs(&base_id);
                sh2.c2 = read_coeffs(&base_id);
                sh2.c3 = read_coeffs(&base_id);
                sh2.c4 = read_coeffs(&base_id);

                if uniforms.num_sh_coeffs > 9 {
                    degree = 3u;
                    sh3.c0 = read_coeffs(&base_id);
                    sh3.c1 = read_coeffs(&base_id);
                    sh3.c2 = read_coeffs(&base_id);
                    sh3.c3 = read_coeffs(&base_id);
                    sh3.c4 = read_coeffs(&base_id);
                    sh3.c5 = read_coeffs(&base_id);
                    sh3.c6 = read_coeffs(&base_id);

                    if uniforms.num_sh_coeffs > 16 {
                        degree = 4u;

                        sh4.c0 = read_coeffs(&base_id);
                        sh4.c1 = read_coeffs(&base_id);
                        sh4.c2 = read_coeffs(&base_id);
                        sh4.c3 = read_coeffs(&base_id);
                        sh4.c4 = read_coeffs(&base_id);
                        sh4.c5 = read_coeffs(&base_id);
                        sh4.c6 = read_coeffs(&base_id);
                        sh4.c7 = read_coeffs(&base_id);
                        sh4.c8 = read_coeffs(&base_id);
                    }
                }
            }
        }

        let opac = sigmoid(opacities[idx]);

        // let viewdir = p_world - uniforms.camera_point;
        let viewdir = p_world - uniforms.camera_point;
        let color = max(sh_coeffs_to_color(degree, viewdir, sh0, sh1, sh2, sh3, sh4) + vec3f(0.5), vec3f(0.0));
        colors[write_id] = vec4f(color, opac);
        depths[write_id] = p_view.z;
        num_tiles_hit[write_id] = tile_area;
        radii[write_id] = radius;
        xys[write_id] = center;
        cov2ds[write_id] = vec4f(cov2d, 1.0);
    }
}
