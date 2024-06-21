#import helpers;

struct Uniforms {
    // View matrix transform world to view position.
    viewmat: mat4x4f,
    // Img resolution (w, h)
    img_size: vec2u,

    tile_bounds: vec2u,

    sh_degree: vec4u,
}

@group(0) @binding(0) var<storage> uniforms: Uniforms;

@group(0) @binding(1) var<storage> means: array<f32>; // packed vec3
@group(0) @binding(2) var<storage> coeffs: array<f32>;
@group(0) @binding(3) var<storage> raw_opacities: array<f32>;
@group(0) @binding(4) var<storage> global_from_compact_gid: array<u32>;

@group(0) @binding(5) var<storage> xys: array<vec2f>;
@group(0) @binding(6) var<storage> conic_comps: array<vec4f>;
@group(0) @binding(7) var<storage> num_visible: u32;

@group(0) @binding(8) var<storage, read_write> colors: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> num_tiles_hit: array<u32>;

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

// Evaluate spherical harmonics bases at unit direction for high orders using approach described by
// Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
// See https://jcgt.org/published/0002/02/06/ for reference implementation
fn sh_coeffs_to_color(
    degree: u32,
    viewdir: vec3f,
    sh: ShCoeffs,
) -> vec3f {
    var colors = 0.2820947917738781f * sh.b0_c0;

    if (degree < 1) {
        return colors;
    }

    let x = viewdir.x;
    let y = viewdir.y;
    let z = viewdir.z;

    let fTmp0A = 0.48860251190292f;
    colors += fTmp0A *
                    (-y * sh.b1_c0 +
                    z * sh.b1_c1 -
                    x * sh.b1_c2);

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
        pSH4 * sh.b2_c0 + 
        pSH5 * sh.b2_c1 +
        pSH6 * sh.b2_c2 + 
        pSH7 * sh.b2_c3 +
        pSH8 * sh.b2_c4;

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
    colors +=   pSH9  * sh.b3_c0 +
                pSH10 * sh.b3_c1 +
                pSH11 * sh.b3_c2 +
                pSH12 * sh.b3_c3 +
                pSH13 * sh.b3_c4 +
                pSH14 * sh.b3_c5 +
                pSH15 * sh.b3_c6;
    
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
    colors += pSH16 * sh.b4_c0 +
                pSH17 * sh.b4_c1 +
                pSH18 * sh.b4_c2 +
                pSH19 * sh.b4_c3 +
                pSH20 * sh.b4_c4 +
                pSH21 * sh.b4_c5 +
                pSH22 * sh.b4_c6 +
                pSH23 * sh.b4_c7 +
                pSH24 * sh.b4_c8;
    return colors;
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn num_sh_coeffs(degree: u32) -> u32 {
    return (degree + 1) * (degree + 1);
}

fn read_coeffs(base_id: ptr<function, u32>) -> vec3f {
    let ret = vec3f(coeffs[*base_id + 0], coeffs[*base_id + 1], coeffs[*base_id + 2]);
    *base_id += 3u;
    return ret;
}

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) compact_gid: vec3u) {
    if compact_gid.x >= num_visible {
        return;
    }

    let global_gid = global_from_compact_gid[compact_gid.x];

    let opac = sigmoid(raw_opacities[global_gid]);

    let sh_degree = uniforms.sh_degree.x;
    let num_coeffs = num_sh_coeffs(sh_degree);
    var base_id = global_gid * num_coeffs * 3;

    var sh = ShCoeffs();
    sh.b0_c0 = read_coeffs(&base_id);

    if sh_degree > 0 {
        sh.b1_c0 = read_coeffs(&base_id);
        sh.b1_c1 = read_coeffs(&base_id);
        sh.b1_c2 = read_coeffs(&base_id);

        if sh_degree > 1 {
            sh.b2_c0 = read_coeffs(&base_id);
            sh.b2_c1 = read_coeffs(&base_id);
            sh.b2_c2 = read_coeffs(&base_id);
            sh.b2_c3 = read_coeffs(&base_id);
            sh.b2_c4 = read_coeffs(&base_id);

            if sh_degree > 2 {
                sh.b3_c0 = read_coeffs(&base_id);
                sh.b3_c1 = read_coeffs(&base_id);
                sh.b3_c2 = read_coeffs(&base_id);
                sh.b3_c3 = read_coeffs(&base_id);
                sh.b3_c4 = read_coeffs(&base_id);
                sh.b3_c5 = read_coeffs(&base_id);
                sh.b3_c6 = read_coeffs(&base_id);

                if sh_degree > 3 {
                    sh.b4_c0 = read_coeffs(&base_id);
                    sh.b4_c1 = read_coeffs(&base_id);
                    sh.b4_c2 = read_coeffs(&base_id);
                    sh.b4_c3 = read_coeffs(&base_id);
                    sh.b4_c4 = read_coeffs(&base_id);
                    sh.b4_c5 = read_coeffs(&base_id);
                    sh.b4_c6 = read_coeffs(&base_id);
                    sh.b4_c7 = read_coeffs(&base_id);
                    sh.b4_c8 = read_coeffs(&base_id);
                }
            }
        }
    }

    let mean = vec3f(means[global_gid * 3 + 0], means[global_gid * 3 + 1], means[global_gid * 3 + 2]);

    let viewdir = normalize(-uniforms.viewmat[3].xyz - mean);

    let color = max(sh_coeffs_to_color(sh_degree, viewdir, sh) + vec3f(0.5), vec3f(0.0));


    // // If this will not be visible in any pixel - bail.
    // let nearest_pix = floor(xy) + 0.5;
    // let vis = opac * helpers::calc_vis(nearest_pix, conic, xy);
    // if vis < 1.0 / 255.0 {
    //     return;
    // }

    // Calculate tbe pixel radius.

    // Original implementation:
    // let b = 0.5 * (cov2d.x + cov2d.z);
    // let v1 = b + sqrt(max(0.1f, b * b - det));
    // let v2 = b - sqrt(max(0.1f, b * b - det));
    // let radius = u32(ceil(3.0 * sqrt(max(0.0, max(v1, v2)))));

    // I think we can do better and derive an exact bound when we hit some eps threshold.
    // Also, we should take into account the opoacity of the gaussian.
    // So, opac * exp(-0.5 * x^T Sigma^-1 x) = eps  (with eps being e.g. 1.0 / 255.0).
    // x^T Sigma^-1 x = -2 * log(eps / opac)
    // Find maximal |x| using quadratic form
    // |x|^2 = c / lambd_min.

    // // Now solve for maximal |r| such that min alpha = 1.0 / 255.0.
    // //
    // // we actually go for 2.0 / 255.0 or so to match the cutoff from gsplat better.
    // // maybe can be more precise here if we don't need 1:1 compat with gsplat anymore.
    let conic = conic_comps[compact_gid.x].xyz;
    let xy = xys[compact_gid.x];
    let radius = helpers::radius_from_conic(conic, opac);

    let tile_minmax = helpers::get_tile_bbox(xy, u32(radius), uniforms.tile_bounds);
    let tile_min = tile_minmax.xy;
    let tile_max = tile_minmax.zw;
    var tile_area = 0u;

    for (var ty = tile_min.y; ty < tile_max.y; ty++) {
        for (var tx = tile_min.x; tx < tile_max.x; tx++) {
            if helpers::can_be_visible(vec2u(tx, ty), xy, conic, opac) {
                tile_area += 1u;
            }
        }
    }

    colors[compact_gid.x] = vec4f(color, opac);
    num_tiles_hit[compact_gid.x] = u32(tile_area);
}