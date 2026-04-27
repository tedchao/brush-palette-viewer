// project_visible_weight.wgsl
//
// Per-Gaussian: project to image, evaluate K_full weights via SH (DC + low-rank
// rest-band reconstruction), pack into ProjectedWeightSplat.
//
// This mirrors the FORWARD path of the CUDA `sh2Weight_fast` kernel exactly
// (same SH constants, same g = lm*K + k indexing, same row/col splits) but
// uses Sloan-style SH evaluation for performance. The two are mathematically
// equivalent.
//
// Inputs per Gaussian:
//   transforms[i]   : 10 f32s = mean(3) + quat(4) + log_scale(3)   (reused from Brush vanilla)
//   raw_opacities[i]: 1  f32                                         (reused)
//   low_shs_w[i]    : K_full f32s — DC band weights
//   high_shs_a[i]   : P f32s      — rank-1 factor A
//   high_shs_b[i]   : Q f32s      — rank-1 factor B
//
// Output per visible Gaussian:
//   projected[i]    : ProjectedWeightSplat (xy, conic, opacity, weights[MAX_K_FULL], depth)
//
// Static limits:
//   MAX_K_FULL = 8 (palette size, padded; runtime k_full is in uniforms)
//   MAX_REST   = 15 (SH rest-band count for degree 3)

#import helpers;

// Static caps. Loops are bounded by these; runtime values are in uniforms.
const MAX_K_FULL: u32 = 8u;
const MAX_REST:   u32 = 15u;

// Standard SH normalization, same as your CUDA SH_C0_0.
const SH_C0: f32 = 0.2820947917738781;

// Per-splat output struct. xy/conic/opac mirror Brush's ProjectedSplat;
// the weights[] array replaces RGB color. We store depth too so the
// downstream rasterizer can sort if needed. Total: 8 + 12 + 4 + 32 + 4 = 60 bytes.
//
// Layout choice: weights as fixed array<f32, MAX_K_FULL>. Tail entries (k >= k_full)
// are written but ignored by the rasterizer.
struct ProjectedWeightSplat {
  xy_x:    f32,
  xy_y:    f32,
  conic_x: f32,
  conic_y: f32,
  conic_z: f32,
  opacity: f32,
  depth:   f32,
  pad:     f32,
  w0: f32, w1: f32, w2: f32, w3: f32,
  w4: f32, w5: f32, w6: f32, w7: f32,
}

// Uniforms: Brush's ProjectUniforms PLUS our palette-specific knobs.
// We keep the layout compatible with Brush's by extending at the end.
struct PaletteProjectUniforms {
    viewmat:         mat4x4f,
    focal:           vec2f,
    img_size:        vec2u,
    tile_bounds:     vec2u,
    pixel_center:    vec2f,
    camera_position: vec4f,

    sh_degree:       u32,
    total_splats:    u32,
    num_visible:     u32,

    k_full:          u32,
    p_factor:        u32,
    q_factor:        u32,

    pad_a:           u32,
    pad_b:           u32,
}

@group(0) @binding(0) var<storage, read>       transforms              : array<f32>;
@group(0) @binding(1) var<storage, read>       raw_opacities           : array<f32>;
@group(0) @binding(2) var<storage, read>       global_from_compact_gid : array<u32>;
@group(0) @binding(3) var<storage, read>       low_shs_w               : array<f32>;
@group(0) @binding(4) var<storage, read>       high_shs_a              : array<f32>;
@group(0) @binding(5) var<storage, read>       high_shs_b              : array<f32>;
@group(0) @binding(6) var<storage, read_write> projected               : array<ProjectedWeightSplat>;
@group(0) @binding(7) var<storage, read>       uniforms                : PaletteProjectUniforms;

const WG_SIZE: u32 = 256u;

@compute
@workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id)             wid: vec3u,
    @builtin(num_workgroups)           num_wgs: vec3u,
    @builtin(local_invocation_index)   lid: u32,
) {
    let compact_gid = helpers::get_global_id(wid, num_wgs, lid, WG_SIZE);
    
    // Defensive: force all storage bindings to remain live in the layout.
    let _live = transforms[0] + raw_opacities[0]
              + low_shs_w[0] + high_shs_a[0] + high_shs_b[0]
              + f32(global_from_compact_gid[0])
              + f32(uniforms.num_visible);
    if _live > 1e30 {
        // Unreachable in practice; keeps naga from eliminating the reads.
        projected[0].xy_x = _live;
    }
    
    if compact_gid >= uniforms.num_visible {
        return;
    }
    let global_gid = global_from_compact_gid[compact_gid];

    // ── Geometry: same as Brush's project_visible ─────────────────────────────
    let base = global_gid * 10u;
    let mean  = vec3f(transforms[base],
                      transforms[base + 1u],
                      transforms[base + 2u]);
    let quat  = normalize(vec4f(transforms[base + 3u],
                                transforms[base + 4u],
                                transforms[base + 5u],
                                transforms[base + 6u]));
    let scale = exp(vec3f(transforms[base + 7u],
                          transforms[base + 8u],
                          transforms[base + 9u]));
    var opac = helpers::sigmoid(raw_opacities[global_gid]);

    let viewmat = uniforms.viewmat;
    let R = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let mean_c = R * mean + viewmat[3].xyz;

    var cov2d = helpers::calc_cov2d(scale, quat, mean_c, uniforms.focal,
                                    uniforms.img_size, uniforms.pixel_center, viewmat);
    opac *= helpers::compensate_cov2d(&cov2d);

    let conic  = helpers::inverse(cov2d);
    let mean2d = uniforms.focal * mean_c.xy * (1.0 / mean_c.z) + uniforms.pixel_center;

    // ── View direction ────────────────────────────────────────────────────────
    let viewdir = normalize(mean - uniforms.camera_position.xyz);
    let x = viewdir.x;
    let y = viewdir.y;
    let z = viewdir.z;

    // ── Sloan-style SH basis evaluation, scalar (no per-channel multiply) ─────
    // pSH[lm] for lm in 1..=MAX_REST  matches your CUDA's Ylm[lm-1].
    //
    // Verified against your constants:
    //   Ylm[0] = SH_C1_0 * y = -0.4886*y     pSH1 = fTmp0A * (-y) = -0.4886*y  ✓
    //   ...etc through all 15 bands
    var pSH: array<f32, 16>; // 1-indexed, pSH[0] unused
    pSH[0] = 0.0;

    let fTmp0A = 0.4886025119029199;
    pSH[1] = -y * fTmp0A;
    pSH[2] =  z * fTmp0A;
    pSH[3] = -x * fTmp0A;

    let z2  = z * z;
    let fTmp0B = -1.0925484305920792 * z;
    let fTmp1A = 0.5462742152960396;
    let fC1 = x * x - y * y;
    let fS1 = 2.0 * x * y;
    pSH[4] = fTmp1A * fS1;
    pSH[5] = fTmp0B * y;
    pSH[6] = 0.9461746957575601 * z2 - 0.3153915652525201;
    pSH[7] = fTmp0B * x;
    pSH[8] = fTmp1A * fC1;

    let fTmp0C = -2.285228997322329 * z2 + 0.4570457994644658;
    let fTmp1B =  1.445305721320277 * z;
    let fTmp2A = -0.5900435899266435;
    let fC2 = x * fC1 - y * fS1;
    let fS2 = x * fS1 + y * fC1;
    pSH[9]  = fTmp2A * fS2;
    pSH[10] = fTmp1B * fS1;
    pSH[11] = fTmp0C * y;
    pSH[12] = z * (1.865881662950577 * z2 - 1.119528997770346);
    pSH[13] = fTmp0C * x;
    pSH[14] = fTmp1B * fC1;
    pSH[15] = fTmp2A * fC2;

    // ── Compute K_full weights ────────────────────────────────────────────────
    // weight[k] = SH_C0 * low[k] + sum_lm A[g/Q] * B[g%Q] * pSH[lm+1]
    // where g = lm*K + k, lm in 0..L (L = num_sh - 1, here capped at MAX_REST).
    let K = uniforms.k_full;
    let Q = uniforms.q_factor;
    let L = uniforms.sh_degree * (uniforms.sh_degree + 2u); // 1*3=3, 2*8=8(no), this gives wrong:
    // Actually L = (sh_degree+1)^2 - 1 = num_sh - 1.
    // We trust the host to set sh_degree consistently with the buffer sizes.
    // For SH degree 3, L = 15.
    let L_eff = (uniforms.sh_degree + 1u) * (uniforms.sh_degree + 1u) - 1u;

    // Read low_shs_w into local register-friendly storage.
    let low_base = global_gid * K;   // K = uniforms.k_full, set above
    var weights: array<f32, 8>; // MAX_K_FULL
    for (var k: u32 = 0u; k < MAX_K_FULL; k = k + 1u) {
        if k >= K {
            weights[k] = 0.0;
        } else {
            weights[k] = SH_C0 * low_shs_w[low_base + k];
        }
    }

    // Low-rank rest-band reconstruction.
    // A is (N, P), B is (N, Q). For each (lm, k), index into A[row], B[col]
    // where g = lm*K + k, row = g/Q, col = g%Q.
    let p_factor = uniforms.p_factor;
    let a_base = global_gid * p_factor;
    let b_base = global_gid * Q;

    for (var lm: u32 = 0u; lm < MAX_REST; lm = lm + 1u) {
        if lm >= L_eff {
            break;
        }
        let y_lm = pSH[lm + 1u];
        for (var k: u32 = 0u; k < MAX_K_FULL; k = k + 1u) {
            if k >= K {
                break;
            }
            let g   = lm * K + k;
            let row = g / Q;
            let col = g % Q;
            weights[k] = weights[k] + high_shs_a[a_base + row] * high_shs_b[b_base + col] * y_lm;
        }
    }

    // ── Pack ProjectedWeightSplat ─────────────────────────────────────────────
    let conic_packed = vec3f(conic[0][0], conic[0][1], conic[1][1]);
    var ps: ProjectedWeightSplat;
    ps.xy_x    = mean2d.x;
    ps.xy_y    = mean2d.y;
    ps.conic_x = conic_packed.x;
    ps.conic_y = conic_packed.y;
    ps.conic_z = conic_packed.z;
    ps.opacity = opac;
    ps.depth   = mean_c.z;
    ps.pad     = 0.0;
    ps.w0 = weights[0];
    ps.w1 = weights[1];
    ps.w2 = weights[2];
    ps.w3 = weights[3];
    ps.w4 = weights[4];
    ps.w5 = weights[5];
    ps.w6 = weights[6];
    ps.w7 = weights[7];
    projected[compact_gid] = ps;
}
