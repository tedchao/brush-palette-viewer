//! Constraint-driven palette + tone-curve optimization for a 3DGS palette
//! representation, with coupled BCD between the L block and the ΔP block.
//!
//! This is a native Rust port of `python/constraint_optimizer.py` (the .py
//! file is kept alongside as the reference implementation). It implements the
//! formulation from our writeup:
//!
//!   E      = E_sp + w_eq (E_img + E_l + E_p)
//!   E_img  = sum_x ||  Σ_{i>=3} W_i(x) ΔP_i  +  δ(x)·(1,1,1)
//!                   - (c*(x) - c(x))  ||²
//!   E_l    = sum_i || S_i ⊙ L_i - C_i ||²       (direct curve-point cons)
//!   E_p    = sum_{i in P_c} || (P_i + ΔP_i) - P̂_i ||²   (direct palette cons)
//!   E_sp   = sum_i sqrt( L_i^T B^T B L_i  +  w_sp ||q_i ΔP_i||² )
//!
//! The image-space residual is kept *unified* (3D) — not orthogonally split
//! into grey-axis vs chromatic — so ΔP can absorb lightness work and keep |δ|
//! small enough to avoid driving the rendered weights negative when the user
//! crosses tetrahedra. BCD couples the two sub-problems through per-iteration
//! target updates:
//!
//!   L-step targets:  ΔL*(x) = (1/3)(c*(x) - c(x))  -  (1/3) Σ W_i(x) (1,1,1)·ΔP_i
//!   P-step targets:  r^P(x) = (c*(x) - c(x))  -  δ(x)·(1,1,1)
//!
//! All internal math is `f64` (matching the NumPy reference); conversion to
//! `f32` happens only at the public API boundary.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use web_time::Instant;

#[derive(Debug, Clone)]
pub struct PixelConstraint {
    pub w_at_pixel: Vec<f32>,
    pub target_rgb: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct PaletteConstraint {
    pub idx: usize,
    pub target: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct CurveConstraint {
    pub idx: usize,
    pub l_x: f32,
    pub l_y: f32,
}

#[derive(Debug, Clone)]
pub struct OptimizerResult {
    pub delta_palette: Vec<f32>,
    pub l_curves: Vec<f32>,
    pub n_iter: u32,
    pub runtime_ms: f64,
}

/// BT.709 luminance weights (perceptual brightness). Sums to 1.
const LUMA_WEIGHTS: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// Perceptual luminance of an RGB color (BT.709 weights).
fn luminance(c: [f64; 3]) -> f64 {
    c[0] * LUMA_WEIGHTS[0] + c[1] * LUMA_WEIGHTS[1] + c[2] * LUMA_WEIGHTS[2]
}

// ─────────────────────────────────────────────────────────────────────────
//  Dense matrix helpers (row-major flat `Vec<f64>`)
// ─────────────────────────────────────────────────────────────────────────

/// out[m×p] = a[m×n] · b[n×p].
fn matmul(a: &[f64], m: usize, n: usize, b: &[f64], p: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(b.len(), n * p);
    let mut out = vec![0.0; m * p];
    for i in 0..m {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..p {
                out[i * p + j] += aik * b[k * p + j];
            }
        }
    }
    out
}

/// Transpose an r×c matrix into c×r.
fn transpose(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = a[i * c + j];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────
//  Operators
// ─────────────────────────────────────────────────────────────────────────

/// Bi-Laplacian operator B^T B with mirroring at endpoints (natural BC).
/// Returns an N×N row-major dense matrix.
fn get_btb(n: usize) -> Vec<f64> {
    // grad_mat: G is (N-1)×N with G[k,k] = -1, G[k,k+1] = +1.
    let nm = n - 1;
    let mut g = vec![0.0; nm * n];
    for k in 0..nm {
        g[k * n + k] = -1.0;
        g[k * n + k + 1] = 1.0;
    }
    // Lap = G^T G  (N×N).
    let gt = transpose(&g, nm, n);
    let mut lap = matmul(&gt, n, nm, &g, n);
    // Mirror at endpoints: zero the first and last rows.
    for c in 0..n {
        lap[c] = 0.0;
        lap[(n - 1) * n + c] = 0.0;
    }
    // BTB = Lap^T Lap.
    let lapt = transpose(&lap, n, n);
    matmul(&lapt, n, n, &lap, n)
}

/// Chromatic-renormalized weights (the tilde-W from the writeup). Suppress
/// black (idx 0) and white (idx 1) by factor `rho`, then renormalize columns
/// to sum to 1.
///
/// `w_at_cons` is (c, K) row-major; the returned tilde-W is (K, c) row-major.
fn renormalize_weights(w_at_cons: &[f64], c: usize, k: usize, rho: f64) -> Vec<f64> {
    let mut wt = vec![0.0; k * c];
    for i in 0..k {
        for nu in 0..c {
            let mut v = w_at_cons[nu * k + i].max(0.0);
            if i == 0 || i == 1 {
                v /= rho;
            }
            wt[i * c + nu] = v;
        }
    }
    for nu in 0..c {
        let mut col_sum = 0.0;
        for i in 0..k {
            col_sum += wt[i * c + nu];
        }
        col_sum = col_sum.max(1e-12);
        for i in 0..k {
            wt[i * c + nu] /= col_sum;
        }
    }
    wt
}

// ─────────────────────────────────────────────────────────────────────────
//  Sparse / dense linear solves (faer)
// ─────────────────────────────────────────────────────────────────────────

/// Solve A x = b for a general (non-symmetric) sparse n×n system via sparse
/// LU. `rows[r]` maps column index → value for row `r`.
fn solve_sparse(n: usize, rows: &[HashMap<usize, f64>], b: &[f64]) -> Result<Vec<f64>> {
    use faer::linalg::solvers::Solve;
    use faer::sparse::{SparseColMat, Triplet};

    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
    for (r, row) in rows.iter().enumerate() {
        for (&col, &val) in row {
            triplets.push(Triplet::new(r, col, val));
        }
    }
    let mat = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets)
        .map_err(|e| anyhow!("L-system sparse build failed: {:?}", e))?;

    let rhs = faer::Mat::<f64>::from_fn(n, 1, |i, _| b[i]);
    let lu = mat
        .sp_lu()
        .map_err(|e| anyhow!("L-system sparse LU failed: {:?}", e))?;
    let sol = lu.solve(&rhs);
    Ok((0..n).map(|i| sol[(i, 0)]).collect())
}

/// Solve H X = RHS for a small dense `kc×kc` system with `ncols` right-hand
/// sides. In the IRLS P-step H is positive-definite for the default penalty
/// `q = 1` (PSD `WtW` plus a strictly positive diagonal), so partial-pivot LU
/// always succeeds — the Python reference's `lstsq` fallback for a singular H
/// is therefore unreachable here and is intentionally not ported.
fn solve_dense(h: &[f64], kc: usize, rhs: &[f64], ncols: usize) -> Vec<f64> {
    use faer::linalg::solvers::Solve;

    if kc == 0 {
        return Vec::new();
    }
    let hm = faer::Mat::<f64>::from_fn(kc, kc, |i, j| h[i * kc + j]);
    let rm = faer::Mat::<f64>::from_fn(kc, ncols, |i, j| rhs[i * ncols + j]);
    let sol = hm.partial_piv_lu().solve(&rm);
    let mut out = vec![0.0; kc * ncols];
    for i in 0..kc {
        for j in 0..ncols {
            out[i * ncols + j] = sol[(i, j)];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────
//  Lightness sub-problem (sparse linear system)
// ─────────────────────────────────────────────────────────────────────────

/// Solve the linear system for tone curves L (N samples per curve, K curves).
///
/// `pl_cons`: image-space lightness constraints `(sample_idx, target_abs_L)`.
/// `cl_cons`: direct curve-point constraints `(flat_idx, value)` where
///            `flat_idx = palette_i * N + sample_idx`.
/// `tilde_w`: (K, c) renormalized weights at image-space constraints.
/// `lum_scale`: (K,) IRLS scale factors d_i (must be > 0).
///
/// Returns L of shape (N, K), row-major: `L[n*K + i]`.
#[allow(clippy::too_many_arguments)]
fn solve_l_system(
    pl_cons: &[(usize, f64)],
    cl_cons: &[(usize, f64)],
    tilde_w: &[f64],
    c: usize,
    n: usize,
    w_eq: f64,
    lum_scale: &[f64],
    k: usize,
    btb: &[f64],
) -> Result<Vec<f64>> {
    let kn = k * n;
    // Row-oriented sparse accumulator (the analogue of scipy's LIL): lets us
    // both `+=` into entries and wholesale-replace pinned rows.
    let mut rows: Vec<HashMap<usize, f64>> = vec![HashMap::new(); kn];
    let mut b = vec![0.0; kn];

    // Smoothness block: D ⊗ B^T B, block-diagonal with K blocks of id_p[i]·BTB.
    for i in 0..k {
        let id_p = 1.0 / lum_scale[i].max(1e-12).sqrt();
        for r in 0..n {
            for col in 0..n {
                let v = btb[r * n + col];
                if v != 0.0 {
                    *rows[i * n + r].entry(i * n + col).or_insert(0.0) += id_p * v;
                }
            }
        }
    }

    // Image-space lightness constraint rows. For each constraint nu at sample
    // row Lidx, over all (i, j) palette pairs add 2·w_eq·tw_i·tw_j to
    // A[(i,Lidx),(j,Lidx)] and 2·w_eq·tw_i·Lt to b[(i,Lidx)].
    for (nu, &(lidx, lt)) in pl_cons.iter().enumerate() {
        for i in 0..k {
            let twi = tilde_w[i * c + nu];
            for j in 0..k {
                let twj = tilde_w[j * c + nu];
                let val = 2.0 * w_eq * twi * twj;
                if val != 0.0 {
                    *rows[i * n + lidx].entry(j * n + lidx).or_insert(0.0) += val;
                }
            }
            b[i * n + lidx] += 2.0 * w_eq * twi * lt;
        }
    }

    // Pin endpoints: zero out rows, set diagonal to 1; b = 0 (start), 1 (end).
    for i in 0..k {
        let z = i * n;
        let o = i * n + (n - 1);
        rows[z].clear();
        rows[z].insert(z, 1.0);
        b[z] = 0.0;
        rows[o].clear();
        rows[o].insert(o, 1.0);
        b[o] = 1.0;
    }

    // Direct curve-point constraints.
    for &(flat_idx, val) in cl_cons {
        rows[flat_idx].clear();
        rows[flat_idx].insert(flat_idx, 1.0);
        b[flat_idx] = val;
    }

    let x = solve_sparse(kn, &rows, &b)?;

    // reshape(N, K, order='F'): L[n, i] = x[i*N + n].
    let mut l = vec![0.0; n * k];
    for i in 0..k {
        for ni in 0..n {
            l[ni * k + i] = x[i * n + ni];
        }
    }
    Ok(l)
}

/// L: (N, K) row-major. Returns (K,) array of L_i^T B^T B L_i.
fn compute_lum_scale(l: &[f64], n: usize, k: usize, btb: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; k];
    for i in 0..k {
        let mut s = 0.0;
        for a in 0..n {
            let la = l[a * k + i];
            if la == 0.0 {
                continue;
            }
            let mut row_dot = 0.0;
            for bb in 0..n {
                row_dot += btb[a * n + bb] * l[bb * k + i];
            }
            s += la * row_dot;
        }
        out[i] = s;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────
//  Palette sub-problem (small QP via IRLS)
// ─────────────────────────────────────────────────────────────────────────

/// Find ΔP_i for chromatic palette colors (i >= 2) by IRLS.
///
/// Black (i=0) and white (i=1) are NOT optimized for image-space matching;
/// however `palette_cons` entries targeting i ∈ {0,1} are enforced as a direct
/// hard pin (ΔP_i ← clip(target) - P_i) outside the IRLS loop.
///
/// `w_at_cons`: (c, K) raw weights. `rp_targets`: (c, 3) coupled residual.
/// `lum_scale`: (K,) k_i values from the current L. `q`: (K,) penalty vector.
///
/// Returns delta_palette of shape (K, 3) row-major.
#[allow(clippy::too_many_arguments)]
fn solve_p_system(
    palette: &[f64],
    k: usize,
    w_at_cons: &[f64],
    c: usize,
    rp_targets: &[f64],
    palette_cons: &[PaletteConstraint],
    lum_scale: &[f64],
    w_sp: f64,
    w_eq: f64,
    q: &[f64],
    irls_iters: usize,
    irls_eps: f64,
) -> Vec<f64> {
    let kc = k - 2;
    let has_img = c > 0;

    // Split palette_cons into achromatic (hard pin) and chromatic (soft).
    let mut achromatic_pins: Vec<(usize, [f64; 3])> = Vec::new();
    let mut chromatic_pins: Vec<(usize, [f64; 3])> = Vec::new();
    for pc in palette_cons {
        let tgt = [
            pc.target[0] as f64,
            pc.target[1] as f64,
            pc.target[2] as f64,
        ];
        if pc.idx < 2 {
            achromatic_pins.push((pc.idx, tgt));
        } else {
            chromatic_pins.push((pc.idx, tgt));
        }
    }

    // IRLS loop over chromatic ΔP only.
    let mut delp_chrom = vec![0.0; kc * 3];

    if kc > 0 {
        // d_i = sqrt(max(lum_scale_chrom, irls_eps)).
        let mut d_i: Vec<f64> = (0..kc).map(|i| lum_scale[i + 2].max(irls_eps).sqrt()).collect();

        // WtW = 2·w_eq·(W_chrom^T W_chrom)  (Kc, Kc);
        // Wtr = 2·w_eq·(W_chrom^T rP_targets)  (Kc, 3).
        let mut wtw = vec![0.0; kc * kc];
        let mut wtr = vec![0.0; kc * 3];
        if has_img {
            for a in 0..kc {
                for bcol in 0..kc {
                    let mut s = 0.0;
                    for nu in 0..c {
                        s += w_at_cons[nu * k + (a + 2)] * w_at_cons[nu * k + (bcol + 2)];
                    }
                    wtw[a * kc + bcol] = 2.0 * w_eq * s;
                }
                for ch in 0..3 {
                    let mut s = 0.0;
                    for nu in 0..c {
                        s += w_at_cons[nu * k + (a + 2)] * rp_targets[nu * 3 + ch];
                    }
                    wtr[a * 3 + ch] = 2.0 * w_eq * s;
                }
            }
        }

        // Chromatic palette-equality penalties (soft, inside IRLS).
        let mut pal_diag = vec![0.0; kc];
        let mut pal_rhs = vec![0.0; kc * 3];
        for &(idx, tgt) in &chromatic_pins {
            let ci = idx - 2;
            pal_diag[ci] = 2.0 * w_eq;
            for ch in 0..3 {
                pal_rhs[ci * 3 + ch] = 2.0 * w_eq * (tgt[ch] - palette[idx * 3 + ch]);
            }
        }

        for _ in 0..irls_iters {
            // sparsity_diag = w_sp · q_chrom² / max(d_i, irls_eps).
            let mut h = wtw.clone();
            for i in 0..kc {
                let qc = q[i + 2];
                let sparsity_diag = w_sp * qc * qc / d_i[i].max(irls_eps);
                h[i * kc + i] += sparsity_diag + pal_diag[i];
            }
            let mut rhs = vec![0.0; kc * 3];
            for idx in 0..kc * 3 {
                rhs[idx] = wtr[idx] + pal_rhs[idx];
            }

            delp_chrom = solve_dense(&h, kc, &rhs, 3);

            // Project to gamut box. The clip is disabled in the reference:
            //   new_pal = P_chrom + delp_chrom
            //   #new_pal = np.clip(new_pal, 0.0, 1.0)
            //   delp_chrom = new_pal - P_chrom
            // With the clip commented out this is a no-op, so delp_chrom is
            // left untouched here.

            // Update d_i ← sqrt(k_i + w_sp ||q_i ΔP_i||²).
            for i in 0..kc {
                let qc = q[i + 2];
                let mut weighted_sq = 0.0;
                for ch in 0..3 {
                    let w = qc * delp_chrom[i * 3 + ch];
                    weighted_sq += w * w;
                }
                let a_i = lum_scale[i + 2] + w_sp * weighted_sq;
                d_i[i] = a_i.max(irls_eps * irls_eps).sqrt();
            }
        }
    }

    // Assemble full (K, 3) ΔP.
    let mut delp_full = vec![0.0; k * 3];
    for i in 0..kc {
        for ch in 0..3 {
            delp_full[(i + 2) * 3 + ch] = delp_chrom[i * 3 + ch];
        }
    }
    // Hard-pin achromatic targets (post IRLS): ΔP_i = clip(target) - P_i.
    for &(idx, tgt) in &achromatic_pins {
        for ch in 0..3 {
            let new_anchor = tgt[ch].clamp(0.0, 1.0);
            delp_full[idx * 3 + ch] = new_anchor - palette[idx * 3 + ch];
        }
    }
    delp_full
}

// ─────────────────────────────────────────────────────────────────────────
//  Coupled target computation (the key BCD coupling step)
// ─────────────────────────────────────────────────────────────────────────

/// Compute the coupled L-step targets:
///   ΔL*(x) = w_L·(c*(x) - c(x)) - w_L·Σ_{i>=3} W_i(x) ΔP_i
/// Returns `(sample_idx, absolute_target_curve_value)` per image constraint.
fn compute_l_targets(
    palette: &[f64],
    k: usize,
    w_at_cons: &[f64],
    c: usize,
    target_colors: &[f64],
    delta_palette: &[f64],
    n: usize,
) -> Vec<(usize, f64)> {
    if c == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(c);
    for nu in 0..c {
        // c_orig = W_at_cons @ palette; dP contribution = W_at_cons @ ΔP.
        let mut c_orig = [0.0; 3];
        let mut dp = [0.0; 3];
        for j in 0..k {
            let w = w_at_cons[nu * k + j];
            for ch in 0..3 {
                c_orig[ch] += w * palette[j * 3 + ch];
                dp[ch] += w * delta_palette[j * 3 + ch];
            }
        }
        let l_hat = luminance(c_orig);
        let dp_lightness = luminance(dp);
        let l_target = luminance([
            target_colors[nu * 3],
            target_colors[nu * 3 + 1],
            target_colors[nu * 3 + 2],
        ]);
        let dl_star = l_target - l_hat - dp_lightness;

        let sample_idx = ((l_hat * (n - 1) as f64).round() as i64).clamp(0, n as i64 - 1) as usize;
        let abs_target = l_hat + dl_star;
        out.push((sample_idx, abs_target));
    }
    out
}

/// Linear interpolation of curve `i` (column of L) at `x ∈ [0,1]`, over a
/// uniform grid of N samples — the np.interp equivalent for our setup.
fn interp_uniform(x: f64, l: &[f64], n: usize, k: usize, i: usize) -> f64 {
    let pos = x * (n - 1) as f64;
    let mut idx = pos.floor() as isize;
    idx = idx.clamp(0, n as isize - 2);
    let idx = idx as usize;
    let frac = pos - idx as f64;
    let lo = l[idx * k + i];
    let hi = l[(idx + 1) * k + i];
    lo * (1.0 - frac) + hi * frac
}

/// Compute the coupled P-step targets:
///   r^P(x) = (c*(x) - c(x)) - δ(x)·(1,1,1)
/// using the current curves L. Returns (c, 3) row-major.
#[allow(clippy::too_many_arguments)]
fn compute_p_targets(
    palette: &[f64],
    k: usize,
    w_at_cons: &[f64],
    c: usize,
    target_colors: &[f64],
    l: &[f64],
    tilde_w: &[f64],
    n: usize,
) -> Vec<f64> {
    if c == 0 {
        return Vec::new();
    }
    let mut rp = vec![0.0; c * 3];
    for nu in 0..c {
        let mut c_orig = [0.0; 3];
        for j in 0..k {
            let w = w_at_cons[nu * k + j];
            for ch in 0..3 {
                c_orig[ch] += w * palette[j * 3 + ch];
            }
        }
        let l_hat = luminance(c_orig);
        let x = l_hat.clamp(0.0, 1.0);

        // δ(x) = Σ_i tilde_W_i(x) [f_i(L̂_0(x)) - L̂_0(x)].
        let mut delta_x = 0.0;
        for i in 0..k {
            let f_i = interp_uniform(x, l, n, k, i);
            delta_x += tilde_w[i * c + nu] * (f_i - l_hat);
        }
        for ch in 0..3 {
            rp[nu * 3 + ch] = (target_colors[nu * 3 + ch] - c_orig[ch]) - delta_x;
        }
    }
    rp
}

// ─────────────────────────────────────────────────────────────────────────
//  Alternating outer loop
// ─────────────────────────────────────────────────────────────────────────

/// Euclidean norm of the difference of two equal-length vectors.
fn norm_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

struct OptimizeOut {
    l: Vec<f64>,  // (N, K) row-major
    dp: Vec<f64>, // (K, 3) row-major
    n_iter: u32,
}

/// Coupled BCD + IRLS solver for `min E_sp + w_eq (E_img + E_l + E_p)` s.t.
/// endpoint pins on curves and the gamut box on chromatic ΔP. The L-block and
/// P-block both contribute to image-space lightness, so plain BCD oscillates;
/// updates are damped by `damping ∈ (0,1]` to break the oscillation.
#[allow(clippy::too_many_arguments)]
fn alternating_optimize(
    palette: &[f64],
    k: usize,
    w_at_cons: &[f64],
    c: usize,
    target_colors: &[f64],
    palette_cons: &[PaletteConstraint],
    curve_cons: &[CurveConstraint],
    n: usize,
) -> Result<OptimizeOut> {
    // Defaults matching the Python `alternating_optimize` signature; only `N`
    // is supplied by the caller.
    let w_eq = 1000.0;
    let w_sp = 0.1;
    let rho = 100.0;
    let max_iter = 10;
    let tol = 1e-5;
    let damping = 0.7;

    let btb = get_btb(n);

    // Direct curve-point constraints (static across iters).
    let cl_cons: Vec<(usize, f64)> = curve_cons
        .iter()
        .map(|cc| {
            let sample = (cc.l_x as f64 * (n - 1) as f64).round() as usize;
            (cc.idx * n + sample, cc.l_y as f64)
        })
        .collect();

    // Renormalized weights for the L-system (static). When there are no image
    // constraints tilde_W is never indexed (every consumer early-returns).
    let tilde_w = if c > 0 {
        renormalize_weights(w_at_cons, c, k, rho)
    } else {
        Vec::new()
    };

    // Init.
    let mut lum_scale = vec![1.0; k];
    let mut delta_palette = vec![0.0; k * 3];
    // L starts as the identity curve: L[n, i] = n / (N - 1).
    let mut l = vec![0.0; n * k];
    for ni in 0..n {
        let v = ni as f64 / (n - 1) as f64;
        for i in 0..k {
            l[ni * k + i] = v;
        }
    }
    // Penalty vector q defaults to ones.
    let q = vec![1.0; k];

    let mut n_iter = 0u32;
    for _ in 0..max_iter {
        n_iter += 1;

        // 1) Coupled L-targets (uses current ΔP).
        let pl_cons = compute_l_targets(palette, k, w_at_cons, c, target_colors, &delta_palette, n);

        // 2) L-system with damped update. Strict IRLS: fold the current ΔP
        // sparsity contribution into lum_scale before the solve.
        for i in 0..k {
            let mut weighted_sq = 0.0;
            for ch in 0..3 {
                let w = q[i] * delta_palette[i * 3 + ch];
                weighted_sq += w * w;
            }
            lum_scale[i] = (lum_scale[i] + w_sp * weighted_sq).max(1e-12);
        }

        let l_new = solve_l_system(&pl_cons, &cl_cons, &tilde_w, c, n, w_eq, &lum_scale, k, &btb)?;
        for idx in 0..n * k {
            l[idx] = (1.0 - damping) * l[idx] + damping * l_new[idx];
        }
        let new_lum_scale = compute_lum_scale(&l, n, k, &btb);

        // 3) Coupled P-targets (uses freshly updated L).
        let rp_targets = compute_p_targets(palette, k, w_at_cons, c, target_colors, &l, &tilde_w, n);

        // 4) P-system with damped update.
        let dp_new = solve_p_system(
            palette,
            k,
            w_at_cons,
            c,
            &rp_targets,
            palette_cons,
            &new_lum_scale,
            w_sp,
            w_eq,
            &q,
            8,
            1e-3,
        );
        let mut new_delta = vec![0.0; k * 3];
        for idx in 0..k * 3 {
            new_delta[idx] = (1.0 - damping) * delta_palette[idx] + damping * dp_new[idx];
        }

        let d_ls = norm_diff(&new_lum_scale, &lum_scale);
        let d_dp = norm_diff(&new_delta, &delta_palette);
        lum_scale = new_lum_scale;
        delta_palette = new_delta;

        if d_ls < tol && d_dp < tol {
            break;
        }
    }

    Ok(OptimizeOut {
        l,
        dp: delta_palette,
        n_iter,
    })
}

// ─────────────────────────────────────────────────────────────────────────
//  Public entry point
// ─────────────────────────────────────────────────────────────────────────

pub fn run_optimizer(
    palette: &[f32],
    k_full: usize,
    pixel_cons: &[PixelConstraint],
    palette_cons: &[PaletteConstraint],
    curve_cons: &[CurveConstraint],
    n_curve_samples: usize,
) -> Result<OptimizerResult> {
    assert_eq!(palette.len(), k_full * 3);
    let t0 = Instant::now();

    let palette_f64: Vec<f64> = palette.iter().map(|&x| x as f64).collect();

    // W_at_cons: (c, K) row-major; target_colors: (c, 3) row-major.
    let c = pixel_cons.len();
    let mut w_at_cons = vec![0.0f64; c * k_full];
    let mut target_colors = vec![0.0f64; c * 3];
    for (nu, pc) in pixel_cons.iter().enumerate() {
        assert_eq!(
            pc.w_at_pixel.len(),
            k_full,
            "pixel constraint weight row must have length k_full"
        );
        for j in 0..k_full {
            w_at_cons[nu * k_full + j] = pc.w_at_pixel[j] as f64;
        }
        for ch in 0..3 {
            target_colors[nu * 3 + ch] = pc.target_rgb[ch] as f64;
        }
    }

    let out = alternating_optimize(
        &palette_f64,
        k_full,
        &w_at_cons,
        c,
        &target_colors,
        palette_cons,
        curve_cons,
        n_curve_samples,
    )?;

    // dP: (K, 3) flat row-major.
    let delta_palette: Vec<f32> = out.dp.iter().map(|&x| x as f32).collect();

    // L: (N, K) → flat col-major (l_curves[k*N + n], as the shader expects).
    let n = n_curve_samples;
    let mut l_curves = Vec::with_capacity(n * k_full);
    for k in 0..k_full {
        for ni in 0..n {
            l_curves.push(out.l[ni * k_full + k] as f32);
        }
    }

    Ok(OptimizerResult {
        delta_palette,
        l_curves,
        n_iter: out.n_iter,
        runtime_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_no_constraints() {
        let palette: Vec<f32> = vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let result = run_optimizer(&palette, 4, &[], &[], &[], 100).unwrap();
        assert_eq!(result.delta_palette.len(), 4 * 3);
        assert_eq!(result.l_curves.len(), 100 * 4);
        for v in &result.delta_palette {
            assert!(v.abs() < 1e-5);
        }
        println!(
            "Optimizer ran in {:.1}ms, n_iter={}",
            result.runtime_ms, result.n_iter
        );
    }

    #[test]
    fn palette_constraint_shifts_dp() {
        // K=4, pin chromatic palette [2] (red) → green
        let palette: Vec<f32> = vec![
            0.0, 0.0, 0.0, // black
            1.0, 1.0, 1.0, // white
            1.0, 0.0, 0.0, // red (chromatic)
            0.0, 0.0, 1.0, // blue (chromatic)
        ];
        let palette_cons = vec![PaletteConstraint {
            idx: 2,
            target: [0.0, 1.0, 0.0],
        }];
        let result = run_optimizer(&palette, 4, &[], &palette_cons, &[], 100).unwrap();

        // ΔP for index 2 should push red toward green
        let dp_idx2 = &result.delta_palette[6..9];
        println!("dP[2] = {:?}", dp_idx2);
        // Expected roughly: [-1, +1, 0]
        assert!(dp_idx2[0] < -0.1, "R component should decrease");
        assert!(dp_idx2[1] > 0.1, "G component should increase");

        // ΔP for other chromatic index should be near zero (sparsity)
        let dp_idx3 = &result.delta_palette[9..12];
        println!("dP[3] = {:?}", dp_idx3);
        let norm3: f32 = dp_idx3.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm3 < 0.05, "dP[3] should be near zero (sparsity)");

        println!("Optimizer ran in {:.1}ms", result.runtime_ms);
    }
}
