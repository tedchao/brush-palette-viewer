"""
constraint_optimize.py
======================
Constraint-driven palette + tone-curve optimization for a 3DGS palette
representation, with coupled BCD between the L block and the ΔP block.

Implements the formulation from our writeup:

  E      = E_sp + w_eq (E_img + E_l + E_p)
  E_img  = sum_x ||  Σ_{i>=3} W_i(x) ΔP_i  +  δ(x)·(1,1,1)
                  - (c*(x) - c(x))  ||²
  E_l    = sum_i || S_i ⊙ L_i - C_i ||²       (direct curve-point cons)
  E_p    = sum_{i in P_c} || (P_i + ΔP_i) - P̂_i ||²   (direct palette cons)
  E_sp   = sum_i sqrt( L_i^T B^T B L_i  +  w_sp ||q_i ΔP_i||² )

The image-space residual is kept *unified* (3D) — not orthogonally split
into grey-axis vs chromatic — so ΔP can absorb lightness work and keep |δ|
small enough to avoid driving the rendered weights negative when the user
crosses tetrahedra. BCD couples the two sub-problems through per-iteration
target updates:

  L-step targets:  ΔL*(x) = (1/3)(c*(x) - c(x))  -  (1/3) Σ W_i(x) (1,1,1)·ΔP_i
                   (lightness shift δ must absorb after ΔP did its share)
  P-step targets:  r^P(x) = (c*(x) - c(x))  -  δ(x)·(1,1,1)
                   (full 3D residual ΔP must absorb after δ did its share)
"""

import time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import minimize, Bounds


# BT.709 luminance weights (perceptual brightness). Sums to 1.
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])


def luminance(c):
    """
    Perceptual luminance of an RGB color. c may be (..., 3); returns (...,).
    Uses BT.709 weights (0.2126, 0.7152, 0.0722).
    """
    return c @ LUMA_WEIGHTS


# ─────────────────────────────────────────────────────────────────────────
#  Operators and caches
# ─────────────────────────────────────────────────────────────────────────

def grad_mat(n):
    G = np.zeros((n - 1, n))
    for k in range(n - 1):
        G[k, k + 1] = 1.0
        G[k, k]     = -1.0
    return G


_BTB_CACHE = {}
def get_BTB(N):
    """Bi-Laplacian operator B^T B with mirroring at endpoints."""
    if N in _BTB_CACHE:
        return _BTB_CACHE[N]
    G   = grad_mat(N)
    Lap = G.T @ G
    Lap[0, :]  = 0      # mirror at endpoints (natural BC)
    Lap[-1, :] = 0
    BTB = Lap.T @ Lap
    _BTB_CACHE[N] = BTB
    return BTB


def renormalize_weights(W, rho=100.0):
    """
    Chromatic-renormalized weights (the tilde-W from the writeup).
    Suppress black (idx 0) and white (idx 1) by factor rho, then
    renormalize columns to sum to 1.

    W:   (K, c) raw weights at c constraints.
    rho: suppression factor for achromatic colors.
    Returns: (K, c) tilde-W.
    """
    Wt = np.maximum(W.copy(), 0.0)
    Wt[0, :] /= rho
    Wt[1, :] /= rho
    col_sum = np.maximum(Wt.sum(axis=0, keepdims=True), 1e-12)
    return Wt / col_sum


# ─────────────────────────────────────────────────────────────────────────
#  Lightness sub-problem (linear system)
# ─────────────────────────────────────────────────────────────────────────

def _kron_A_I(A, n):
    """Equivalent to np.kron(A, np.eye(n)). Built via 4D indexing trick."""
    m, k = A.shape
    out  = np.zeros((m, n, k, n), dtype=A.dtype)
    r    = np.arange(n)
    out[:, r, :, r] = A          # CC's formulation (works due to advanced indexing)
    return out.reshape(m * n, k * n)


def _kron_diag_A(d, A):
    """Equivalent to np.kron(np.diag(d), A)."""
    n    = len(d)
    m, k = A.shape
    out  = np.zeros((n, m, n, k), dtype=A.dtype)
    r    = np.arange(n)
    out[r, :, r, :] = d[:, None, None] * A
    return out.reshape(n * m, n * k)


def _BTB_block_diag_sparse(d, BTB):
    """Sparse equivalent of np.kron(np.diag(d), BTB) — block-diagonal with
    K blocks of d[i] * BTB. Returns CSC."""
    K = len(d)
    blocks = [d[i] * BTB for i in range(K)]
    return scipy.sparse.block_diag(blocks, format='csc')


_BTB_SPARSE_CACHE = {}
def _get_BTB_sparse(N):
    if N in _BTB_SPARSE_CACHE:
        return _BTB_SPARSE_CACHE[N]
    BTB_sp = scipy.sparse.csc_matrix(get_BTB(N))
    _BTB_SPARSE_CACHE[N] = BTB_sp
    return BTB_sp


def solve_L_system(pl_cons, cl_cons, tilde_W, N, w_eq, lum_scale):
    """
    Solve linear system for tone curves L (N samples per curve, K curves).
    Builds A directly as sparse (CSC) for speed.

    pl_cons:   list of (sample_idx, target_abs_L) for image-space lightness
               constraints. target_abs_L is the absolute target value of
               the curve at sample_idx — i.e., L̂_0(x) + ΔL*(x).
    cl_cons:   list of (flat_idx, target_val) for direct curve-point
               constraints (flat_idx = palette_i * N + sample_idx).
    tilde_W:   (K, c) renormalized weights at image-space constraints.
    N:         curve sample count.
    w_eq:      constraint weight (large, e.g. 100).
    lum_scale: (K,) IRLS scale factors d_i (must be > 0).

    Returns: L of shape (N, K).
    """
    K   = tilde_W.shape[0]
    BTB = _get_BTB_sparse(N)

    # Smoothness block: D ⊗ B^T B, built directly as sparse block-diag.
    id_p = 1.0 / np.sqrt(np.maximum(lum_scale, 1e-12))
    A    = _BTB_block_diag_sparse(id_p, BTB).tolil()    # LIL for row edits

    # Image-space lightness constraint rows. The constraint block
    # has rank ≤ c·s but in practice is small; build via sparse outer.
    if len(pl_cons) == 0:
        b = np.zeros(K * N)
    else:
        # For each constraint nu at sample row Lidx:
        # the constraint contributes (over all (i, j) palette pairs)
        # 2 w_eq tilde_W[i, nu] tilde_W[j, nu] at row (i, Lidx), col (j, Lidx),
        # and 2 w_eq tilde_W[i, nu] Ltarget at row (i, Lidx) for b.
        rows, cols, data = [], [], []
        b = np.zeros(K * N)
        for nu, (Lidx, Lt) in enumerate(pl_cons):
            tw = tilde_W[:, nu]                                # (K,)
            outer = 2.0 * w_eq * np.outer(tw, tw)              # (K, K)
            for i in range(K):
                for j in range(K):
                    if outer[i, j] != 0.0:
                        rows.append(i * N + Lidx)
                        cols.append(j * N + Lidx)
                        data.append(outer[i, j])
                b[i * N + Lidx] += 2.0 * w_eq * tw[i] * Lt
        cons_sp = scipy.sparse.coo_matrix(
            (data, (rows, cols)), shape=(K * N, K * N)
        )
        A = (A + cons_sp).tolil()

    # Pin endpoints: zero out rows, set diagonal to 1.
    zero_idx = [i * N for i in range(K)]
    one_idx  = [i * N + (N - 1) for i in range(K)]
    for idx in zero_idx + one_idx:
        A.rows[idx] = [idx]
        A.data[idx] = [1.0]
    for idx in zero_idx: b[idx] = 0.0
    for idx in one_idx:  b[idx] = 1.0

    # Direct curve-point constraints.
    if len(cl_cons) > 0:
        for (flat_idx, val) in cl_cons:
            A.rows[flat_idx] = [flat_idx]
            A.data[flat_idx] = [1.0]
            b[flat_idx] = val

    A_csc = A.tocsc()
    x_l   = scipy.sparse.linalg.spsolve(A_csc, b)
    return x_l.reshape(N, K, order='F')


def compute_lum_scale(L, N):
    """L: (N, K). Returns (K,) array of L_i^T B^T B L_i."""
    BTB = get_BTB(N)
    return np.array([L[:, i].T @ BTB @ L[:, i] for i in range(L.shape[1])])


# ─────────────────────────────────────────────────────────────────────────
#  Palette sub-problem (small QP via SLSQP)
# ─────────────────────────────────────────────────────────────────────────

def solve_P_system(palette, W_at_cons, rP_targets, palette_cons,
                    lum_scale, w_sp=0.001, w_eq=100.0, q=None,
                    color_tol=5e-3, irls_iters=8, irls_eps=1e-3):
    """
    Find ΔP_i for chromatic palette colors (i >= 2) by IRLS.

        min  Σ_i sqrt( k_i + w_sp ||q_i ΔP_i||² )                (sparsity)
        s.t. Σ_i W_i(x) ΔP_i  ≈  rP_targets[x]                    (image-space)
             P_i + ΔP_i ∈ [0,1]^3                                 ∀ chromatic i

    Black (i=0) and white (i=1) are NOT optimized for image-space matching
    (this avoids the bilinear cross-term δ · ΔP_{1,2} and keeps them as
    anchors). However, palette-equality constraints `palette_cons` may
    still target i ∈ {0, 1}; those are enforced as a *direct hard pin*
    (ΔP_i ← target - P_i) outside the IRLS loop.

    IRLS rewrite. The L21 sparsity term is non-smooth at ΔP_i = 0, which
    makes general solvers unable to localize edits to a single palette
    color. We use the variational identity √a = min_d a/(2d) + d/2 to
    reweight:

        min Σ_i (k_i + w_sp ||q_i ΔP_i||²) / (2 d_i)  +  const

    With {d_i} frozen, the objective is a quadratic in ΔP. We solve it
    iteratively, updating d_i ← √(k_i + w_sp ||q_i ΔP_i||²) each round.
    Image-space and palette-equality (chromatic) constraints are enforced
    as soft quadratic penalties weighted by w_eq, giving a closed-form
    linear solve per IRLS iteration. The gamut box is enforced after each
    iteration by clipping.

    palette:      (K, 3) original palette. i=0 black, i=1 white,
                  i>=2 chromatic.
    W_at_cons:    (c, K) raw weights at image-space constraints.
    rP_targets:   (c, 3) coupled target residual (full 3D).
    palette_cons: list of (i, target_RGB_3vec). i may be in [0, K).
                  i ∈ {0, 1} → hard pin (no optimization).
                  i >= 2     → soft penalty inside IRLS.
    lum_scale:    (K,) k_i values from current L.
    w_sp:         relative weight of palette term in sparsity objective.
    w_eq:         weight on soft image-space and palette losses.
    q:            (K,) penalty vector (default ones).
    color_tol:    unused; kept for API compatibility.
    irls_iters:   number of IRLS rounds (default 8).
    irls_eps:     floor on d_i to avoid division by zero.

    Returns: delta_palette of shape (K, 3). Rows 0,1 are zero unless a
             palette_cons entry directly pins them.
    """
    K  = palette.shape[0]
    Kc = K - 2

    if q is None:
        q = np.ones(K)
    q_chrom         = q[2:]                          # (Kc,)
    lum_scale_chrom = lum_scale[2:]                  # (Kc,)
    P_chrom         = palette[2:]                    # (Kc, 3)

    has_img = W_at_cons.shape[0] > 0
    W_chrom = W_at_cons[:, 2:]                       # (c, Kc)

    # ── Split palette_cons into achromatic (hard pin) and chromatic (soft) ─
    achromatic_pins = []     # list of (i, target_rgb)  for i in {0, 1}
    chromatic_pins  = []     # list of (i, target_rgb)  for i >= 2
    for (idx, tgt) in palette_cons:
        if idx < 2:
            achromatic_pins.append((idx, np.asarray(tgt, dtype=np.float64)))
        else:
            chromatic_pins.append((idx, np.asarray(tgt, dtype=np.float64)))

    # ── Chromatic palette-equality bookkeeping (soft, inside IRLS) ────────
    has_pal_chrom = len(chromatic_pins) > 0
    if has_pal_chrom:
        ind_chrom   = np.array([pc[0] - 2 for pc in chromatic_pins], dtype=int)
        pal_targets = np.array([pc[1]     for pc in chromatic_pins])  # (m, 3)

    # ── IRLS loop over chromatic ΔP only ──────────────────────────────────
    delp_chrom = np.zeros((Kc, 3))
    d_i        = np.sqrt(np.maximum(lum_scale_chrom, irls_eps))

    if has_img:
        WtW = 2.0 * w_eq * (W_chrom.T @ W_chrom)             # (Kc, Kc)
        Wtr = 2.0 * w_eq * (W_chrom.T @ rP_targets)          # (Kc, 3)
    else:
        WtW = np.zeros((Kc, Kc))
        Wtr = np.zeros((Kc, 3))

    pal_diag = np.zeros(Kc)
    pal_rhs  = np.zeros((Kc, 3))
    if has_pal_chrom:
        pal_diag[ind_chrom] = 2.0 * w_eq
        pal_rhs[ind_chrom]  = 2.0 * w_eq * (pal_targets - P_chrom[ind_chrom])

    for it in range(irls_iters):
        sparsity_diag = w_sp * (q_chrom ** 2) / np.maximum(d_i, irls_eps)

        H = WtW.copy()
        H[np.arange(Kc), np.arange(Kc)] += sparsity_diag + pal_diag
        rhs = Wtr + pal_rhs

        try:
            delp_chrom = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            delp_chrom = np.linalg.lstsq(H, rhs, rcond=None)[0]

        # Project to gamut box.
        new_pal = P_chrom + delp_chrom
        new_pal = np.clip(new_pal, 0.0, 1.0)
        delp_chrom = new_pal - P_chrom

        # Update d_i.
        weighted = q_chrom[:, None] * delp_chrom
        a_i = lum_scale_chrom + w_sp * np.sum(weighted ** 2, axis=1)
        d_i = np.sqrt(np.maximum(a_i, irls_eps ** 2))

    # ── Assemble full (K, 3) ΔP ────────────────────────────────────────
    delp_full = np.zeros((K, 3))
    delp_full[2:] = delp_chrom

    # Hard-pin achromatic targets (post IRLS — they don't participate in
    # the image-space fit but the user may still want to recolor B/W).
    for (idx, tgt) in achromatic_pins:
        # ΔP_i = target - P_i, then clipped so P_i + ΔP_i ∈ [0,1]^3.
        new_anchor = np.clip(tgt, 0.0, 1.0)
        delp_full[idx] = new_anchor - palette[idx]

    return delp_full


# ─────────────────────────────────────────────────────────────────────────
#  Coupled target computation (the key BCD coupling step)
# ─────────────────────────────────────────────────────────────────────────

def compute_L_targets(palette, W_at_cons, target_colors, delta_palette, N):
    """
    Compute the coupled L-step targets:
        ΔL*(x) = w_L^T (c*(x) - c(x)) - w_L^T Σ_{i>=3} W_i(x) ΔP_i
    where w_L = (0.2126, 0.7152, 0.0722) is BT.709 luminance.
    Returns pl_cons format expected by solve_L_system: each entry is
    (sample_idx, absolute_target_curve_value).

    palette:       (K, 3)
    W_at_cons:     (c, K)
    target_colors: (c, 3)
    delta_palette: (K, 3) current ΔP (rows 0,1 zero)
    N:             curve sample count

    Returns: list of (sample_idx, L̂_0(x) + ΔL*(x)).
    """
    if W_at_cons.shape[0] == 0:
        return []
    c_orig = W_at_cons @ palette                          # (c, 3)
    L_hat  = luminance(c_orig)                             # (c,)

    # Luminance already provided by ΔP at each pixel.
    dP_lightness = luminance(W_at_cons @ delta_palette)    # (c,)

    L_target = luminance(target_colors)                    # (c,)
    DL_star  = L_target - L_hat - dP_lightness             # (c,)

    pl_cons = []
    for nu in range(W_at_cons.shape[0]):
        sample_idx = int(np.clip(round(L_hat[nu] * (N - 1)), 0, N - 1))
        abs_target = float(L_hat[nu] + DL_star[nu])       # = L_target - dP_lightness
        pl_cons.append((sample_idx, abs_target))
    return pl_cons


def compute_P_targets(palette, W_at_cons, target_colors, L, tilde_W, N):
    """
    Compute the coupled P-step targets:
        r^P(x) = (c*(x) - c(x)) - δ(x)·(1,1,1)
    using the current curves L. Returns (c, 3).

    palette:       (K, 3)
    W_at_cons:     (c, K)  raw weights
    target_colors: (c, 3)
    L:             (N, K)  current curves
    tilde_W:       (K, c)  renormalized weights (used by δ)
    N:             curve sample count
    """
    if W_at_cons.shape[0] == 0:
        return np.zeros((0, 3))

    c_orig = W_at_cons @ palette                          # (c, 3)
    L_hat  = luminance(c_orig)                             # (c,) — BT.709 luminance
    grid   = np.linspace(0.0, 1.0, N)

    # Evaluate each curve at each constraint's L̂_0:  f_i(L̂_0)  →  (K, c)
    f_vals = np.empty((tilde_W.shape[0], W_at_cons.shape[0]))
    for i in range(tilde_W.shape[0]):
        f_vals[i] = np.interp(np.clip(L_hat, 0, 1), grid, L[:, i])

    # δ(x) = Σ_i tilde_W_i(x) [f_i(L̂_0(x)) - L̂_0(x)]
    delta_x = (tilde_W * (f_vals - L_hat[None, :])).sum(axis=0)   # (c,)

    rP = (target_colors - c_orig) - delta_x[:, None] * np.array([1.0, 1.0, 1.0])
    return rP


# ─────────────────────────────────────────────────────────────────────────
#  Alternating outer loop
# ─────────────────────────────────────────────────────────────────────────

def alternating_optimize(palette, W_at_cons, target_colors, palette_cons,
                          curve_cons, N=100, w_eq=10000.0, w_sp=0.001,
                          rho=100.0, q=None, max_iter=10, tol=1e-5,
                          damping=0.7, verbose=False):
    """
    Coupled BCD + IRLS solver for
        min E_sp + w_eq (E_img + E_l + E_p)
    s.t. endpoint pins on curves and gamut box on chromatic ΔP.

    The L-block (curves, δ) and P-block (ΔP) both contribute to image-space
    lightness, which makes plain BCD oscillate (each block keeps shifting
    work onto the other). We damp the updates by `damping` ∈ (0, 1] to
    break the oscillation; `damping=0.5` gives clean geometric convergence
    in ~10–20 outer iterations.

    palette:          (K, 3) in [0,1]. i=0 black, i=1 white, i>=2 chromatic.
    W_at_cons:        (c, K) raw weights at image-space constraints.
    target_colors:    (c, 3) target RGB at each image-space constraint.
    palette_cons:     list of (i, target_RGB_3vec). i may be in [0, K):
                      i ∈ {0, 1} → hard pin (no optimization, just direct
                                   assignment ΔP_i = target - P_i, clipped).
                      i >= 2     → soft penalty inside the IRLS P-step.
    curve_cons:       list of (i, L_x, L_y) direct curve-point constraints.
    damping:          in (0, 1]; 1 = standard BCD (oscillates), 0.5 = balanced.
    tol:              outer-loop convergence tolerance.

    Returns: dict { L: (N,K), dP: (K,3), n_iter, runtime }.
    """
    K  = palette.shape[0]
    t0 = time.time()

    # ── Direct curve-point constraints (static across iters) ─────
    cl_cons = [(i * N + int(round(Lx * (N - 1))), Ly)
               for (i, Lx, Ly) in curve_cons]

    # ── Renormalized weights for L-system (static) ───────────────
    if W_at_cons.shape[0] > 0:
        tilde_W = renormalize_weights(W_at_cons.T, rho=rho)
    else:
        tilde_W = np.zeros((K, 1))

    # ── Init ─────────────────────────────────────────────────────
    lum_scale     = np.ones(K)
    delta_palette = np.zeros((K, 3))
    L             = np.tile(np.linspace(0, 1, N)[:, None], (1, K))   # identity

    n_iter = 0
    for it in range(max_iter):
        n_iter += 1

        # 1) Compute coupled L-targets (uses current ΔP).
        pl_cons = compute_L_targets(palette, W_at_cons, target_colors,
                                     delta_palette, N)

        # 2) L-system, with damped update.
        L_new = solve_L_system(pl_cons, cl_cons, tilde_W, N, w_eq, lum_scale)
        L = (1 - damping) * L + damping * L_new
        new_lum_scale = compute_lum_scale(L, N)

        # 3) Compute coupled P-targets (uses freshly updated L).
        rP_targets = compute_P_targets(palette, W_at_cons, target_colors,
                                        L, tilde_W, N)

        # 4) P-system, with damped update.
        dP_new = solve_P_system(palette, W_at_cons, rP_targets,
                                 palette_cons, new_lum_scale,
                                 w_sp=w_sp, w_eq=w_eq, q=q)
        new_delta = (1 - damping) * delta_palette + damping * dP_new

        d_ls = np.linalg.norm(new_lum_scale - lum_scale)
        d_dp = np.linalg.norm(new_delta - delta_palette)
        lum_scale     = new_lum_scale
        delta_palette = new_delta

        if verbose:
            print(f"  iter {it+1:2d}  d_lum_scale={d_ls:.2e}  "
                  f"d_dP={d_dp:.2e}")
        if d_ls < tol and d_dp < tol:
            break

    runtime = time.time() - t0
    return {"L": L, "dP": delta_palette,
            "n_iter": n_iter, "runtime": runtime}