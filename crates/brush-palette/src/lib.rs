//! brush-palette
//!
//! Loader for our palette-based 3DGS representation:
//!   - `*.pply` : geometry-only PLY (positions, scales, opacities, rotations).
//!                Same on-disk structure as Inria PLY but with f_dc/f_rest
//!                fields *absent* (or zero). Handled by piggybacking on
//!                brush-serde's existing streaming parser.
//!   - `*.gswp` : sidecar binary with palette + per-Gaussian weight SH
//!                coefficients. Parsed in Stage B.
//!
//! Stage B: parse .gswp, compute DC-only baked colors, inject them as the
//! degree-0 SH of the underlying Splats so Brush's existing rasterizer
//! renders the scene with the initial palette. View-dependent SH (the A/B
//! factors) is ignored at Stage B and added in Stage C with a forked
//! weight-splatting rasterizer.
pub mod shaders;
pub mod render;

use std::path::{Path, PathBuf};

use brush_render::gaussian_splats::Splats;
use brush_serde::{SplatMessage, stream_splat_from_ply};
use brush_vfs::DynRead;
use thiserror::Error;
use tokio_stream::Stream;

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------

/// 1 / (2 * sqrt(pi)). Same constant used in our CUDA kernel and in Brush's
/// sh.wgsl. The DC-only color in our convention is `SH_C0 * dc`.
pub const SH_C0: f32 = 0.282_094_77;

/// Brush's project_visible.wgsl applies `+0.5` after SH evaluation. To make
/// our DC-only render match the original colors WITHOUT forking the shader
/// in Stage B, we subtract this from the DC coefficients we inject.
/// Stage C will fork the shader and remove this hack entirely.
pub const DC_OFFSET_FOR_BRUSH_BIAS: f32 = 0.5 / SH_C0;

const GSWP_MAGIC: &[u8; 4] = b"GSWP";
const GSWP_VERSION: u32 = 1;
const GSWP_HEADER_SIZE: usize = 32;

// ----------------------------------------------------------------------------
// Errors
// ----------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum SidecarError {
    #[error("I/O error reading sidecar: {0}")]
    Io(#[from] std::io::Error),

    #[error("Sidecar too small: {0} bytes (need at least 32 for header)")]
    TooSmall(usize),

    #[error("Bad magic: expected GSWP, got {0:?}")]
    BadMagic([u8; 4]),

    #[error("Unsupported sidecar version: {0} (we support {1})")]
    BadVersion(u32, u32),

    #[error(
        "Sidecar gaussian count {sidecar_n} doesn't match PLY count {ply_n}"
    )]
    CountMismatch { sidecar_n: u32, ply_n: u32 },

    #[error(
        "Truncated sidecar payload: expected {expected} more bytes after header for {what}, got {got}"
    )]
    Truncated {
        what: &'static str,
        expected: usize,
        got: usize,
    },

    #[error(
        "Inconsistent low-rank dims: P*Q={pq} but splat_dim*(num_sh-1)={kl} (K={k}, num_sh={num_sh}, P={p}, Q={q})"
    )]
    BadLowRankDims {
        pq: u32,
        kl: u32,
        k: u32,
        num_sh: u32,
        p: u32,
        q: u32,
    },
}

// ----------------------------------------------------------------------------
// Path / extension helpers
// ----------------------------------------------------------------------------

pub fn is_pply_path(path: &Path) -> bool {
    // Match either:
    //   foo.pply         (final extension)
    //   foo.pply.ply     (macOS may append .ply when picking; check stem)
    let final_ext_pply = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("pply"))
        .unwrap_or(false);
    let stem_ends_in_pply = path
        .file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| std::path::Path::new(s).extension().and_then(|e| e.to_str()))
        .map(|s| s.eq_ignore_ascii_case("pply"))
        .unwrap_or(false);
    final_ext_pply || stem_ends_in_pply
}

pub fn sidecar_path_for(pply_path: &Path) -> Option<PathBuf> {
    if !is_pply_path(pply_path) {
        return None;
    }
    // For "foo.pply" -> "foo.gswp"
    // For "foo.pply.ply" -> strip both extensions, then add .gswp -> "foo.gswp"
    let mut p = pply_path.to_path_buf();
    p.set_extension("");           // strip last extension (.ply or .pply)
    if p.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("pply")).unwrap_or(false) {
        p.set_extension("");       // strip the .pply too if there was a .ply after it
    }
    p.set_extension("gswp");
    Some(p)
}

// ----------------------------------------------------------------------------
// Sidecar data
// ----------------------------------------------------------------------------

#[derive(Debug)]
pub enum HighShs {
    /// Dense rest-band weights, shape (N, K_full * (num_sh - 1)), band-major.
    Dense { coeffs: Vec<f32> },
    /// Rank-1 factorization. A is (N, P), B is (N, Q), reconstruct via
    /// `outer(a_n, b_n).reshape(K_full*(num_sh-1))`.
    LowRank {
        a: Vec<f32>,
        b: Vec<f32>,
        p: u32,
        q: u32,
    },
}

#[derive(Debug)]
pub struct PaletteSidecar {
    pub n_splats: u32,
    pub k_full: u32,    // splat_dim, includes black + white + K chromatic
    pub num_sh: u32,    // SH coeffs per channel (e.g. 16 for degree 3)
    pub palette: Vec<f32>,    // (K_full * 3) row-major
    pub low_shs_w: Vec<f32>,  // (N * K_full) row-major; DC-band weights
    pub high_shs: HighShs,
}

impl PaletteSidecar {
    /// Parse a .gswp file from a slice. Validates magic + version + sizes
    /// against the expected splat count from the PLY (pass `expected_n_splats`
    /// so a mismatched pair is caught up-front).
    pub fn parse(bytes: &[u8], expected_n_splats: u32) -> Result<Self, SidecarError> {
        if bytes.len() < GSWP_HEADER_SIZE {
            return Err(SidecarError::TooSmall(bytes.len()));
        }

        // Header: 4-byte magic + 7 little-endian u32s = 32 bytes
        let magic: [u8; 4] = bytes[0..4].try_into().expect("len checked");
        if &magic != GSWP_MAGIC {
            return Err(SidecarError::BadMagic(magic));
        }
        let version = read_u32_le(&bytes[4..8]);
        if version != GSWP_VERSION {
            return Err(SidecarError::BadVersion(version, GSWP_VERSION));
        }

        let n_splats   = read_u32_le(&bytes[8..12]);
        let k_full     = read_u32_le(&bytes[12..16]);
        let num_sh     = read_u32_le(&bytes[16..20]);
        let low_rank   = read_u32_le(&bytes[20..24]);
        let p          = read_u32_le(&bytes[24..28]);
        let q          = read_u32_le(&bytes[28..32]);

        if n_splats != expected_n_splats {
            return Err(SidecarError::CountMismatch {
                sidecar_n: n_splats,
                ply_n: expected_n_splats,
            });
        }

        let mut cursor = GSWP_HEADER_SIZE;

        // palette: K_full × 3 floats
        let palette_floats = (k_full as usize) * 3;
        let palette = read_f32_block(bytes, &mut cursor, palette_floats, "palette")?;

        // low_shs_w: N × K_full floats
        let low_floats = (n_splats as usize) * (k_full as usize);
        let low_shs_w = read_f32_block(bytes, &mut cursor, low_floats, "low_shs_w")?;

        // high_shs: dense or low-rank
        let high_shs = if low_rank == 0 {
            let high_floats = (n_splats as usize) * (k_full as usize) * (num_sh as usize - 1);
            let coeffs = read_f32_block(bytes, &mut cursor, high_floats, "high_shs (dense)")?;
            HighShs::Dense { coeffs }
        } else {
            // P*Q must equal K_full * (num_sh - 1)
            let pq = p.saturating_mul(q);
            let kl = k_full.saturating_mul(num_sh - 1);
            if pq != kl {
                return Err(SidecarError::BadLowRankDims {
                    pq,
                    kl,
                    k: k_full,
                    num_sh,
                    p,
                    q,
                });
            }
            let a_floats = (n_splats as usize) * (p as usize);
            let b_floats = (n_splats as usize) * (q as usize);
            let a = read_f32_block(bytes, &mut cursor, a_floats, "high_shs_A")?;
            let b = read_f32_block(bytes, &mut cursor, b_floats, "high_shs_B")?;
            HighShs::LowRank { a, b, p, q }
        };

        Ok(Self {
            n_splats,
            k_full,
            num_sh,
            palette,
            low_shs_w,
            high_shs,
        })
    }

    /// Compute DC-only baked color per Gaussian:
    ///     color[n] = SH_C0 * sum_k low_shs_w[n, k] * palette[k]
    /// Returns flat Vec<f32> of length N*3 in [r, g, b, r, g, b, ...] order.
    ///
    /// `subtract_brush_bias`: if true, subtract 0.5/SH_C0 from each DC
    /// coefficient before scaling, so that Brush's stock shader (which adds
    /// +0.5 at render time) produces the right color. Set this to true for
    /// Stage B (where we route through the unmodified rasterizer).
    pub fn bake_dc_colors(&self, subtract_brush_bias: bool) -> Vec<f32> {
        let n = self.n_splats as usize;
        let k = self.k_full as usize;
        let mut out = vec![0.0f32; n * 3];

        for i in 0..n {
            let w = &self.low_shs_w[i * k..(i + 1) * k];
            let mut acc_r = 0.0f32;
            let mut acc_g = 0.0f32;
            let mut acc_b = 0.0f32;
            for j in 0..k {
                let pj = j * 3;
                acc_r += w[j] * self.palette[pj];
                acc_g += w[j] * self.palette[pj + 1];
                acc_b += w[j] * self.palette[pj + 2];
            }
            // Our renderer applies SH_C0 in shader: color = SH_C0 * dc.
            // We're injecting these as DC coefficients, so leave SH_C0 to be
            // applied later -- store the un-scaled DC value here.
            //
            // However, *our* DC value here represents `weighted_color / SH_C0`
            // (so that SH_C0 * stored_dc == weighted_color). Brush will add
            // +0.5 to the result, so to compensate we subtract 0.5/SH_C0 from
            // the stored DC.
            let mut r = acc_r / SH_C0;
            let mut g = acc_g / SH_C0;
            let mut b = acc_b / SH_C0;
            if subtract_brush_bias {
                r -= DC_OFFSET_FOR_BRUSH_BIAS;
                g -= DC_OFFSET_FOR_BRUSH_BIAS;
                b -= DC_OFFSET_FOR_BRUSH_BIAS;
            }
            let oi = i * 3;
            out[oi] = r;
            out[oi + 1] = g;
            out[oi + 2] = b;
        }
        out
    }
}

// ----------------------------------------------------------------------------
// Streaming loader: same as Stage A, just delegates to brush-serde
// ----------------------------------------------------------------------------

pub fn stream_pply_as_geometry(
    reader: Box<dyn DynRead>,
    update_every: Option<u32>,
    skip_sh: bool,
) -> impl Stream<Item = Result<SplatMessage, brush_serde::DeserializeError>> {
    stream_splat_from_ply(reader, update_every, skip_sh)
}

// ----------------------------------------------------------------------------
// PaletteSplats wrapper (unchanged from Stage A)
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// PaletteSplats: geometry (via vanilla Splats<B>) + palette/weight tensors.
// ----------------------------------------------------------------------------
//
// Stage C2.5a: holds the data the new shaders will need at render time.
//   - `splats`     : geometry (positions, rotations, scales, opacities) and
//                    the SH-color tensor (which we use for vanilla compatibility
//                    only; our render path ignores it).
//   - `palette`    : (K_full, 3) RGB colors, ROW-MAJOR.
//   - `low_shs_w`  : (N, K_full) DC-band per-Gaussian weights.
//   - `high_shs_a` : (N, P) low-rank factor A.
//   - `high_shs_b` : (N, Q) low-rank factor B.
//   - `k_full`, `num_sh`, `p`, `q` : scalar dims used by shader uniforms.
//
// Stage B's bake-into-Splats DC injection is no longer needed; the dedicated
// shader path will read these tensors directly.

use burn::prelude::*;
use burn::tensor::TensorData;

#[derive(Clone)]
pub struct PaletteSplats<B: burn::prelude::Backend> {
    pub splats: Splats<B>,
    
    pub palette:    Tensor<B, 2>,
    pub low_shs_w:  Tensor<B, 2>,
    pub high_shs_a: Tensor<B, 2>,
    pub high_shs_b: Tensor<B, 2>,
    
    pub k_full: u32,
    pub num_sh: u32,
    pub p:      u32,
    pub q:      u32,
}

impl<B: burn::prelude::Backend> PaletteSplats<B> {
    /// Build a PaletteSplats from already-loaded geometry and a parsed sidecar.
    /// Uploads palette + weight buffers to `device` as Burn tensors. The dense
    /// path materialises a synthetic (P,Q) rank-1 factorisation so the same
    /// shader code works for both paths -- but we error rather than do that
    /// silently for now; rank-1 only is supported.
    pub fn from_parts(
        splats: Splats<B>,
        sidecar: PaletteSidecar,
        device: &B::Device,
    ) -> Result<Self, anyhow::Error> {
        let n_splats = splats.num_splats() as usize;
        if n_splats != sidecar.n_splats as usize {
            anyhow::bail!(
                "PaletteSplats::from_parts: splats.num_splats() = {} but sidecar.n_splats = {}",
                n_splats,
                sidecar.n_splats,
            );
        }
        
        let k_full = sidecar.k_full as usize;
        let num_sh = sidecar.num_sh as usize;
        
        // Upload palette: (K_full, 3) row-major.
        let palette = Tensor::<B, 2>::from_data(
            TensorData::new(sidecar.palette.clone(), [k_full, 3]),
            device,
        );
        
        // Upload low_shs_w: (N, K_full).
        let low_shs_w = Tensor::<B, 2>::from_data(
            TensorData::new(sidecar.low_shs_w.clone(), [n_splats, k_full]),
            device,
        );
        
        // Resolve A/B. For Stage C2 we only support the low-rank path. The
        // dense path can be supported later by allocating P=1, Q=K*(num_sh-1)
        // and storing high_shs as B with A=ones.
        let (a_vec, b_vec, p_dim, q_dim) = match &sidecar.high_shs {
            HighShs::LowRank { a, b, p, q } => {
                (a.clone(), b.clone(), *p as usize, *q as usize)
            }
            HighShs::Dense { coeffs } => {
                // Dense fallback: treat as P=1 rank-1 factorisation.
                // A = ones (N, 1); B = the dense coeff block (N, K_full * (num_sh-1)).
                let q = k_full * (num_sh - 1);
                let ones = vec![1.0f32; n_splats]; // (N, 1)
                if coeffs.len() != n_splats * q {
                    anyhow::bail!(
                        "Dense high_shs has {} floats, expected N*K_full*(num_sh-1) = {}",
                        coeffs.len(),
                        n_splats * q,
                    );
                }
                (ones, coeffs.clone(), 1, q)
            }
        };
        
        let high_shs_a = Tensor::<B, 2>::from_data(
            TensorData::new(a_vec, [n_splats, p_dim]),
            device,
        );
        let high_shs_b = Tensor::<B, 2>::from_data(
            TensorData::new(b_vec, [n_splats, q_dim]),
            device,
        );
        
        Ok(Self {
            splats,
            palette,
            low_shs_w,
            high_shs_a,
            high_shs_b,
            k_full: sidecar.k_full,
            num_sh: sidecar.num_sh,
            p:      p_dim as u32,
            q:      q_dim as u32,
        })
    }
    
    pub fn num_splats(&self) -> u32 {
        self.splats.num_splats()
    }
    
    pub fn sh_degree(&self) -> u32 {
        self.splats.sh_degree()
    }
    
    /// Expose the inner geometry-only Splats. Callers that only need the
    /// vanilla render path (or need to satisfy an existing API) can use this.
    pub fn into_inner(self) -> Splats<B> {
        self.splats
    }
    
    /// The original Stage A constructor; kept for backward compatibility with
    /// any code that wraps a Splats without the sidecar (should be unused now,
    /// but harmless to keep).
    pub fn from_splats(splats: Splats<B>) -> Self
    where
        B::Device: Default,
    {
        // Caller is responsible for not invoking the palette render path on
        // an instance built this way; the tensors are zero-sized stubs.
        let device = B::Device::default();
        let zero1 = Tensor::<B, 2>::from_data(TensorData::new(vec![0.0f32; 0], [0, 0]), &device);
        let zero2 = zero1.clone();
        let zero3 = zero1.clone();
        let zero4 = zero1.clone();
        Self {
            splats,
            palette:    zero1,
            low_shs_w:  zero2,
            high_shs_a: zero3,
            high_shs_b: zero4,
            k_full: 0,
            num_sh: 0,
            p: 0,
            q: 0,
        }
    }
}

// Suppress unused-warning hints; these will be used in C2.5b.
#[allow(dead_code)]
fn _palette_splats_field_uses<B: burn::prelude::Backend>(p: &PaletteSplats<B>) {
    let _ = (&p.palette, &p.low_shs_w, &p.high_shs_a, &p.high_shs_b);
    let _ = (p.k_full, p.num_sh, p.p, p.q);
}

// ----------------------------------------------------------------------------
// Internal helpers
// ----------------------------------------------------------------------------

fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes(b[..4].try_into().expect("slice >= 4"))
}

fn read_f32_block(
    bytes: &[u8],
    cursor: &mut usize,
    n_floats: usize,
    what: &'static str,
) -> Result<Vec<f32>, SidecarError> {
    let need = n_floats * 4;
    let avail = bytes.len().saturating_sub(*cursor);
    if avail < need {
        return Err(SidecarError::Truncated {
            what,
            expected: need,
            got: avail,
        });
    }
    let mut out = Vec::with_capacity(n_floats);
    for i in 0..n_floats {
        let off = *cursor + i * 4;
        out.push(f32::from_le_bytes(
            bytes[off..off + 4].try_into().expect("range checked"),
        ));
    }
    *cursor += need;
    Ok(out)
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pply_extension_detection() {
        assert!(is_pply_path(Path::new("scene.pply")));
        assert!(is_pply_path(Path::new("/tmp/foo/scene.PPLY")));
        assert!(is_pply_path(Path::new("scene.pply.ply")));   // macOS double-extension
        assert!(!is_pply_path(Path::new("scene.ply")));
        assert!(!is_pply_path(Path::new("scene")));
    }
    
    #[test]
    fn sidecar_path_resolution() {
        assert_eq!(sidecar_path_for(Path::new("/data/scene.pply")), Some(Path::new("/data/scene.gswp").to_path_buf()));
        assert_eq!(sidecar_path_for(Path::new("/data/scene.pply.ply")), Some(Path::new("/data/scene.gswp").to_path_buf()));
        assert_eq!(sidecar_path_for(Path::new("scene.ply")), None);
    }

    fn write_header(buf: &mut Vec<u8>, n: u32, k: u32, num_sh: u32, low_rank: u32, p: u32, q: u32) {
        buf.extend_from_slice(GSWP_MAGIC);
        buf.extend_from_slice(&GSWP_VERSION.to_le_bytes());
        buf.extend_from_slice(&n.to_le_bytes());
        buf.extend_from_slice(&k.to_le_bytes());
        buf.extend_from_slice(&num_sh.to_le_bytes());
        buf.extend_from_slice(&low_rank.to_le_bytes());
        buf.extend_from_slice(&p.to_le_bytes());
        buf.extend_from_slice(&q.to_le_bytes());
    }

    fn extend_f32s(buf: &mut Vec<u8>, vals: &[f32]) {
        for &v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    #[test]
    fn parse_low_rank_round_trip() {
        let n = 4u32;
        let k = 5u32;
        let num_sh = 16u32;
        let p = 5u32;
        let q = 15u32; // p*q = 75 = k*(num_sh-1) = 5*15

        let mut buf = Vec::new();
        write_header(&mut buf, n, k, num_sh, 1, p, q);

        let palette: Vec<f32> = (0..k * 3).map(|i| i as f32 * 0.1).collect();
        let low: Vec<f32> = (0..n * k).map(|i| i as f32).collect();
        let a: Vec<f32> = (0..n * p).map(|i| i as f32 + 100.0).collect();
        let b: Vec<f32> = (0..n * q).map(|i| i as f32 + 1000.0).collect();
        extend_f32s(&mut buf, &palette);
        extend_f32s(&mut buf, &low);
        extend_f32s(&mut buf, &a);
        extend_f32s(&mut buf, &b);

        let sc = PaletteSidecar::parse(&buf, n).expect("parse");
        assert_eq!(sc.n_splats, n);
        assert_eq!(sc.k_full, k);
        assert_eq!(sc.num_sh, num_sh);
        assert_eq!(sc.palette, palette);
        assert_eq!(sc.low_shs_w, low);
        match sc.high_shs {
            HighShs::LowRank { a: aa, b: bb, p: pp, q: qq } => {
                assert_eq!(aa, a);
                assert_eq!(bb, b);
                assert_eq!(pp, p);
                assert_eq!(qq, q);
            }
            _ => panic!("expected low-rank"),
        }
    }

    #[test]
    fn bad_magic_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"NOPE");
        buf.extend_from_slice(&[0u8; 28]);
        let err = PaletteSidecar::parse(&buf, 0).unwrap_err();
        assert!(matches!(err, SidecarError::BadMagic(_)));
    }

    #[test]
    fn count_mismatch_rejected() {
        let mut buf = Vec::new();
        write_header(&mut buf, 100, 5, 16, 0, 0, 0);
        buf.extend_from_slice(&[0u8; 1024]); // junk payload
        let err = PaletteSidecar::parse(&buf, 99).unwrap_err();
        assert!(matches!(err, SidecarError::CountMismatch { sidecar_n: 100, ply_n: 99 }));
    }

    #[test]
    fn dc_bake_math() {
        // Single splat, K=2, palette = [[1,0,0], [0,1,0]], low = [0.6, 0.4]
        // weighted color = 0.6*[1,0,0] + 0.4*[0,1,0] = [0.6, 0.4, 0]
        // Stored DC (no bias subtract) = [0.6, 0.4, 0] / SH_C0
        let mut buf = Vec::new();
        write_header(&mut buf, 1, 2, 1, 0, 0, 0); // num_sh=1 means no rest band
        extend_f32s(&mut buf, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);  // palette
        extend_f32s(&mut buf, &[0.6, 0.4]);                     // low_shs_w
        // num_sh - 1 = 0, so dense block is zero floats
        let sc = PaletteSidecar::parse(&buf, 1).unwrap();

        let dc_no_bias = sc.bake_dc_colors(false);
        assert!((dc_no_bias[0] - 0.6 / SH_C0).abs() < 1e-5);
        assert!((dc_no_bias[1] - 0.4 / SH_C0).abs() < 1e-5);
        assert!(dc_no_bias[2].abs() < 1e-5);

        let dc_bias = sc.bake_dc_colors(true);
        assert!((dc_bias[0] - (0.6 / SH_C0 - DC_OFFSET_FOR_BRUSH_BIAS)).abs() < 1e-5);
    }
}