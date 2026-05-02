// palette_remix.wgsl
//
// Final pass: read per-pixel K_full weights, multiply by palette, write RGB.
// Now also applies delta_palette (ΔP) and per-palette tone curves (L).
//
// Color computation:
//   c(x)     = Σ_k W_k * P_k                         (original)
//   L_hat    = dot(c, BT709)                          (luminance)
//   delta(x) = Σ_k tilde_W_k * (f_k(L_hat) - L_hat) (curve shift)
//   out(x)   = Σ_k W_k * (P_k + dP_k) + delta*(1,1,1)
//
// Buffer layouts:
//   weights       : pixel-major [H*W*MAX_K_FULL]
//   palette       : (k_full, 3) original palette
//   alpha_in      : [H*W]
//   output        : packed RGBA u32 [H*W]
//   uniforms      : see Uniforms struct
//   delta_palette : (k_full, 3) ΔP from optimizer
//   L_curves      : (n_curve_samples, k_full) tone curves, col-major

const MAX_K_FULL: u32 = 8u;
const BT709_R: f32 = 0.2126;
const BT709_G: f32 = 0.7152;
const BT709_B: f32 = 0.0722;

@group(0) @binding(0) var<storage, read>       weights       : array<f32>;
@group(0) @binding(1) var<storage, read>       palette       : array<f32>;
@group(0) @binding(2) var<storage, read>       alpha_in      : array<f32>;
@group(0) @binding(3) var<storage, read_write> output        : array<u32>;
@group(0) @binding(4) var<storage, read>       delta_palette : array<f32>;
@group(0) @binding(5) var<storage, read>       L_curves      : array<f32>;

struct Uniforms {
    img_w          : u32,
    img_h          : u32,
    k_full         : u32,
    n_curve_samples: u32,
    bg_r           : f32,
    bg_g           : f32,
    bg_b           : f32,
    rho            : f32,
}
@group(0) @binding(6) var<storage, read> uniforms : Uniforms;

const WG_SIZE: u32 = 256u;

@compute
@workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let pix_idx = gid.x;
    let total_pixels = uniforms.img_w * uniforms.img_h;
    if pix_idx >= total_pixels { return; }
    
    let K = uniforms.k_full;
    let N = uniforms.n_curve_samples;
    
    var W: array<f32, 8>;
    for (var k: u32 = 0u; k < MAX_K_FULL; k++) {
        W[k] = select(0.0, weights[pix_idx * MAX_K_FULL + k], k < K);
    }
    
    var cr: f32 = 0.0; var cg: f32 = 0.0; var cb: f32 = 0.0;
    for (var k: u32 = 0u; k < MAX_K_FULL; k++) {
        if k >= K { break; }
        let pb = k * 3u;
        cr += W[k] * palette[pb];
        cg += W[k] * palette[pb + 1u];
        cb += W[k] * palette[pb + 2u];
    }
    
    let L_hat = clamp(BT709_R * cr + BT709_G * cg + BT709_B * cb, 0.0, 1.0);
    
    var tW: array<f32, 8>;
    var tW_sum: f32 = 0.0;
    for (var k: u32 = 0u; k < MAX_K_FULL; k++) {
        if k >= K { break; }
        let suppressed = select(W[k], W[k] / uniforms.rho, k < 2u);
        tW[k] = max(suppressed, 0.0);
        tW_sum += tW[k];
    }
    tW_sum = max(tW_sum, 1e-12);
    for (var k: u32 = 0u; k < MAX_K_FULL; k++) {
        tW[k] /= tW_sum;
    }
    
    let L_scaled = L_hat * f32(N - 1u);
    let n0 = u32(L_scaled);
    let n1 = min(n0 + 1u, N - 1u);
    let t  = L_scaled - f32(n0);
    
    var delta: f32 = 0.0;
    for (var k: u32 = 0u; k < MAX_K_FULL; k++) {
        if k >= K { break; }
        let base = k * N;
        let f_k = mix(L_curves[base + n0], L_curves[base + n1], t);
        delta += tW[k] * (f_k - L_hat);
    }
    
    var r: f32 = delta; var g: f32 = delta; var b: f32 = delta;
    for (var k: u32 = 0u; k < MAX_K_FULL; k++) {
        if k >= K { break; }
        let pb = k * 3u;
        r += W[k] * (palette[pb]      + delta_palette[pb]);
        g += W[k] * (palette[pb + 1u] + delta_palette[pb + 1u]);
        b += W[k] * (palette[pb + 2u] + delta_palette[pb + 2u]);
    }
    
    let a = alpha_in[pix_idx];
    let inv_a = 1.0 - a;
    let r_out = r + inv_a * uniforms.bg_r;
    let g_out = g + inv_a * uniforms.bg_g;
    let b_out = b + inv_a * uniforms.bg_b;
    
    let rgba = vec4f(r_out, g_out, b_out, 1.0);
    let cu = vec4u(clamp(rgba * 255.0, vec4f(0.0), vec4f(255.0)));
    output[pix_idx] = cu.x | (cu.y << 8u) | (cu.z << 16u) | (cu.w << 24u);
}