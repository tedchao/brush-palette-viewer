// palette_remix.wgsl
//
// Final pass: read per-pixel K_full weights, multiply by palette, write RGB.
//
// This is the cheapest shader in the palette pipeline; runs every time the
// palette changes (and once per frame even if it doesn't). Reads K weights
// per pixel, does K*3 multiplies, writes 3 floats. Bandwidth-bound.
//
// Buffer layouts:
//   weights : pixel-major, weights[ (y*W + x)*K_FULL + k ] for k in 0..k_full
//             length = W * H * K_FULL   (K_FULL is the static MAX; tail unused)
//   palette : (k_full, 3), row-major, length = k_full * 3
//   output  : pixel-major RGB, length = W * H * 3
//             output[ (y*W + x)*3 + c ] for c in 0..3

const MAX_K_FULL: u32 = 8u;

@group(0) @binding(0) var<storage, read>       weights : array<f32>;
@group(0) @binding(1) var<storage, read>       palette : array<f32>;
@group(0) @binding(2) var<storage, read>       alpha_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> output  : array<u32>;

struct Uniforms {
    img_w  : u32,
    img_h  : u32,
    k_full : u32,
    pad_a  : u32,
    bg_r   : f32,
    bg_g   : f32,
    bg_b   : f32,
    pad_b  : u32,
}
@group(0) @binding(4) var<storage, read> uniforms : Uniforms;

const WG_SIZE: u32 = 256u;

@compute
@workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let pix_idx = gid.x;
    let total_pixels = uniforms.img_w * uniforms.img_h;
    if pix_idx >= total_pixels {
        return;
    }
    let x = pix_idx % uniforms.img_w;
    let y = pix_idx / uniforms.img_w;
    let weight_base = pix_idx * MAX_K_FULL;

    var r: f32 = 0.0;
    var g: f32 = 0.0;
    var b: f32 = 0.0;
    
    for (var k: u32 = 0u; k < MAX_K_FULL; k = k + 1u) {
        if k >= uniforms.k_full {
            break;
        }
        let w = weights[pix_idx * MAX_K_FULL + k];
        let pal_base = k * 3u;
        r = r + w * palette[pal_base + 0u];
        g = g + w * palette[pal_base + 1u];
        b = b + w * palette[pal_base + 2u];
    }
    
    // r, g, b above are already premultiplied (sum of T*alpha_t * w_kt * P_k).
    // Composite over background:  out = premul_rgb + (1 - alpha) * bg
    let a = alpha_in[pix_idx];
    let one_minus_a = 1.0 - a;
    let r_out = r + one_minus_a * uniforms.bg_r;
    let g_out = g + one_minus_a * uniforms.bg_g;
    let b_out = b + one_minus_a * uniforms.bg_b;
    
    let rgba = vec4f(r_out, g_out, b_out, 1.0);
    let cu = vec4u(clamp(rgba * 255.0, vec4f(0.0), vec4f(255.0)));
    let packed: u32 = cu.x | (cu.y << 8u) | (cu.z << 16u) | (cu.w << 24u);
    output[pix_idx] = packed;
}
