// stub.wgsl
//
// Minimal compute shader to validate that brush-palette can register a
// shader via #[wgsl_kernel(...)] and have it compile through Brush's
// shader pipeline.
//
// This shader does ALMOST nothing: it reads one f32 from an input buffer,
// adds 1.0, and writes it to an output buffer. Just enough to exercise the
// proc-macro.
//
// No #import — we don't need any helpers yet.
// No uniforms binding here either; we'll add that in real shaders.

@group(0) @binding(0) var<storage, read>       input  : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

const WG_SIZE: u32 = 64u;

@compute
@workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let i = gid.x;
    if i >= arrayLength(&output) {
        return;
    }
    output[i] = input[i] + 1.0;
}
