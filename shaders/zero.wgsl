@group(0) @binding(0) var<storage, read_write> arr: array<u32>;

@compute
@workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    if global_id.x < arrayLength(&arr) {
        arr[global_id.x] = 0u;
    }
}