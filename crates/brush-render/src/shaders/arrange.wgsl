@group(0) @binding(0) var<storage, read_write> arranged_ids: array<u32>;

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    if global_id.x >= arrayLength(&arranged_ids) {
        return;
    }
    arranged_ids[global_id.x] = global_id.x;
}