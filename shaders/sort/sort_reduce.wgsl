#import sorting

@group(0) @binding(0) var<storage> counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> reduced: array<u32>;
@group(0) @binding(2) var<storage> config: sorting::Config;

var<workgroup> sums: array<u32, sorting::WG>;

@compute
@workgroup_size(sorting::WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let bin_id = group_id.x / config.num_reduce_wg_per_bin;
    let bin_offset = bin_id * config.num_wgs;
    let base_index = (group_id.x % config.num_reduce_wg_per_bin) * sorting::BLOCK_SIZE;
    var sum = 0u;
    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * sorting::WG + local_id.x;
        if data_index < config.num_wgs {
            sum += counts[bin_offset + data_index];
        }
    }
    sums[local_id.x] = sum;
    for (var i = 0u; i < 8u; i++) {
        workgroupBarrier();
        if local_id.x < ((sorting::WG / 2u) >> i) {
            sum += sums[local_id.x + ((sorting::WG / 2u) >> i)];
            sums[local_id.x] = sum;
        }
    }
    if local_id.x == 0u {
        reduced[group_id.x] = sum;
    }
}
