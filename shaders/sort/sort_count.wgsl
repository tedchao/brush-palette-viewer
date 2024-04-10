#import sorting


@group(0) @binding(0) var<storage> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<storage> config: sorting::Config;

var<workgroup> histogram: array<atomic<u32>, sorting::BIN_COUNT>;

@compute
@workgroup_size(sorting::WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    if local_id.x < sorting::BIN_COUNT {
        histogram[local_id.x] = 0u;
    }
    workgroupBarrier();
    var num_blocks = config.num_blocks_per_wg;
    var wg_block_start = sorting::BLOCK_SIZE * num_blocks * group_id.x;
    let num_not_additional = config.num_wgs - config.num_wgs_with_additional_blocks;
    if group_id.x >= num_not_additional {
        wg_block_start += (group_id.x - num_not_additional) * sorting::BLOCK_SIZE;
        num_blocks += 1u;
    }
    var block_index = wg_block_start + local_id.x;
    let shift_bit = config.shift;
    for (var block_count = 0u; block_count < num_blocks; block_count++) {
        var data_index = block_index;
        for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
            if data_index < config.num_keys {
                let local_key = (src[data_index] >> shift_bit) & 0xfu;
                atomicAdd(&histogram[local_key], 1u);
            }
            data_index += sorting::WG;
        }
        block_index += sorting::BLOCK_SIZE;
    }
    workgroupBarrier();
    if local_id.x < sorting::BIN_COUNT {
        counts[local_id.x * config.num_wgs + group_id.x] = histogram[local_id.x];
    }
}
