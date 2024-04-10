#import sorting


@group(0) @binding(0) var<storage> src: array<u32>;
@group(0) @binding(1) var<storage> values: array<u32>;
@group(0) @binding(2) var<storage> counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_values: array<u32>;
@group(0) @binding(5) var<storage> config: sorting::Config;

var<workgroup> sums: array<u32, sorting::WG>;
var<workgroup> bin_offset_cache: array<u32, sorting::WG>;
var<workgroup> local_histogram: array<atomic<u32>, sorting::BIN_COUNT>;

@compute
@workgroup_size(sorting::WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    if local_id.x < sorting::BIN_COUNT {
        bin_offset_cache[local_id.x] = counts[local_id.x * config.num_wgs + group_id.x];
    }
    workgroupBarrier();
    let wg_block_start = sorting::BLOCK_SIZE * config.num_blocks_per_wg * group_id.x;
    let num_blocks = config.num_blocks_per_wg;
    // TODO: handle additional as above
    let block_index = wg_block_start + local_id.x;
    for (var block_count = 0u; block_count < num_blocks; block_count++) {
        var data_index = block_index;
        for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
            if local_id.x < sorting::BIN_COUNT {
                local_histogram[local_id.x] = 0u;
            }
            var local_key = ~0u;
            var local_value = 0u;

            if data_index < config.num_keys {
                local_key = src[data_index];
                local_value = values[data_index];
            }

            for (var bit_shift = 0u; bit_shift < sorting::BITS_PER_PASS; bit_shift += 2u) {
                let key_index = (local_key >> config.shift) & 0xfu;
                let bit_key = (key_index >> bit_shift) & 3u;
                var packed_histogram = 1u << (bit_key * 8u);
                // workgroup prefix sum
                var sum = packed_histogram;
                sums[local_id.x] = sum;
                for (var i = 0u; i < 8u; i++) {
                    workgroupBarrier();
                    if local_id.x >= (1u << i) {
                        sum += sums[local_id.x - (1u << i)];
                    }
                    workgroupBarrier();
                    sums[local_id.x] = sum;
                }
                workgroupBarrier();
                packed_histogram = sums[sorting::WG - 1u];
                packed_histogram = (packed_histogram << 8u) + (packed_histogram << 16u) + (packed_histogram << 24u);
                var local_sum = packed_histogram;
                if local_id.x > 0u {
                    local_sum += sums[local_id.x - 1u];
                }
                let key_offset = (local_sum >> (bit_key * 8u)) & 0xffu;
                sums[key_offset] = local_key;
                workgroupBarrier();
                local_key = sums[local_id.x];

                sums[key_offset] = local_value;
                workgroupBarrier();
                local_value = sums[local_id.x];
                workgroupBarrier();
            }
            let key_index = (local_key >> config.shift) & 0xfu;
            atomicAdd(&local_histogram[key_index], 1u);
            workgroupBarrier();
            var histogram_local_sum = 0u;
            if local_id.x < sorting::BIN_COUNT {
                histogram_local_sum = local_histogram[local_id.x];
            }
            // workgroup prefix sum of histogram
            var histogram_prefix_sum = histogram_local_sum;
            if local_id.x < sorting::BIN_COUNT {
                sums[local_id.x] = histogram_prefix_sum;
            }
            for (var i = 0u; i < 4u; i++) {
                workgroupBarrier();
                if local_id.x >= (1u << i) && local_id.x < sorting::BIN_COUNT {
                    histogram_prefix_sum += sums[local_id.x - (1u << i)];
                }
                workgroupBarrier();
                if local_id.x < sorting::BIN_COUNT {
                    sums[local_id.x] = histogram_prefix_sum;
                }
            }
            let global_offset = bin_offset_cache[key_index];
            workgroupBarrier();
            var local_offset = local_id.x;
            if key_index > 0u {
                local_offset -= sums[key_index - 1u];
            }
            let total_offset = global_offset + local_offset;
            if total_offset < config.num_keys {
                out[total_offset] = local_key;
                out_values[total_offset] = local_value;
            }
            if local_id.x < sorting::BIN_COUNT {
                bin_offset_cache[local_id.x] += local_histogram[local_id.x];
            }
            workgroupBarrier();
            data_index += sorting::WG;
        }
    }
}
