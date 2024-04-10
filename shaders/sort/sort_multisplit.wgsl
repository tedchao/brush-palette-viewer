#import sorting

@group(0) @binding(0) var<storage> src: array<u32>;
@group(0) @binding(1) var<storage> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<storage> config: sorting::Config;

const WMLS_SIZE = sorting::N_WARPS * sorting::BIN_COUNT;

var<workgroup> sh_wmls_histogram: array<u32, WMLS_SIZE>;
var<workgroup> sh_wmls_keys: array<u32, sorting::WG>;
var<workgroup> sh_wmls_ballot: array<u32, 64>;
var<workgroup> bin_offset_cache: array<u32, sorting::WG>;

@compute
@workgroup_size(sorting::WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    if local_id.x < sorting::BIN_COUNT {
        bin_offset_cache[local_id.x] = counts[local_id.x * config.num_wgs + group_id.x];
    }
    if local_id.x < WMLS_SIZE {
        sh_wmls_histogram[local_id.x] = 0u;
    }
    let warp_ix = local_id.x / sorting::WARP_SIZE;
    let lane_ix = local_id.x % sorting::WARP_SIZE;
    let base_ix = sorting::BLOCK_SIZE * group_id.x + warp_ix * (sorting::WARP_SIZE * sorting::ELEMENTS_PER_THREAD) + lane_ix;
    workgroupBarrier();
    // Note: these can be stored packed, either u16 or packed by hand
    var offsets: array<u32, sorting::ELEMENTS_PER_THREAD>;
    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        let ix = base_ix + i * sorting::WARP_SIZE;
        var key = ~0u;
        if ix < config.num_keys {
            key = src[ix];
        }
        let digit = (key >> config.shift) % sorting::BIN_COUNT;
        sh_wmls_keys[local_id.x] = digit;
        workgroupBarrier();
        var ballot = 0u;
        for (var j = 0u; j < sorting::WARP_SIZE; j++) {
            if digit == sh_wmls_keys[warp_ix * sorting::WARP_SIZE + j] {
                ballot |= (1u << j);
            }
        }
        let rank = countOneBits(ballot << (31u - lane_ix));
        let wmls_ix = digit * sorting::N_WARPS + warp_ix;
        offsets[i] = sh_wmls_histogram[wmls_ix] + rank - 1u;
        workgroupBarrier();
        if rank == 1u {
            sh_wmls_histogram[wmls_ix] += countOneBits(ballot);
        }
    }
    // Prefix sum over warps for each digit
    workgroupBarrier();
    var sum = 0u;
    if local_id.x < WMLS_SIZE {
        sum = sh_wmls_histogram[local_id.x];
    }
    let sub_ix = local_id.x % sorting::N_WARPS;
    for (var i = 0u; i < firstTrailingBit(sorting::N_WARPS); i++) {
        if local_id.x < WMLS_SIZE && sub_ix >= (1u << i) {
            sum += sh_wmls_histogram[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        if local_id.x < WMLS_SIZE && sub_ix >= (1u << i) {
            sh_wmls_histogram[local_id.x] = sum;
        }
        workgroupBarrier();
    }
    // scatter
    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        let ix = base_ix + i * sorting::WARP_SIZE;
        var key = ~0u;
        if ix < config.num_keys {
            key = src[ix];
        }
        let digit = (key >> config.shift) % sorting::BIN_COUNT;
        var total_offset = bin_offset_cache[digit];
        if warp_ix > 0u {
            total_offset += sh_wmls_histogram[digit * sorting::N_WARPS + warp_ix - 1u];
        }
        total_offset += offsets[i];
        if total_offset < config.num_keys {
            out[total_offset] = key;
        }
    }
    // TODO (multiple blocks per wg): update bin_offset_cache
}