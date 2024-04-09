// 'software' wave emulation. This is obviously awful and much less performant than wave intrinsics.
// but, it will do for now.

// Mutable globals for the kernel. Obviously not ideal, should just be passed around,
// but too much refactoring of the ported code for now.
const MAX_DIM: u32 = 256u; // max(US_DIM, max(SCAN_DIM, DS_DIM))

var<private> local_invocation_idx: u32;

var<workgroup> g_wave_u32: array<u32, MAX_DIM>;
var<workgroup> g_wave_bool: array<bool, MAX_DIM>;

fn init(idx: u32) {
    local_invocation_idx = idx;
}

// Emulate a 32 lane system as the code seems best set up for it.
fn WaveGetLaneCount() -> u32 {
    return 32u;
}

fn getWaveIndex(gtid: u32) -> u32 {
    return gtid / WaveGetLaneCount();
}


fn _softWaveOffset() -> u32 {
    return WaveGetLaneCount() * getWaveIndex(local_invocation_idx);
}

fn WaveGetLaneIndex() -> u32 {
    return local_invocation_idx % WaveGetLaneCount();
}


// Software emulation of some wave functions.
// These are obviouisly much slower than their respective intrinsics!
fn WavePrefixSum(value: u32) -> u32 {
    workgroupBarrier();
    g_wave_u32[local_invocation_idx] = value;
    workgroupBarrier();
    var acc = 0u;
    let offset = _softWaveOffset();
    for(var i = 0u; i < WaveGetLaneIndex(); i++) {
        acc += g_wave_u32[i + offset];
    }
    return acc;
}

fn WaveReadLaneAt(expr: u32, laneIndex: u32) -> u32 {
    workgroupBarrier();
    g_wave_u32[local_invocation_idx] = expr;
    workgroupBarrier();
    return g_wave_u32[laneIndex + _softWaveOffset()];
}

fn WaveActiveBallot(expr: bool) -> u32 {
    workgroupBarrier();
    g_wave_bool[local_invocation_idx] = expr;
    workgroupBarrier();
    var mask: u32 = 0u;
    let offset = _softWaveOffset();
    for(var i = 0u; i < WaveGetLaneCount(); i++) {
        if g_wave_bool[offset + i] {
            mask = mask | (1u << i);
        }
    }
    return mask;
}
