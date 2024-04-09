// 'software' wave emulation. This is obviously awful and much less performant than wave intrinsics.
// I don't have a good sense about whether a 'fancy' radix sort is still worth it
// without these intrinsics, but, it's at least easier to port for now
// and maybe WGSL gets these intrinsics one day.

// Mutable globals for the kernel. Obviously not ideal, should just be passed around,
// but too much refactoring of the ported code.

// Nb: needs to be at laest max(US_DIM, max(SCAN_DIM, DS_DIM))
const MAX_DIM: u32 = 512u;

var<private> local_invocation_idx: u32;

var<workgroup> g_wave_u32: array<u32, MAX_DIM>;

fn init(idx: u32) {
    local_invocation_idx = idx;
}

// Emulate a 32 lane system as the code seems best set up for it.
fn WaveGetLaneCount() -> u32 {
    return 16u;
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
fn WavePrefixSum(value: u32) -> u32 {
    workgroupBarrier();

    for(var i = 0u; i < MAX_DIM; i++) {
        g_wave_u32[i] = 0u;
    }
    workgroupBarrier();
    g_wave_u32[local_invocation_idx] = value;
    workgroupBarrier();
    var acc = 0u;
    let offset = _softWaveOffset();
    for(var i = 0u; i < WaveGetLaneIndex(); i++) {
        acc += g_wave_u32[i + offset];
    }
    workgroupBarrier();
    return acc;
}

fn WaveReadLaneAt(expr: u32, laneIndex: u32) -> u32 {
    // workgroupBarrier();
    // g_wave_u32[getWaveIndex(local_invocation_idx)] = 512512512u;

    // workgroupBarrier();

    // if WaveGetLaneIndex() == laneIndex {
    //     g_wave_u32[getWaveIndex(local_invocation_idx)] = expr;
    // }

    // workgroupBarrier();
    // return g_wave_u32[getWaveIndex(local_invocation_idx)];

    for(var i = 0u; i < MAX_DIM; i++) {
        // This is undefined as per the d3d spec.
        g_wave_u32[i] = 512512512u;
    }
    
    workgroupBarrier();
    g_wave_u32[local_invocation_idx] = expr;
    workgroupBarrier();
    let value = g_wave_u32[_softWaveOffset()  + laneIndex];
    workgroupBarrier();
    return value;
}

fn WaveActiveBallot(expr: bool) -> u32 {
    workgroupBarrier();

    // if local_invocation_idx == 0 {
    for(var i = 0u; i < MAX_DIM; i++) {
        g_wave_u32[i] = 0u;
    }
    // }

    workgroupBarrier();
    g_wave_u32[local_invocation_idx] = select(0u, 1u, expr);
    workgroupBarrier();
    var mask: u32 = 0u;
    let offset = _softWaveOffset();
    for(var i = 0u; i < WaveGetLaneCount(); i++) {
        if g_wave_u32[offset + i] == 1u {
            mask |= 1u << i;
        }
    }
    workgroupBarrier();
    return mask;
}
