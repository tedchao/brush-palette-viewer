/******************************************************************************
 * Device Level 8-bit LSD Radix Sort using reduce then scan
 * 
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 2/13/2023
 * https://github.com/b0nes164/GPUSorting
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 ******************************************************************************/
//General macros 
const  PART_SIZE: u32 =       3840u;       //size of a partition tile
const  US_DIM: u32 =          128u;        //The number of threads in a Upsweep threadblock
const  SCAN_DIM: u32 =        128u;        //The number of threads in a Scan threadblock
const  DS_DIM: u32 =          256u;        //The number of threads in a Downsweep threadblock
const  RADIX: u32 =           256u;        //Number of digit bins
const  RADIX_MASK: u32 =      255u;        //Mask of digit bins
const  RADIX_LOG: u32 =       8u;          //log2(RADIX)
const  HALF_RADIX: u32 =      128u;        //For smaller waves where bit packing is necessary
const  HALF_MASK: u32 =       127u;        // '' 

//For the downsweep kernels
const DS_KEYS_PER_THREAD: u32  = 15u;     //The number of keys per thread in a Downsweep Threadblock
const MAX_DS_SMEM: u32       = 4096u;   //shared memory for downsweep kernel

var<private> e_numKeys: u32;
var<private> e_radixShift: u32;
var<private> e_threadBlocks: u32;
var<private> local_invocation_idx: u32;

@group(0) @binding(0) var<storage, read> b_sort: array<u32>;
@group(0) @binding(1) var<storage, read_write> b_passHist: array<u32>;
@group(0) @binding(2) var<storage, read_write> b_globalHist: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> b_sortPayload: array<u32>;
@group(0) @binding(4) var<storage, read_write> b_alt: array<u32>;
@group(0) @binding(5) var<storage, read_write> b_altPayload: array<u32>;

var<workgroup> g_us: array<atomic<u32>, 512>;
var<workgroup> g_scan: array<u32, SCAN_DIM>;
var<workgroup> g_ds: array<u32, MAX_DS_SMEM>;

/// Wavee
const MAX_DIM: u32 = 512u;
var<workgroup> g_wave_scratch: array<u32, MAX_DIM>;

fn init(idx: u32) {
    e_numKeys = arrayLength(&b_sort);
    e_radixShift = 0u;
    e_threadBlocks = (e_numKeys + PART_SIZE - 1) / PART_SIZE;
    local_invocation_idx = idx;
}

// Emulate a 32 lane system as the code seems best set up for it.
fn WaveGetLaneCount() -> u32 {
    return 32u;
}

fn WaveStartGTIndex() -> u32 {
    return WaveGetLaneCount() * getWaveIndex(local_invocation_idx);
}

fn WaveGetLaneIndex() -> u32 {
    return local_invocation_idx % WaveGetLaneCount();
}


// Software emulation of some wave functions.
fn WavePrefixSum(value: u32) -> u32 {
    workgroupBarrier();
    for(var i = 0u; i < MAX_DIM; i++) {
        g_wave_scratch[i] = 0u;
    }
    workgroupBarrier();
    g_wave_scratch[local_invocation_idx] = value;
    workgroupBarrier();
    var acc = 0u;
    let offset = WaveStartGTIndex();
    for(var i = 0u; i < WaveGetLaneIndex(); i++) {
        acc += g_wave_scratch[i + offset];
    }
    workgroupBarrier();
    return acc;
}

fn WaveReadLaneAt(expr: u32, laneIndex: u32) -> u32 {
    workgroupBarrier();
    g_wave_scratch[local_invocation_idx] = expr;
    workgroupBarrier();
    return g_wave_scratch[WaveStartGTIndex() + laneIndex];
}

fn WaveActiveBallot(expr: bool) -> u32 {
    workgroupBarrier();
    // if local_invocation_idx == 0 {
    // for(var i = 0u; i < MAX_DIM; i++) {
    //     g_wave_u32[i] = 0u;
    // }
    // }
    workgroupBarrier();
    g_wave_scratch[local_invocation_idx] = select(0u, 1u, expr);
    workgroupBarrier();
    var mask: u32 = 0u;
    let offset = WaveStartGTIndex();
    for(var i = 0u; i < WaveGetLaneCount(); i++) {
        if g_wave_scratch[offset + i] > 0 {
            mask |= (1u << i);
        }
    }
    workgroupBarrier();
    return mask;
}

////


fn GroupMemoryBarrierWithGroupSync() {
    workgroupBarrier();
}

fn getWaveIndex(gtid: u32) -> u32
{
    return gtid / WaveGetLaneCount();
}

fn ExtractDigit(key: u32) -> u32
{
    return key >> e_radixShift & RADIX_MASK;
}


fn SubPartSizeWGE16() -> u32
{
    return DS_KEYS_PER_THREAD * WaveGetLaneCount();
}

fn SharedOffsetWGE16(gtid: u32) -> u32
{
    return WaveGetLaneIndex() + getWaveIndex(gtid) * SubPartSizeWGE16();
}

fn DeviceOffsetWGE16(gtid: u32, gid: u32) -> u32
{
    return SharedOffsetWGE16(gtid) + gid * PART_SIZE;
}


fn GlobalHistOffset() -> u32
{
    return e_radixShift << 5;
}

fn WaveHistsSizeWGE16() -> u32
{
    return DS_DIM / WaveGetLaneCount() * RADIX;
}

fn ternary(cond: bool, tr: u32, fl: u32) -> u32 {
    if cond {
        return tr;    
    }
    return fl;
}

fn Upsweep(gtid: vec3u, gid : vec3u)
{
    init(gtid.x);

    //clear shared memory
    let histsEnd = RADIX * 2;
    for (var i = gtid.x; i < histsEnd; i += US_DIM) {
        g_us[i] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    //histogram, 64 threads to a histogram
    let histOffset = gtid.x / 64 * RADIX;
    let partitionEnd = ternary(gid.x == e_threadBlocks - 1,
        e_numKeys, (gid.x + 1) * PART_SIZE);
    
    for (var i = gtid.x + gid.x * PART_SIZE; i < partitionEnd; i += US_DIM) {
        atomicAdd(&g_us[ExtractDigit(b_sort[i]) + histOffset], 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    
    //reduce and pass to tile histogram
    for (var i = gtid.x; i < RADIX; i += US_DIM)
    {
        g_us[i] += g_us[i + RADIX];
        b_passHist[i * e_threadBlocks + gid.x] = g_us[i];
    }
    
    //Larger 16 or greater can perform a more elegant scan because 16 * 16 = 256
    for (var i = gtid.x; i < RADIX; i += US_DIM) {
        g_us[i] += WavePrefixSum(g_us[i]);
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (gtid.x < (RADIX / WaveGetLaneCount()))
    {
        g_us[(gtid.x + 1) * WaveGetLaneCount() - 1] +=
            WavePrefixSum(g_us[(gtid.x + 1) * WaveGetLaneCount() - 1]);
    }
    GroupMemoryBarrierWithGroupSync();
    
    //atomically add to global histogram
    let globalHistOffset = GlobalHistOffset();
    let laneMask = WaveGetLaneCount() - 1;
    let circularLaneShift = WaveGetLaneIndex() + 1 & laneMask;
    for (var i = gtid.x; i < RADIX; i += US_DIM)
    {
        let index = circularLaneShift + (i & ~laneMask);
        atomicAdd(&b_globalHist[index + globalHistOffset],
            ternary(WaveGetLaneIndex() != laneMask, g_us[i], 0u) +
            ternary(i >= WaveGetLaneCount(), WaveReadLaneAt(g_us[i - 1], 0u), 0u));
    }
}

//Scan along the spine of the upsweep
fn Scan(gtid : vec3u, gid : vec3u)
{
    init(gtid.x);

    var aggregate = 0u;
    let laneMask = WaveGetLaneCount() - 1;
    let circularLaneShift = WaveGetLaneIndex() + 1 & laneMask;
    let partionsEnd = e_threadBlocks / SCAN_DIM * SCAN_DIM;
    let offset = gid.x * e_threadBlocks;
    var i = gtid.x;
    for (; i < partionsEnd; i += SCAN_DIM)
    {
        g_scan[gtid.x] = b_passHist[i + offset];
        g_scan[gtid.x] += WavePrefixSum(g_scan[gtid.x]);
        GroupMemoryBarrierWithGroupSync();
        
        if (gtid.x < SCAN_DIM / WaveGetLaneCount())
        {
            g_scan[(gtid.x + 1) * WaveGetLaneCount() - 1] +=
                WavePrefixSum(g_scan[(gtid.x + 1) * WaveGetLaneCount() - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
        
        b_passHist[circularLaneShift + (i & ~laneMask) + offset] =
            ternary(WaveGetLaneIndex() != laneMask, g_scan[gtid.x], 0u) +
            ternary(gtid.x >= WaveGetLaneCount(),
            WaveReadLaneAt(g_scan[gtid.x - 1], 0u), 0u) +
            aggregate;

        aggregate += g_scan[SCAN_DIM - 1];
        GroupMemoryBarrierWithGroupSync();
    }
    
    //partial
    if (i < e_threadBlocks) {
        g_scan[gtid.x] = b_passHist[offset + i];
    }
    g_scan[gtid.x] += WavePrefixSum(g_scan[gtid.x]);
    GroupMemoryBarrierWithGroupSync();
        
    if (gtid.x < SCAN_DIM / WaveGetLaneCount())
    {
        g_scan[(gtid.x + 1) * WaveGetLaneCount() - 1] +=
            WavePrefixSum(g_scan[(gtid.x + 1) * WaveGetLaneCount() - 1]);
    }
    GroupMemoryBarrierWithGroupSync();
    
    let index = circularLaneShift + (i & ~laneMask);
    if (index < e_threadBlocks)
    {
        b_passHist[index + offset] = ternary(WaveGetLaneIndex() != laneMask, g_scan[gtid.x], 0u) +
            ternary(gtid.x >= WaveGetLaneCount(), g_scan[(gtid.x & ~laneMask) - 1], 0u) + aggregate;

    }
}

fn Downsweep(gtid : vec3u, gid : vec3u)
{
    init(gtid.x);

    if (gid.x < e_threadBlocks - 1)
    {
        var keys = array<u32, DS_KEYS_PER_THREAD>();
        var offsets = array<u32, DS_KEYS_PER_THREAD>();
        
        //Load keys into registers
        {
            var t = DeviceOffsetWGE16(gtid.x, gid.x);
            for (var i = 0u;
                    i < DS_KEYS_PER_THREAD;
                    i++)
            {
                keys[i] = b_sort[t];
                t += WaveGetLaneCount();
            }
        }
        
        //Clear histogram memory
        for (var i = gtid.x; i < WaveHistsSizeWGE16(); i += DS_DIM) {
            g_ds[i] = 0u;
        }

        GroupMemoryBarrierWithGroupSync();

        //Warp Level Multisplit
        let waveParts = (WaveGetLaneCount() + 31) / 32;
        
        for (var i = 0u; i < DS_KEYS_PER_THREAD; i++)
        {
            var waveFlags = ternary((WaveGetLaneCount() & 31) > 0,
                (1u << WaveGetLaneCount()) - 1, 0xffffffffu);
            
            for (var k = 0u; k < RADIX_LOG; k++)
            {
                let t = keys[i] >> (k + e_radixShift) & 1u;
                let ballot = WaveActiveBallot(t > 0);
                waveFlags &= ternary(t > 0u, 0u, 0xffffffffu) ^ ballot;
            }
                
            var bits = 0u;
            let wavePart = 0u;
            if (WaveGetLaneIndex() >= wavePart * 32u)
            {
                let ltMask = ternary(WaveGetLaneIndex() >=(wavePart + 1) * 32u,
                    0xffffffffu, (1u << (WaveGetLaneIndex() & 31)) - 1);
                bits += countOneBits(waveFlags & ltMask);
            }
                
            let index = ExtractDigit(keys[i]) + (getWaveIndex(gtid.x) * RADIX);
            offsets[i] = g_ds[index] + bits;
                
            GroupMemoryBarrierWithGroupSync();
            if (bits == 0)
            {
                g_ds[index] += countOneBits(waveFlags);
            }
            GroupMemoryBarrierWithGroupSync();
        }
        
        //inclusive/exclusive prefix sum up the histograms
        //followed by exclusive prefix sum across the reductions
        var reduction = g_ds[gtid.x];
        for (var i = gtid.x + RADIX; i < WaveHistsSizeWGE16(); i += RADIX)
        {
            reduction += g_ds[i];
            g_ds[i] = reduction - g_ds[i];
        }
        
        reduction += WavePrefixSum(reduction);
        GroupMemoryBarrierWithGroupSync();

        let laneMask = WaveGetLaneCount() - 1;
        g_ds[((WaveGetLaneIndex() + 1) & laneMask) + (gtid.x & ~laneMask)] = reduction;
        GroupMemoryBarrierWithGroupSync();
            
        if (gtid.x < RADIX / WaveGetLaneCount())
        {
            g_ds[gtid.x * WaveGetLaneCount()] =
                WavePrefixSum(g_ds[gtid.x * WaveGetLaneCount()]);
        }
        GroupMemoryBarrierWithGroupSync();
            
        if (WaveGetLaneIndex() > 0) {
            g_ds[gtid.x] += WaveReadLaneAt(g_ds[gtid.x - 1], 1u);
        }
        GroupMemoryBarrierWithGroupSync();
    
        //Update offsets
        if (gtid.x >= WaveGetLaneCount())
        {
            let t = getWaveIndex(gtid.x) * RADIX;
            
            for (var i = 0u; i < DS_KEYS_PER_THREAD; i++)
            {
                let t2 = ExtractDigit(keys[i]);
                offsets[i] += g_ds[t2 + t] + g_ds[t2];
            }
        }
        else
        {
            for (var i = 0u; i < DS_KEYS_PER_THREAD; i++) {
                offsets[i] += g_ds[ExtractDigit(keys[i])];
            }
        }
        
        //take advantage of barrier
        let exclusiveWaveReduction = g_ds[gtid.x];
        GroupMemoryBarrierWithGroupSync();
        
        //scatter keys into shared memory
        for (var i = 0u; i < DS_KEYS_PER_THREAD; i++) {
            g_ds[offsets[i]] = keys[i];
        }
    
        g_ds[gtid.x + PART_SIZE] = b_globalHist[gtid.x + GlobalHistOffset()] +
                b_passHist[gtid.x * e_threadBlocks + gid.x] - exclusiveWaveReduction;
        GroupMemoryBarrierWithGroupSync();
        
        {
            var t = SharedOffsetWGE16(gtid.x);
            for (var i = 0u;
                    i < DS_KEYS_PER_THREAD;
                    i++)
            {
                keys[i] = g_ds[ExtractDigit(g_ds[t]) + PART_SIZE] + t;
                b_alt[keys[i]] = g_ds[t];
                t += WaveGetLaneCount();
            }
        }
        GroupMemoryBarrierWithGroupSync();
            

        {
            var t = DeviceOffsetWGE16(gtid.x, gid.x);

            for (var i = 0u;
                    i < DS_KEYS_PER_THREAD; 
                    i++)
            {
                g_ds[offsets[i]] = b_sortPayload[t];
                t += WaveGetLaneCount();
            }
        }
        GroupMemoryBarrierWithGroupSync();
        
        
        {
            var t = SharedOffsetWGE16(gtid.x);
            for (var i = 0u;
                    i < DS_KEYS_PER_THREAD;
                    i++)
            {
                b_altPayload[keys[i]] = g_ds[t];
                t += WaveGetLaneCount();
            }
        }
    }
    
    //perform the sort on the final partition slightly differently 
    //to handle input sizes not perfect multiples of the partition
    if (gid.x == e_threadBlocks - 1)
    {
        //load the global and pass histogram values into shared memory
        if (gtid.x < RADIX)
        {
            g_ds[gtid.x] = b_globalHist[gtid.x + GlobalHistOffset()] +
                b_passHist[gtid.x * e_threadBlocks + gid.x];
        }
        GroupMemoryBarrierWithGroupSync();
        
        let waveParts = (WaveGetLaneCount() + 31) / 32;
        let partEnd = (e_numKeys + DS_DIM - 1) / DS_DIM * DS_DIM;
        for (var i = gtid.x + gid.x * PART_SIZE; i < partEnd; i += DS_DIM)
        {
            var key = 0u;
            if (i < e_numKeys)
            {
                key = b_sort[i];
            }
            
            var waveFlags = ternary((WaveGetLaneCount() & 31) > 0, (1u << WaveGetLaneCount()) - 1, 0xffffffffu);
            var offset = 0u;
            var bits = 0u;
            if (i < e_numKeys)
            {
                
                for (var k = 0u; k < RADIX_LOG; k++)
                {
                    let t = key >> (k + e_radixShift) & 1u;
                    let ballot = WaveActiveBallot(t > 0);
                    waveFlags &= ternary(t > 0, 0u, 0xffffffffu) ^ ballot;
                }
        
                let wavePart = 0u;
                if (WaveGetLaneIndex() >= wavePart * 32)
                {
                    let ltMask = ternary(WaveGetLaneIndex() >= (wavePart + 1) * 32,
                        0xffffffffu, (1u << (WaveGetLaneIndex() & 31)) - 1);
                    bits += countOneBits(waveFlags & ltMask);
                }
            }
            
            for (var k = 0u; k < DS_DIM / WaveGetLaneCount(); k++)
            {
                if (getWaveIndex(gtid.x) == k && i < e_numKeys) {
                    offset = g_ds[ExtractDigit(key)] + bits;
                }
                GroupMemoryBarrierWithGroupSync();
                
                if (getWaveIndex(gtid.x) == k && i < e_numKeys && bits == 0u)
                {
                    g_ds[ExtractDigit(key)] += countOneBits(waveFlags);
                }
                GroupMemoryBarrierWithGroupSync();
            }

            if (i < e_numKeys)
            {
                b_alt[offset] = key;
                b_altPayload[offset] = b_sortPayload[i];
            }
        }
    }
}