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
#import sorting

const DS_DIM: u32 = 256u;        //The number of threads in a Downsweep threadblock
const DS_KEYS_PER_THREAD: u32 = 15u;     //The number of keys per thread in a Downsweep Threadblock
const MAX_DS_SMEM: u32 = 4096u;   //shared memory for downsweep kernel
const MASK_FULL: u32 = 4294967295;

struct Uniforms {
    radixShift: u32,
}

@group(0) @binding(0) var<storage, read> b_sort: array<u32>;              //buffer to be sorted      
@group(0) @binding(1) var<storage, read> b_sortPayload: array<u32>;       //double buffer

@group(0) @binding(2) var<storage, read> b_globalHist: array<u32>;        //buffer holding global device level offsets  for each digit during a binning pass
@group(0) @binding(3) var<storage, read_write> b_passHist: array<u32>;          //buffer used to store device level offsets for 

@group(0) @binding(4) var<storage, read_write> b_alt: array<u32>;               //payload buffer
@group(0) @binding(5) var<storage, read_write> b_altPayload: array<u32>;        //double buffer payload
@group(0) @binding(6) var<storage, read> info: array<Uniforms>;    
                                                                                //each partition tile for each digit during a binning pass

var<workgroup> g_ds: array<u32, MAX_DS_SMEM>;         //Shared memory for the downsweep kernel

fn SubPartSizeWGE16() -> u32 {
    return DS_KEYS_PER_THREAD * wave::WaveGetLaneCount();
}

fn SharedOffsetWGE16(gtid: u32)  -> u32 {
    return wave::WaveGetLaneIndex() + wave::getWaveIndex(gtid) * SubPartSizeWGE16();
}

fn DeviceOffsetWGE16(gtid: u32, gid: u32)  -> u32 {
    return SharedOffsetWGE16(gtid) + gid * sorting::PART_SIZE;
}

fn WaveHistsSizeWGE16() -> u32 {
    return (DS_DIM / wave::WaveGetLaneCount()) * sorting::RADIX;
}

@compute
@workgroup_size(DS_DIM, 1, 1)
fn main(@builtin(local_invocation_id) gtid: vec3u, @builtin(workgroup_id) gid: vec3u) {
    let info = info[0];
    let e_radixShift = info.radixShift;
    wave::init(gtid.x);

    let e_numKeys: u32 = arrayLength(&b_sort);
    let e_threadBlocks: u32 = (e_numKeys + sorting::PART_SIZE - 1) / sorting::PART_SIZE;

    if (gid.x < e_threadBlocks - 1)
    {
        var keys = array<u32, DS_KEYS_PER_THREAD>();
        var offsets = array<u32, DS_KEYS_PER_THREAD>();

        //Load keys into registers
        {
            var t = DeviceOffsetWGE16(gtid.x, gid.x);
            for (var i = 0u; i < DS_KEYS_PER_THREAD; i++)
            {
                keys[i] = b_sort[t];
                t += wave::WaveGetLaneCount();
            }
        }
        
        //Clear histogram memory
        for (var i = gtid.x; i < WaveHistsSizeWGE16(); i += DS_DIM) {
            g_ds[i] = 0u;
        }
        workgroupBarrier();

        //Warp Level Multisplit
        for (var i = 0u; i < DS_KEYS_PER_THREAD; i++)
        {
            var waveFlags = sorting::ternary((wave::WaveGetLaneCount() & 31) > 0,
                    (1u << wave::WaveGetLaneCount()) - 1u, MASK_FULL);

            for (var k = 0u; k < sorting::RADIX_LOG; k++)
            {
                let t = (keys[i] >> (k + e_radixShift)) & 1u;
                let ballot = wave::WaveActiveBallot(t > 0);
                waveFlags &= sorting::ternary(t > 0, 0u, MASK_FULL) ^ ballot;
            }

            var bits = 0u;
            let ltMask = (1u << (wave::WaveGetLaneIndex() & 31)) - 1u;
            bits += countOneBits(waveFlags & ltMask);
                
            let index = sorting::ExtractDigit(keys[i], e_radixShift) + (wave::getWaveIndex(gtid.x) * sorting::RADIX);
            offsets[i] = g_ds[index] + bits;
            workgroupBarrier();
            if (bits == 0) {
                g_ds[index] += countOneBits(waveFlags);
            }
            workgroupBarrier();
        }
        
        //inclusive/exclusive prefix sum up the histograms
        //followed by exclusive prefix sum across the reductions
        var reduction = g_ds[gtid.x];
        for (var i = gtid.x + sorting::RADIX; i < WaveHistsSizeWGE16(); i += sorting::RADIX) {
            reduction += g_ds[i];
            g_ds[i] = reduction - g_ds[i];
        }
        
        reduction += wave::WavePrefixSum(reduction);
        workgroupBarrier();

        let laneMask = wave::WaveGetLaneCount() - 1;
        g_ds[((wave::WaveGetLaneIndex() + 1) & laneMask) + (gtid.x & ~laneMask)] = reduction;
        workgroupBarrier();
            
        if (gtid.x < sorting::RADIX / wave::WaveGetLaneCount())
        {
            g_ds[gtid.x * wave::WaveGetLaneCount()] =
                wave::WavePrefixSum(g_ds[gtid.x * wave::WaveGetLaneCount()]);
        }
        workgroupBarrier();
            
        if (wave::WaveGetLaneIndex() > 0) {
            g_ds[gtid.x] += wave::WaveReadLaneAt(g_ds[gtid.x - 1u], 1u);
        }
        workgroupBarrier();
    
        //Update offsets
        if (gtid.x >= wave::WaveGetLaneCount())
        {
            let t = wave::getWaveIndex(gtid.x) * sorting::RADIX;
            for (var i = 0u; i < DS_KEYS_PER_THREAD; i++)
            {
                let t2 = sorting::ExtractDigit(keys[i], e_radixShift);
                offsets[i] += g_ds[t2 + t] + g_ds[t2];
            }
        }
        else
        {
            for (var i = 0u; i < DS_KEYS_PER_THREAD; i++) {
                offsets[i] += g_ds[sorting::ExtractDigit(keys[i], e_radixShift)];
            }
        }
        
        //take advantage of barrier
        let exclusiveWaveReduction = g_ds[gtid.x];
        workgroupBarrier();
        
        //scatter keys into shared memory
        for (var i = 0u; i < DS_KEYS_PER_THREAD; i++) {
            g_ds[offsets[i]] = keys[i];
        }

        g_ds[gtid.x + sorting::PART_SIZE] = b_globalHist[gtid.x + sorting::GlobalHistOffset(e_radixShift)] +
                b_passHist[gtid.x * e_threadBlocks + gid.x] - exclusiveWaveReduction;
        workgroupBarrier();
        
        {
            var t = SharedOffsetWGE16(gtid.x);
            for (var i = 0u; i < DS_KEYS_PER_THREAD; i++) {
                keys[i] = g_ds[sorting::ExtractDigit(g_ds[t], e_radixShift) + sorting::PART_SIZE] + t;
                b_alt[keys[i]] = g_ds[t];
                t += wave::WaveGetLaneCount();
            }
        }
        workgroupBarrier();
        
        {
            var t = DeviceOffsetWGE16(gtid.x, gid.x);
            for (var i = 0u; i < DS_KEYS_PER_THREAD; i++) {
                g_ds[offsets[i]] = b_sortPayload[t];
                t += wave::WaveGetLaneCount();
            }
            workgroupBarrier();
        }
        
        {
            var t = SharedOffsetWGE16(gtid.x);
            for (var i = 0u; i < DS_KEYS_PER_THREAD; i++)
            {
                b_altPayload[keys[i]] = g_ds[t];
                t += wave::WaveGetLaneCount();
            }
        }
    }
    else
    {
        //perform the sort on the final partition slightly differently 
        //to handle input sizes not perfect multiples of the partition

        //load the global and pass histogram values into shared memory
        if (gtid.x < sorting::RADIX)
        {
            g_ds[gtid.x] = b_globalHist[gtid.x + sorting::GlobalHistOffset(e_radixShift)] +
                b_passHist[gtid.x * e_threadBlocks + gid.x];
        }
        workgroupBarrier();
        
        let partEnd = ((e_numKeys + DS_DIM - 1) / DS_DIM) * DS_DIM;
        for (var i = gtid.x + gid.x * sorting::PART_SIZE; i < partEnd; i += DS_DIM)
        {
            var key = 0u;
            if (i < e_numKeys) {
                key = b_sort[i];
            }
            
            var waveFlags = MASK_FULL;
            var offset = 0u;
            var bits = 0u;
            
            if (i < e_numKeys)
            {
                for (var k = 0u; k < sorting::RADIX_LOG; k++)
                {
                    let t = (key >> (k + e_radixShift)) & 1u;
                    let ballot = wave::WaveActiveBallot(t > 0);
                    waveFlags &= sorting::ternary(t > 0, 0u, MASK_FULL) ^ ballot;
                }
            
                let ltMask = (1u << (wave::WaveGetLaneIndex() & 31)) - 1;
                bits += countOneBits(waveFlags & ltMask);
            }

            for (var k = 0u; k < DS_DIM / wave::WaveGetLaneCount(); k++)
            {
                if (wave::getWaveIndex(gtid.x) == k && i < e_numKeys) {
                    offset = g_ds[sorting::ExtractDigit(key, e_radixShift)] + bits;
                }
                workgroupBarrier();
                
                if (wave::getWaveIndex(gtid.x) == k && i < e_numKeys && bits == 0)
                {
                    g_ds[sorting::ExtractDigit(key, e_radixShift)] += countOneBits(waveFlags);
                }
                workgroupBarrier();
            }

            if (i < e_numKeys)
            {
                b_alt[offset] = key;
                b_altPayload[offset] = b_sortPayload[i];
            }
        }
    }
}