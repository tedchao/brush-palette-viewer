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
#import wave

@group(0) @binding(0) var<storage, read> b_sort: array<u32>;                            //buffer to be sorted      
@group(0) @binding(4) var<storage, read_write> b_globalHist: array<atomic<u32>>;        //buffer holding global device level offsets  for each digit during a binning pass
@group(0) @binding(5) var<storage, read_write> b_passHist: array<u32>;                  //buffer used to store device level offsets for 
                                                                                        //each partition tile for each digit during a binning pass
@group(0) @binding(6) var<storage, read> info: array<sorting::Uniforms>;                //buffer to be sorted      

var<workgroup> g_us: array<atomic<u32>, sorting::RADIX_2>;           //Shared memory for upsweep kernel

const US_DIM: u32 = 128u;        //The number of threads in a Upsweep threadblock

@compute
@workgroup_size(US_DIM, 1, 1)
fn Upsweep(@builtin(local_invocation_id) gtid: vec3u, @builtin(workgroup_id) gid: vec3u) {
    let info = info[0];
    let radixShift = info.radixShift;

    wave::init(gtid.x);
    let e_numKeys: u32 = arrayLength(&b_sort);
    let e_threadBlocks: u32 = (e_numKeys + sorting::PART_SIZE - 1) / sorting::PART_SIZE;

    // Clear shared memory.
    // TODO: Not needed in WGSL - specced as 0 init?
    let histsEnd: u32 = sorting::RADIX * 2;
    for (var i = gtid.x; i < histsEnd; i += US_DIM) {
        g_us[i] = 0u;
    }
    workgroupBarrier();

    //histogram, 64 threads to a histogram
    let histOffset = gtid.x / 64 * sorting::RADIX;

    let partitionEnd = sorting::ternary(gid.x == e_threadBlocks - 1, e_numKeys, (gid.x + 1) * sorting::PART_SIZE);

    for (var i = gtid.x + gid.x * sorting::PART_SIZE; i < partitionEnd; i += US_DIM) {
        atomicAdd(&g_us[sorting::ExtractDigit(b_sort[i], radixShift) + histOffset], 1u);
    }
    workgroupBarrier();
    
    //reduce and pass to tile histogram
    for (var i = gtid.x; i < sorting::RADIX; i += US_DIM)
    {
        g_us[i] += g_us[i + sorting::RADIX];
        b_passHist[i * e_threadBlocks + gid.x] = g_us[i];
    }
    
    for (var i = gtid.x; i < sorting::RADIX; i += US_DIM) {
        g_us[i] += wave::WavePrefixSum(g_us[i]);
    }
    workgroupBarrier();
    
    if (gtid.x < (sorting::RADIX / wave::WaveGetLaneCount())) {
        g_us[(gtid.x + 1) * wave::WaveGetLaneCount() - 1] += wave::WavePrefixSum(g_us[(gtid.x + 1u) * wave::WaveGetLaneCount() - 1u]);
    }
    workgroupBarrier();

    //atomically add to global histogram
    let globalHistOffset = sorting::GlobalHistOffset(radixShift);
    let laneMask = wave::WaveGetLaneCount() - 1;
    let circularLaneShift = wave::WaveGetLaneIndex() + 1 & laneMask;

    for (var i = gtid.x; i < sorting::RADIX; i += US_DIM) {
        let index = circularLaneShift + (i & ~laneMask);
        atomicAdd(&b_globalHist[index + globalHistOffset],
            sorting::ternary(wave::WaveGetLaneIndex() != laneMask, g_us[i], 0u) +
            sorting::ternary(i >= wave::WaveGetLaneCount(), wave::WaveReadLaneAt(g_us[i - 1], 0u), 0u)
        );
    }
}
