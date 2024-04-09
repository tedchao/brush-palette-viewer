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

const SCAN_DIM: u32 = 128u;        //The number of threads in a Scan threadblock
@group(0) @binding(0) var<storage, read> b_sort: array<u32>;                    //buffer to be sorted      
@group(0) @binding(1) var<storage, read_write> b_passHist: array<u32>;          //buffer used to store device level offsets for 

var<workgroup> g_scan: array<u32, SCAN_DIM>;          //Shared memory for the scan kernel

//Scan along the spine of the upsweep
@compute
@workgroup_size(SCAN_DIM, 1, 1)
fn main(@builtin(local_invocation_id) gtid: vec3u, @builtin(workgroup_id) gid: vec3u) {
    wave::init(gtid.x);

    let e_numKeys = arrayLength(&b_sort);
    let e_threadBlocks = (e_numKeys + sorting::PART_SIZE - 1) / sorting::PART_SIZE;

    var aggregate = 0u;
    let laneMask = wave::WaveGetLaneCount() - 1u;
    let circularLaneShift = (wave::WaveGetLaneIndex() + 1) & laneMask;
    let partionsEnd = e_threadBlocks / SCAN_DIM * SCAN_DIM;
    let offset = gid.x * e_threadBlocks;
    var i = gtid.x;
    for (; i < partionsEnd; i += SCAN_DIM)
    {
        g_scan[gtid.x] = b_passHist[i + offset];
        g_scan[gtid.x] += wave::WavePrefixSum(g_scan[gtid.x]);
        workgroupBarrier();
        storageBarrier();
        
        if (gtid.x < SCAN_DIM / wave::WaveGetLaneCount())
        {
            g_scan[(gtid.x + 1) * wave::WaveGetLaneCount() - 1] +=
                wave::WavePrefixSum(g_scan[(gtid.x + 1) * wave::WaveGetLaneCount() - 1]);
        }
        workgroupBarrier();
        storageBarrier();

        
        var passHist = aggregate;

        if wave::WaveGetLaneIndex() != laneMask {
            passHist += g_scan[gtid.x];
        }

        if gtid.x >= wave::WaveGetLaneCount() {
            passHist += wave::WaveReadLaneAt(g_scan[gtid.x - 1], 0u);
        }

        b_passHist[circularLaneShift + (i & ~laneMask) + offset] = passHist;

        aggregate += g_scan[SCAN_DIM - 1];
        workgroupBarrier();
        storageBarrier();

    }
    
    //partial
    if (i < e_threadBlocks) {
        g_scan[gtid.x] = b_passHist[offset + i];
    }
    g_scan[gtid.x] += wave::WavePrefixSum(g_scan[gtid.x]);
    workgroupBarrier();
    storageBarrier();
        
    if (gtid.x < SCAN_DIM / wave::WaveGetLaneCount())
    {
        g_scan[(gtid.x + 1) * wave::WaveGetLaneCount() - 1] +=
            wave::WavePrefixSum(g_scan[(gtid.x + 1) * wave::WaveGetLaneCount() - 1]);
    }
    workgroupBarrier();
    storageBarrier();
    
    let index = circularLaneShift + (i & ~laneMask);
    if (index < e_threadBlocks)
    {
        var passHist = aggregate;

        if wave::WaveGetLaneIndex() != laneMask {
            passHist += g_scan[gtid.x];
        }

        if gtid.x >= wave::WaveGetLaneCount() {
            passHist += g_scan[(gtid.x & ~laneMask) - 1];
        }

        b_passHist[index + offset] = passHist;
    }
}