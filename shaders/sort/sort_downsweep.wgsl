/******************************************************************************
 * GPUSorting
 * Device Level 8-bit LSD Radix Sort using reduce then scan
 *
 * SPDX-License-Identifier: MIT
 * Author:  Thomas Smith 3/13/2024
 * https://github.com/b0nes164/GPUSorting
 * 
 ******************************************************************************/
#import sorting

@compute
@workgroup_size(sorting::DS_DIM, 1, 1)
fn main(
    @builtin(local_invocation_id) gtid: vec3u,
    @builtin(workgroup_id) gid: vec3u
) {
    sorting::Downsweep(gtid, gid);
}