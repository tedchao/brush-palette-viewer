// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.
struct Config {
    num_keys: u32,
    num_blocks_per_wg: u32,
    num_wgs: u32,
    num_wgs_with_additional_blocks: u32,
    num_reduce_wg_per_bin: u32,
    num_scan_values: u32,
    shift: u32,
}

const OFFSET = 42u;
const WG = 256u;
const BITS_PER_PASS = 4u;
const BIN_COUNT = 1u << BITS_PER_PASS;
const HISTOGRAM_SIZE = WG * BIN_COUNT;
const ELEMENTS_PER_THREAD = 4u;
const BLOCK_SIZE = WG * ELEMENTS_PER_THREAD;

// An implementation of warp-local multisplit. See the Onesweep paper.
// Since WGSL doesn't yet have subgroups, we use the terms "warp" and "lane"
// to describe the arrangement of data, and simulate the warp ballot operation
// using simple iteration across values in shared memory.
const WARP_SIZE = 16u;
const N_WARPS = WG / WARP_SIZE;