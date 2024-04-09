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

#import wave

const PART_SIZE: u32 = 3840u;       //size of a partition tile

const RADIX: u32 = 256u;        //Number of digit bins
const RADIX_MASK: u32 = 255u;        //Mask of digit bins
const RADIX_LOG: u32 = 8u;          //log2(RADIX)

fn ExtractDigit(key: u32, radixShift: u32) -> u32 {
    return (key >> radixShift) & RADIX_MASK;
}

fn GlobalHistOffset(radixShift: u32)  -> u32 {
    return radixShift << 5u;
}

fn ternary(cond: bool, on_true: u32, on_false: u32) -> u32 {
    return select(on_false, on_true, cond);
}
