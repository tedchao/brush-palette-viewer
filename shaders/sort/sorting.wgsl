#import wave
const PART_SIZE: u32 = 3840u;       //size of a partition tile

const RADIX: u32 = 256u;        //Number of digit bins
const RADIX_2: u32 = RADIX * 2;

const RADIX_MASK: u32 = 255u;        //Mask of digit bins
const RADIX_LOG: u32 = 8u;          //log2(RADIX)

const HALF_RADIX: u32 = 128u;        //For smaller waves where bit packing is necessary
const HALF_MASK: u32 = 127u;        // '' 

struct Uniforms {
    radixShift: u32,
}

fn ExtractDigit(key: u32, radixShift: u32) -> u32 {
    return key >> radixShift & RADIX_MASK;
}

fn GlobalHistOffset(radixShift: u32)  -> u32 {
    return radixShift << 5u;
}

fn ternary(cond: bool, on_true: u32, on_false: u32) -> u32 {
    return select(on_false, on_true, cond);
}
