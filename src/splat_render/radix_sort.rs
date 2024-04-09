use burn::tensor::Shape;
use burn_wgpu::JitTensor;

use burn::tensor::ops::IntTensorOps;

use super::kernels::SplatKernel;
use super::{
    create_tensor, generated_bindings,
    kernels::{SortDownsweep, SortScan, SortUpsweep},
    BurnBack, BurnClient, BurnRuntime,
};
use super::{div_round_up, read_buffer_to_u32};

//The size of our radix in bits
const DEVICE_RADIX_SORT_BITS: usize = 8;

//Number of digits in our radix.
const DEVICE_RADIX_SORT_RADIX: usize = 1 << DEVICE_RADIX_SORT_BITS;

//Number of sorting passes required to sort a 32bit key, KEY_BITS / DEVICE_RADIX_SORT_BITS
const DEVICE_RADIX_SORT_PASSES: usize = 32 / DEVICE_RADIX_SORT_BITS;

//The size of a threadblock partition in the sort
const DEVICE_RADIX_SORT_PARTITION_SIZE: usize = generated_bindings::sorting::PART_SIZE as usize;

pub fn radix_argsort(
    client: &BurnClient,
    input_keys: &JitTensor<BurnRuntime, u32, 1>,
    input_values: &JitTensor<BurnRuntime, u32, 1>,
) -> (
    JitTensor<BurnRuntime, u32, 1>,
    JitTensor<BurnRuntime, u32, 1>,
) {
    let count = input_keys.shape.dims[0];

    //This is threadBlocks * DEVICE_RADIX_SORT_RADIX
    let scratch_buffer_size =
        div_round_up(count, DEVICE_RADIX_SORT_PARTITION_SIZE) * DEVICE_RADIX_SORT_RADIX;
    let reduced_scratch_buffer_size = DEVICE_RADIX_SORT_RADIX * DEVICE_RADIX_SORT_PASSES;

    let device = &input_keys.device;
    let alt_buffer = create_tensor(client, device, [count]);
    let alt_payload_buffer = create_tensor(client, device, [count]);

    let pass_hist_buffer = BurnBack::int_zeros(Shape::new([scratch_buffer_size]), device);
    let global_hist_buffer = BurnBack::int_zeros(Shape::new([reduced_scratch_buffer_size]), device);

    let thread_blocks = div_round_up(count, DEVICE_RADIX_SORT_PARTITION_SIZE) as u32;

    // Execute the sort algorithm in 8-bit increments

    let (mut src_key_buffer, mut src_payload_buffer) = (input_keys, input_values);
    let (mut dst_key_buffer, mut dst_payload_buffer) = (&alt_buffer, &alt_payload_buffer);

    for radix_shift in (0..32).step_by(DEVICE_RADIX_SORT_BITS) {
        let keys = read_buffer_to_u32(client, &src_key_buffer.handle);
        println!("At step {radix_shift} before {:?}", keys);

        //Upsweep
        SortUpsweep::execute(
            client,
            generated_bindings::sort_upsweep::Uniforms::new(radix_shift),
            &[&src_key_buffer.handle],
            &[&global_hist_buffer.handle, &pass_hist_buffer.handle],
            [
                thread_blocks * generated_bindings::sort_upsweep::US_DIM,
                1,
                1,
            ],
        );

        println!(
            "Upsweep {radix_shift} global hist {:?} pass hist {:?}",
            read_buffer_to_u32(client, &global_hist_buffer.handle),
            read_buffer_to_u32(client, &pass_hist_buffer.handle)
        );

        // Scan
        SortScan::execute(
            client,
            (),
            &[&src_key_buffer.handle],
            &[&pass_hist_buffer.handle],
            [
                (DEVICE_RADIX_SORT_RADIX as u32) * generated_bindings::sort_scan::SCAN_DIM,
                1,
                1,
            ],
        );

        println!(
            "Scan {radix_shift} pass hist {:?}",
            read_buffer_to_u32(client, &pass_hist_buffer.handle)
        );

        // Downsweep
        SortDownsweep::execute(
            client,
            generated_bindings::sort_downsweep::Uniforms::new(radix_shift),
            &[
                &src_key_buffer.handle,
                &src_payload_buffer.handle,
                &global_hist_buffer.handle,
                &pass_hist_buffer.handle,
            ],
            &[&dst_key_buffer.handle, &dst_payload_buffer.handle],
            [
                thread_blocks * generated_bindings::sort_downsweep::DS_DIM,
                1,
                1,
            ],
        );

        // Swap if not last pass.
        (src_key_buffer, dst_key_buffer) = (dst_key_buffer, src_key_buffer);
        (src_payload_buffer, dst_payload_buffer) = (dst_payload_buffer, src_payload_buffer);

        let keys = read_buffer_to_u32(client, &src_key_buffer.handle);
        println!("At step {radix_shift} after {:?}", keys);
    }

    (dst_key_buffer.clone(), dst_payload_buffer.clone())
}
