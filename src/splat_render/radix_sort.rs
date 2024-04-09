use burn_jit::JitElement;
use burn_wgpu::JitTensor;

use super::{
    create_tensor, generated_bindings,
    kernels::{PrefixSumAddScannedSums, PrefixSumScan, PrefixSumScanSums, SplatKernel},
    BurnClient, BurnRuntime,
};

pub fn radix_argsort<E: JitElement>(
    client: &BurnClient,
    input: &JitTensor<BurnRuntime, E, 1>,
) -> JitTensor<BurnRuntime, E, 1> {
    let threads_per_group = generated_bindings::prefix_sum_helpers::THREADS_PER_GROUP;
    let num = input.shape.dims[0] as u32;
    let outputs = create_tensor(client, &input.device, input.shape.dims);

    // 1. Per group scan
    // N = num.
    PrefixSumScan::execute(
        client,
        (),
        &[&input.handle],
        &[&outputs.handle],
        [num, 1, 1],
    );

    if num < threads_per_group {
        return outputs;
    }

    let mut group_buffer = vec![];
    let mut work_sz = num;

    while work_sz > threads_per_group {
        work_sz = num_groups(work_sz, threads_per_group);
        group_buffer.push(create_tensor::<E, 1>(
            client,
            &input.device,
            [work_sz as usize],
        ));
    }

    PrefixSumScanSums::execute(
        client,
        (),
        &[&outputs.handle],
        &[&group_buffer[0].handle],
        [group_buffer[0].shape.dims[0] as u32, 1, 1],
    );

    // Continue down the pyramid
    for l in 0..(group_buffer.len() - 1) {
        let ouput = &group_buffer[l + 1];
        PrefixSumScanSums::execute(
            client,
            (),
            &[&group_buffer[l].handle],
            &[&ouput.handle],
            [ouput.shape.dims[0] as u32, 1, 1],
        );
    }

    for l in (0..(group_buffer.len() - 1)).rev() {
        let ouput = &group_buffer[l - 1];

        PrefixSumAddScannedSums::execute(
            client,
            (),
            &[&group_buffer[l].handle],
            &[&ouput.handle],
            [ouput.shape.dims[0] as u32, 1, 1],
        );
    }

    PrefixSumAddScannedSums::execute(
        client,
        (),
        &[&group_buffer[0].handle],
        &[&outputs.handle],
        [num, 1, 1],
    );

    outputs
}
