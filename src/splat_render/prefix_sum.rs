use burn_jit::JitElement;
use burn_wgpu::JitTensor;
use tracing::info_span;

use super::{
    create_tensor, div_round_up, generated_bindings,
    kernels::{PrefixSumAddScannedSums, PrefixSumScan, PrefixSumScanSums, SplatKernel},
    BurnClient, BurnRuntime,
};

pub fn prefix_sum<E: JitElement>(
    client: &BurnClient,
    input: &JitTensor<BurnRuntime, E, 1>,
    sync: bool,
) -> JitTensor<BurnRuntime, E, 1> {
    let _span = info_span!("prefix sum");

    let threads_per_group = generated_bindings::prefix_sum_helpers::THREADS_PER_GROUP as usize;
    let num = input.shape.dims[0];
    let outputs = create_tensor(client, &input.device, input.shape.dims);

    // 1. Per group scan
    // N = num.
    PrefixSumScan::new().execute(
        client,
        (),
        &[&input.handle],
        &[&outputs.handle],
        [num as u32, 1, 1],
        sync,
    );

    if num < threads_per_group {
        return outputs;
    }

    let size = (num as f32 * 1.5) as usize;
    let mut group_buffer = vec![];
    let mut work_size = vec![];
    let mut work_sz = size;
    while work_sz > threads_per_group {
        work_sz = div_round_up(work_sz, threads_per_group);
        group_buffer.push(create_tensor::<E, 1>(client, &input.device, [work_sz]));
        work_size.push(work_sz);
    }

    PrefixSumScanSums::new().execute(
        client,
        (),
        &[&outputs.handle],
        &[&group_buffer[0].handle],
        [work_size[0] as u32, 1, 1],
        sync,
    );

    // Continue down the pyramid
    for l in 0..(group_buffer.len() - 1) {
        PrefixSumScanSums::new().execute(
            client,
            (),
            &[&group_buffer[l].handle],
            &[&group_buffer[l + 1].handle],
            [work_size[l + 1] as u32, 1, 1],
            sync,
        );
    }

    for l in (1..group_buffer.len()).rev() {
        let work_sz = work_size[l - 1];
        PrefixSumAddScannedSums::new().execute(
            client,
            (),
            &[&group_buffer[l].handle],
            &[&group_buffer[l - 1].handle],
            [work_sz as u32, 1, 1],
            sync,
        );
    }

    PrefixSumAddScannedSums::new().execute(
        client,
        (),
        &[&group_buffer[0].handle],
        &[&outputs.handle],
        [(work_size[0] * threads_per_group) as u32, 1, 1],
        sync,
    );

    outputs
}

mod tests {
    #[allow(unused_imports)]
    use crate::splat_render::{
        prefix_sum::prefix_sum, radix_sort::radix_argsort, read_buffer_to_u32, BurnBack,
    };
    #[allow(unused_imports)]
    use burn::tensor::{Int, Tensor};

    #[test]
    fn test_sum() {
        const ITERS: usize = 512 * 16;
        let mut data = vec![];
        for i in 0..ITERS {
            data.push(2 + i as i32);
            data.push(0);
        }

        type Backend = BurnBack;
        let device = Default::default();
        let keys = Tensor::<Backend, 1, Int>::from_data(data.as_slice(), &device).into_primitive();

        let summed = prefix_sum(&keys.client, &keys, false);
        let summed = read_buffer_to_u32(&summed.client, &summed.handle);

        let prefix_sum_ref: Vec<_> = data
            .into_iter()
            .scan(0, |x, y| {
                *x += y;
                Some(*x)
            })
            .collect();

        for (summed, reff) in summed.into_iter().zip(prefix_sum_ref) {
            assert_eq!(summed, reff as u32)
        }
    }
}
