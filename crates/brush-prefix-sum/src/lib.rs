mod shaders;

use crate::shaders::prefix_sum_scan;
use brush_kernel::create_tensor;
use brush_kernel::kernel_source_gen;
use brush_kernel::BurnRuntime;
use brush_kernel::SplatKernel;
use shaders::prefix_sum_add_scanned_sums;
use shaders::prefix_sum_scan_sums;

kernel_source_gen!(PrefixSumScan {}, prefix_sum_scan, ());
kernel_source_gen!(PrefixSumScanSums {}, prefix_sum_scan_sums, ());
kernel_source_gen!(PrefixSumAddScannedSums {}, prefix_sum_add_scanned_sums, ());

use burn_wgpu::JitTensor;
use tracing::info_span;

pub fn prefix_sum(input: JitTensor<BurnRuntime, u32, 1>) -> JitTensor<BurnRuntime, u32, 1> {
    let _span = info_span!("prefix sum");

    let threads_per_group = shaders::prefix_sum_helpers::THREADS_PER_GROUP as usize;
    let num = input.shape.dims[0];
    let client = &input.client;
    let outputs = create_tensor(input.shape.dims, &input.device, client);

    PrefixSumScan::new().execute(
        client,
        (),
        &[input.handle.binding()],
        &[outputs.handle.clone().binding()],
        [num as u32, 1, 1],
    );

    if num < threads_per_group {
        return outputs;
    }

    let size = (num as f32 * 1.5) as usize;
    let mut group_buffer = vec![];
    let mut work_size = vec![];
    let mut work_sz = size;
    while work_sz > threads_per_group {
        work_sz = work_sz.div_ceil(threads_per_group);
        group_buffer.push(create_tensor::<u32, 1>([work_sz], &input.device, client));
        work_size.push(work_sz);
    }

    PrefixSumScanSums::new().execute(
        client,
        (),
        &[outputs.handle.clone().binding()],
        &[group_buffer[0].handle.clone().binding()],
        [work_size[0] as u32],
    );

    for l in 0..(group_buffer.len() - 1) {
        PrefixSumScanSums::new().execute(
            client,
            (),
            &[group_buffer[l].handle.clone().binding()],
            &[group_buffer[l + 1].handle.clone().binding()],
            [work_size[l + 1] as u32],
        );
    }

    for l in (1..group_buffer.len()).rev() {
        let work_sz = work_size[l - 1];
        PrefixSumAddScannedSums::new().execute(
            client,
            (),
            &[group_buffer[l].handle.clone().binding()],
            &[group_buffer[l - 1].handle.clone().binding()],
            [work_sz as u32],
        );
    }

    PrefixSumAddScannedSums::new().execute(
        client,
        (),
        &[group_buffer[0].handle.clone().binding()],
        &[outputs.handle.clone().binding()],
        [(work_size[0] * threads_per_group) as u32, 1, 1],
    );

    outputs
}

#[cfg(test)]
mod tests {
    use crate::prefix_sum;
    use brush_kernel::{read_buffer_as_u32, BurnBack};
    use burn::tensor::{Int, Tensor};
    use burn_wgpu::JitTensor;

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
        let keys = JitTensor::new(keys.client.clone(), keys.device, keys.shape, keys.handle);
        let summed = prefix_sum(keys.clone());
        let summed = read_buffer_as_u32(&keys.client, summed.handle.binding());

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
