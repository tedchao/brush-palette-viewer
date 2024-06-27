mod shaders;

use brush_kernel::create_tensor;
use brush_kernel::kernel_source_gen;
use brush_kernel::SplatKernel;
use burn_wgpu::WgpuRuntime;
use shaders::prefix_sum_add_scanned_sums;
use shaders::prefix_sum_scan;
use shaders::prefix_sum_scan_sums;

kernel_source_gen!(PrefixSumScan {}, prefix_sum_scan);
kernel_source_gen!(PrefixSumScanSums {}, prefix_sum_scan_sums);
kernel_source_gen!(PrefixSumAddScannedSums {}, prefix_sum_add_scanned_sums);

use burn_wgpu::JitTensor;
use tracing::info_span;

pub fn prefix_sum(input: JitTensor<WgpuRuntime, u32, 1>) -> JitTensor<WgpuRuntime, u32, 1> {
    let _span = info_span!("prefix sum");

    let threads_per_group = shaders::prefix_sum_helpers::THREADS_PER_GROUP as usize;
    let num = input.shape.dims[0];
    let client = &input.client;
    let outputs = create_tensor(input.shape.dims, &input.device, client);

    PrefixSumScan::new().execute(
        client,
        &[input.handle.binding(), outputs.handle.clone().binding()],
        [num as u32, 1, 1],
    );

    if num < threads_per_group {
        return outputs;
    }

    let mut group_buffer = vec![];
    let mut work_size = vec![];
    let mut work_sz = num;
    while work_sz > threads_per_group {
        work_sz = work_sz.div_ceil(threads_per_group);
        group_buffer.push(create_tensor::<u32, 1, WgpuRuntime>(
            [work_sz],
            &input.device,
            client,
        ));
        work_size.push(work_sz);
    }

    PrefixSumScanSums::new().execute(
        client,
        &[
            outputs.handle.clone().binding(),
            group_buffer[0].handle.clone().binding(),
        ],
        [work_size[0] as u32],
    );

    for l in 0..(group_buffer.len() - 1) {
        PrefixSumScanSums::new().execute(
            client,
            &[
                group_buffer[l].handle.clone().binding(),
                group_buffer[l + 1].handle.clone().binding(),
            ],
            [work_size[l + 1] as u32],
        );
    }

    for l in (1..group_buffer.len()).rev() {
        let work_sz = work_size[l - 1];
        PrefixSumAddScannedSums::new().execute(
            client,
            &[
                group_buffer[l].handle.clone().binding(),
                group_buffer[l - 1].handle.clone().binding(),
            ],
            [work_sz as u32],
        );
    }

    PrefixSumAddScannedSums::new().execute(
        client,
        &[
            group_buffer[0].handle.clone().binding(),
            outputs.handle.clone().binding(),
        ],
        [(work_size[0] * threads_per_group) as u32, 1, 1],
    );

    outputs
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use crate::prefix_sum;
    use brush_kernel::bitcast_tensor;
    use burn::tensor::{Int, Tensor};
    use burn_wgpu::{JitBackend, JitTensor, WgpuRuntime};

    #[test]
    fn test_sum_tiny() {
        type Backend = JitBackend<WgpuRuntime, f32, i32>;
        let device = Default::default();
        let keys = Tensor::<Backend, 1, Int>::from_data([1, 1, 1, 1], &device).into_primitive();
        let keys = JitTensor::new(keys.client.clone(), keys.device, keys.shape, keys.handle);
        let summed = prefix_sum(keys.clone());
        let summed = Tensor::<Backend, 1, Int>::from_primitive(bitcast_tensor(summed)).to_data();
        let summed = summed.as_slice::<i32>().unwrap();
        assert_eq!(summed.len(), 4);
        assert_eq!(summed, [1, 2, 3, 4])
    }

    #[test]
    fn test_512_multiple() {
        const ITERS: usize = 1024;
        let mut data = vec![];
        for i in 0..ITERS {
            data.push(90 + i as i32);
        }
        type Backend = JitBackend<WgpuRuntime, f32, i32>;
        let device = Default::default();
        let keys = Tensor::<Backend, 1, Int>::from_data(data.as_slice(), &device).into_primitive();
        let keys = JitTensor::new(keys.client.clone(), keys.device, keys.shape, keys.handle);
        let summed = prefix_sum(keys.clone());
        let summed = Tensor::<Backend, 1, Int>::from_primitive(bitcast_tensor(summed)).to_data();
        let prefix_sum_ref: Vec<_> = data
            .into_iter()
            .scan(0, |x, y| {
                *x += y;
                Some(*x)
            })
            .collect();
        for (summed, reff) in summed.as_slice::<i32>().unwrap().iter().zip(prefix_sum_ref) {
            assert_eq!(*summed, reff)
        }
    }

    #[test]
    fn test_sum() {
        const ITERS: usize = 512 * 16;
        let mut data = vec![];
        for i in 0..ITERS {
            data.push(2 + i as i32);
            data.push(0);
            data.push(32);
            data.push(512);
            data.push(30965);
        }

        type Backend = JitBackend<WgpuRuntime, f32, i32>;
        let device = Default::default();
        let keys = Tensor::<Backend, 1, Int>::from_data(data.as_slice(), &device).into_primitive();
        let keys = JitTensor::new(keys.client.clone(), keys.device, keys.shape, keys.handle);
        let summed = prefix_sum(keys.clone());
        let summed = Tensor::<Backend, 1, Int>::from_primitive(bitcast_tensor(summed)).to_data();

        let prefix_sum_ref: Vec<_> = data
            .into_iter()
            .scan(0, |x, y| {
                *x += y;
                Some(*x)
            })
            .collect();

        for (summed, reff) in summed.as_slice::<i32>().unwrap().iter().zip(prefix_sum_ref) {
            assert_eq!(*summed, reff)
        }
    }
}
