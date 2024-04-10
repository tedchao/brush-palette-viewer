use burn::tensor::Shape;
use burn_jit::JitElement;
use burn_wgpu::JitTensor;

use burn::tensor::ops::IntTensorOps;

use crate::splat_render::create_buffer;
use crate::splat_render::kernels::{SortCount, SortReduce, SortScanAdd, SortScatter};

use super::kernels::SplatKernel;
use super::{generated_bindings, kernels::SortScan, BurnBack, BurnClient, BurnRuntime};

// TODO: Get from bindings
const WG: u32 = 256;
const ELEMENTS_PER_THREAD: u32 = 4;
const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;
const BIN_COUNT: u32 = 16;

pub fn radix_argsort<E: JitElement>(
    client: &BurnClient,
    input_keys: &JitTensor<BurnRuntime, E, 1>,
) -> JitTensor<BurnRuntime, E, 1> {
    let n = input_keys.shape.dims[0] as u32;

    // compute buffer and dispatch sizes
    let num_blocks = (n + (BLOCK_SIZE - 1)) / BLOCK_SIZE;

    let num_wgs = num_blocks;
    let num_blocks_per_wg = num_blocks / num_wgs;
    let num_wgs_with_additional_blocks = num_blocks % num_wgs;

    // I think the else always has the same value, but fix later.
    let num_reduce_wgs = BIN_COUNT
        * if BLOCK_SIZE > num_wgs {
            1
        } else {
            (num_wgs + BLOCK_SIZE - 1) / BLOCK_SIZE
        };

    let num_reduce_wg_per_bin = num_reduce_wgs / BIN_COUNT;

    let device = &input_keys.device;

    let num_blocks = num_blocks as usize;
    let count_buf = BurnBack::int_zeros(Shape::new([num_blocks * 16]), device);
    let reduced_buf = BurnBack::int_zeros(Shape::new([BLOCK_SIZE as usize]), device);

    let output_buf = create_buffer::<E, 1>(client, [n as usize]);
    let output_buf_2 = create_buffer::<E, 1>(client, [n as usize]);

    let mut config = generated_bindings::sorting::Config {
        num_keys: n,
        num_blocks_per_wg,
        num_wgs,
        num_wgs_with_additional_blocks,
        num_reduce_wg_per_bin,
        num_scan_values: num_reduce_wgs,
        shift: 0,
    };

    const N_PASSES: u32 = 8;

    for pass in 0..N_PASSES {
        let (from, to) = if pass == 0 {
            (&input_keys.handle, &output_buf)
        } else if pass % 2 == 0 {
            (&output_buf_2, &output_buf)
        } else {
            (&output_buf, &output_buf_2)
        };

        // The most straightforward way to update the shift amount would be
        // queue.buffer_write, but that has performance problems, so we copy
        // from a pre-initialized buffer.
        // TODO: set config based on pass
        config.shift = pass * 4;

        let wg = generated_bindings::sorting::WG;

        SortCount::execute(
            client,
            config,
            &[from],
            &[&count_buf.handle],
            [config.num_wgs * wg, 1, 1],
        );

        // Todo: in -> out -> out2
        SortReduce::execute(
            client,
            config,
            &[&count_buf.handle],
            &[&reduced_buf.handle],
            [num_reduce_wgs * wg, 1, 1],
        );

        SortScan::execute(client, config, &[], &[&reduced_buf.handle], [1, 1, 1]);

        SortScanAdd::execute(
            client,
            config,
            &[&reduced_buf.handle],
            &[&count_buf.handle],
            [num_reduce_wgs * wg, 1, 1],
        );

        SortScatter::execute(
            client,
            config,
            &[from, &count_buf.handle],
            &[to],
            [config.num_wgs * wg, 1, 1],
        );
    }
    JitTensor::new(
        client.clone(),
        device.clone(),
        Shape::new([n as usize]),
        output_buf_2.clone(),
    )
}

mod tests {
    use crate::splat_render::{radix_sort::radix_argsort, read_buffer_to_u32, BurnBack};
    use burn::tensor::{Int, Tensor};

    #[test]
    fn test_sorting() {
        for i in 0..128 {
            // Look i'm not going to say this is the most sophisticated test :)
            let data = [
                5 + i * 4,
                i,
                6,
                123,
                74657,
                123,
                999,
                2i32.pow(30) + 123,
                6,
                7,
                8,
                0,
                i * 2,
                16 + i,
                128 * i,
            ];
            type Backend = BurnBack;
            let device = Default::default();
            let keys = Tensor::<Backend, 1, Int>::from_ints(data, &device).into_primitive();
            let ret_keys = radix_argsort(&keys.client, &keys);
            let ret_keys = read_buffer_to_u32(&ret_keys.client, &ret_keys.handle);
            let mut sorted = data.to_vec();
            sorted.sort();

            for (val, sort_val) in ret_keys.into_iter().zip(sorted) {
                assert_eq!(val, sort_val as u32)
            }
        }
    }
}
