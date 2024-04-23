use burn::tensor::Shape;
use burn_wgpu::JitTensor;

use burn::tensor::ops::IntTensorOps;
use tracing::info_span;

use crate::splat_render::create_tensor;
use crate::splat_render::kernels::{SortCount, SortReduce, SortScanAdd, SortScatter};

use super::kernels::SplatKernel;
use super::shaders;
use super::{kernels::SortScan, BurnBack, BurnClient, BurnRuntime};

const WG: u32 = shaders::sorting::WG;
const ELEMENTS_PER_THREAD: u32 = shaders::sorting::ELEMENTS_PER_THREAD;
const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;
const BIN_COUNT: u32 = shaders::sorting::BIN_COUNT;

pub fn radix_argsort(
    client: BurnClient,
    input_keys: JitTensor<BurnRuntime, u32, 1>,
    input_values: JitTensor<BurnRuntime, u32, 1>,
) -> (
    JitTensor<BurnRuntime, u32, 1>,
    JitTensor<BurnRuntime, u32, 1>,
) {
    let _span = info_span!("Radix sort").entered();

    let n = input_keys.shape.dims[0] as u32;
    assert_eq!(input_keys.shape.dims[0], input_values.shape.dims[0]);

    // compute buffer and dispatch sizes
    let num_blocks = n.div_ceil(BLOCK_SIZE);

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
    let reduced_buf = BurnBack::int_zeros(Shape::new([(BLOCK_SIZE) as usize]), device);

    let output_keys = create_tensor::<u32, 1>(&client, device, [n as usize]);
    let output_values = create_tensor::<u32, 1>(&client, device, [n as usize]);
    let output_keys_swap = create_tensor::<u32, 1>(&client, device, [n as usize]);
    let output_values_swap = create_tensor::<u32, 1>(&client, device, [n as usize]);

    let mut config = shaders::sorting::Config {
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
        let ((from, to), (from_val, to_val)) = if pass == 0 {
            (
                (&input_keys, &output_keys_swap),
                (&input_values, &output_values_swap),
            )
        } else if pass % 2 == 0 {
            (
                (&output_keys, &output_keys_swap),
                (&output_values, &output_values_swap),
            )
        } else {
            (
                (&output_keys_swap, &output_keys),
                (&output_values_swap, &output_values),
            )
        };

        // The most straightforward way to update the shift amount would be
        // queue.buffer_write, but that has performance problems, so we copy
        // from a pre-initialized buffer.
        // TODO: set config based on pass
        config.shift = pass * 4;

        let wg = shaders::sorting::WG;

        SortCount::new().execute(
            &client,
            config,
            &[from.clone().handle.binding()],
            &[count_buf.clone().handle.binding()],
            [config.num_wgs * wg, 1, 1],
        );

        // Todo: in -> out -> out2
        SortReduce::new().execute(
            &client,
            config,
            &[count_buf.clone().handle.binding()],
            &[reduced_buf.clone().handle.binding()],
            [num_reduce_wgs * wg, 1, 1],
        );

        SortScan::new().execute(
            &client,
            config,
            &[],
            &[reduced_buf.clone().handle.binding()],
            [1, 1, 1],
        );

        SortScanAdd::new().execute(
            &client,
            config,
            &[reduced_buf.clone().handle.binding()],
            &[count_buf.clone().handle.binding()],
            [num_reduce_wgs * wg, 1, 1],
        );

        SortScatter::new().execute(
            &client,
            config,
            &[
                from.handle.clone().binding(),
                from_val.handle.clone().binding(),
                count_buf.handle.clone().binding(),
            ],
            &[to.handle.clone().binding(), to_val.handle.clone().binding()],
            [config.num_wgs * wg, 1, 1],
        );
    }
    (output_keys, output_values)
}

mod tests {
    #[allow(unused_imports)]
    use std::{
        fs::File,
        io::{BufReader, Read},
    };

    #[allow(unused_imports)]
    use crate::splat_render::{
        bitcast_tensor, radix_sort::radix_argsort, read_buffer_to_u32, BurnBack,
    };

    #[allow(unused_imports)]
    use burn::{
        backend::Autodiff,
        tensor::{Int, Tensor},
    };

    pub fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
        let mut indices = (0..data.len()).collect::<Vec<_>>();
        indices.sort_by_key(|&i| &data[i]);
        indices
    }

    #[test]
    fn test_sorting() {
        for i in 0..128 {
            let keys_inp = [
                5 + i * 4,
                i,
                6,
                123,
                74657,
                123,
                999,
                2i32.pow(24) + 123,
                6,
                7,
                8,
                0,
                i * 2,
                16 + i,
                128 * i,
            ];

            let values_inp: Vec<_> = keys_inp.iter().copied().map(|x| x * 2 + 5).collect();

            type Backend = BurnBack;
            let device = Default::default();
            let keys = Tensor::<Backend, 1, Int>::from_ints(keys_inp, &device).into_primitive();
            let keys = bitcast_tensor(keys);

            let values = Tensor::<Backend, 1, Int>::from_ints(values_inp.as_slice(), &device)
                .into_primitive();
            let values = bitcast_tensor(values);
            let client = keys.client.clone();
            let (ret_keys, ret_values) = radix_argsort(client, keys, values);

            let ret_keys = read_buffer_to_u32(&ret_keys.client, ret_keys.handle.binding());
            let ret_values = read_buffer_to_u32(&ret_values.client, ret_values.handle.binding());

            let inds = argsort(&keys_inp);

            let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i] as u32).collect();
            let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i] as u32).collect();

            for (((key, val), ref_key), ref_val) in ret_keys
                .into_iter()
                .zip(ret_values)
                .zip(ref_keys)
                .zip(ref_values)
            {
                assert_eq!(key, ref_key);
                assert_eq!(val, ref_val);
            }
        }
    }

    #[test]
    fn test_sorting_big() {
        let file = File::open("sort_test.txt").unwrap();
        let mut reader = BufReader::new(file);
        let mut content = String::new();
        reader.read_to_string(&mut content).unwrap();

        // Split on newlines to get each record
        let keys_inp: Vec<i32> = content
            .split(',')
            .map(|field| field.trim().parse().unwrap())
            .collect();

        let values_inp: Vec<_> = keys_inp.iter().copied().map(|x| x * 2 + 5).collect();

        type Backend = BurnBack;
        let device = Default::default();
        let keys =
            Tensor::<Backend, 1, Int>::from_ints(keys_inp.as_slice(), &device).into_primitive();
        let keys = bitcast_tensor(keys);

        let values =
            Tensor::<Backend, 1, Int>::from_ints(values_inp.as_slice(), &device).into_primitive();
        let values = bitcast_tensor(values);

        let client = keys.client.clone();
        let (ret_keys, ret_values) = radix_argsort(client, keys, values);

        let ret_keys = read_buffer_to_u32(&ret_keys.client, ret_keys.handle.binding());
        let ret_values = read_buffer_to_u32(&ret_values.client, ret_values.handle.binding());

        let inds = argsort(&keys_inp);

        let ref_keys: Vec<i32> = inds.iter().map(|&i| keys_inp[i]).collect();
        let ref_values: Vec<i32> = inds.iter().map(|&i| values_inp[i]).collect();

        for (((key, val), ref_key), ref_val) in ret_keys
            .into_iter()
            .zip(ret_values)
            .zip(ref_keys)
            .zip(ref_values)
        {
            assert_eq!(key, ref_key as u32);
            assert_eq!(val, ref_val as u32);
        }
    }
}
