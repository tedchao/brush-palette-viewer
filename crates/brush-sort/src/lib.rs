use brush_kernel::create_tensor;
use brush_kernel::BurnRuntime;
use burn_wgpu::JitTensor;
use shaders::sort_count;
use shaders::sort_reduce;
use shaders::sort_scan;
use shaders::sort_scan_add;
use shaders::sort_scatter;
use shaders::sorting;
use tracing::info_span;

use brush_kernel::kernel_source_gen;
use brush_kernel::SplatKernel;

mod shaders;

const WG: u32 = shaders::sorting::WG;
const ELEMENTS_PER_THREAD: u32 = shaders::sorting::ELEMENTS_PER_THREAD;
const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;
const BIN_COUNT: u32 = shaders::sorting::BIN_COUNT;

kernel_source_gen!(SortCount {}, sort_count, sorting::Config);
kernel_source_gen!(SortReduce {}, sort_reduce, ());
kernel_source_gen!(SortScanAdd {}, sort_scan_add, ());
kernel_source_gen!(SortScan {}, sort_scan, ());
kernel_source_gen!(SortScatter {}, sort_scatter, sorting::Config);

pub fn radix_argsort(
    input_keys: JitTensor<BurnRuntime, u32, 1>,
    input_values: JitTensor<BurnRuntime, u32, 1>,
    num_points: JitTensor<BurnRuntime, u32, 1>,
    sorting_bits: u32,
) -> (
    JitTensor<BurnRuntime, u32, 1>,
    JitTensor<BurnRuntime, u32, 1>,
) {
    assert_eq!(input_keys.shape.dims[0], input_values.shape.dims[0]);
    assert!(sorting_bits <= 32);

    let _span = info_span!("Radix sort").entered();

    let client = &input_keys.client.clone();
    let max_n = input_keys.shape.dims[0] as u32;

    // compute buffer and dispatch sizes
    let max_needed_wgs = max_n.div_ceil(BLOCK_SIZE);
    let max_num_reduce_wgs = BIN_COUNT * max_needed_wgs.div_ceil(BLOCK_SIZE);

    let device = &input_keys.device.clone();

    let count_buf = create_tensor::<u32, 1>([(max_needed_wgs as usize) * 16], device, client);
    let reduced_buf = create_tensor::<u32, 1>([BLOCK_SIZE as usize], device, client);

    let output_keys = input_keys;
    let output_values = input_values;
    let output_keys_swap = create_tensor::<u32, 1>([max_n as usize], device, client);
    let output_values_swap = create_tensor::<u32, 1>([max_n as usize], device, client);

    // NB: We fill in num_keys from the GPU!
    // This at least prevents sorting values we don't need, but really
    // we should use indirect dispatch for this.
    let mut config = shaders::sorting::Config { shift: 0 };

    let (mut last_out, mut last_out_values) = (&output_keys, &output_values);

    for pass in 0..sorting_bits.div_ceil(4) {
        let (to, to_val) = if pass % 2 == 0 {
            (&output_keys_swap, &output_values_swap)
        } else {
            (&output_keys, &output_values)
        };

        config.shift = pass * 4;

        let wg = shaders::sorting::WG;

        let effective_wg_vert = max_needed_wgs.div_ceil(shaders::sorting::VERTICAL_GROUPS);
        SortCount::new().execute(
            client,
            config,
            &[
                num_points.handle.clone().binding(),
                last_out.handle.clone().binding(),
            ],
            &[count_buf.clone().handle.binding()],
            [effective_wg_vert * wg, shaders::sorting::VERTICAL_GROUPS],
        );

        SortReduce::new().execute(
            client,
            (),
            &[
                num_points.handle.clone().binding(),
                count_buf.clone().handle.binding(),
            ],
            &[reduced_buf.clone().handle.binding()],
            [max_num_reduce_wgs * wg],
        );

        SortScan::new().execute(
            client,
            (),
            &[num_points.handle.clone().binding()],
            &[reduced_buf.clone().handle.binding()],
            [1, 1, 1],
        );

        SortScanAdd::new().execute(
            client,
            (),
            &[
                num_points.handle.clone().binding(),
                reduced_buf.clone().handle.binding(),
            ],
            &[count_buf.clone().handle.binding()],
            [max_num_reduce_wgs * wg],
        );

        SortScatter::new().execute(
            client,
            config,
            &[
                num_points.handle.clone().binding(),
                last_out.handle.clone().binding(),
                last_out_values.handle.clone().binding(),
                count_buf.handle.clone().binding(),
            ],
            &[to.handle.clone().binding(), to_val.handle.clone().binding()],
            [effective_wg_vert * wg, shaders::sorting::VERTICAL_GROUPS],
        );

        (last_out, last_out_values) = (&to, &to_val);
    }
    (last_out.clone(), last_out_values.clone())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::{
        fs::File,
        io::{BufReader, Read},
    };

    use crate::radix_argsort;
    use brush_kernel::{bitcast_tensor, BurnBack};
    use burn::tensor::{Int, Tensor};

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

            let device = Default::default();
            let keys = Tensor::<BurnBack, 1, Int>::from_ints(keys_inp, &device).into_primitive();
            let keys = bitcast_tensor(keys);

            let values = Tensor::<BurnBack, 1, Int>::from_ints(values_inp.as_slice(), &device)
                .into_primitive();
            let num_points = bitcast_tensor(
                Tensor::<BurnBack, 1, Int>::from_ints([keys_inp.len() as i32], &device)
                    .into_primitive(),
            );
            let values = bitcast_tensor(values);
            let (ret_keys, ret_values) = radix_argsort(keys, values, num_points, 32);

            let ret_keys: Vec<i32> =
                Tensor::<BurnBack, 1, Int>::from_primitive(bitcast_tensor(ret_keys))
                    .to_data()
                    .value;

            let ret_values: Vec<i32> =
                Tensor::<BurnBack, 1, Int>::from_primitive(bitcast_tensor(ret_values))
                    .to_data()
                    .value;

            let inds = argsort(&keys_inp);

            let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i] as u32).collect();
            let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i] as u32).collect();

            for (((key, val), ref_key), ref_val) in ret_keys
                .into_iter()
                .zip(ret_values)
                .zip(ref_keys)
                .zip(ref_values)
            {
                assert_eq!(key, ref_key as i32);
                assert_eq!(val, ref_val as i32);
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

        let device = Default::default();
        let keys = bitcast_tensor(
            Tensor::<BurnBack, 1, Int>::from_ints(keys_inp.as_slice(), &device).into_primitive(),
        );
        let values = bitcast_tensor(
            Tensor::<BurnBack, 1, Int>::from_ints(values_inp.as_slice(), &device).into_primitive(),
        );
        let num_points = bitcast_tensor(
            Tensor::<BurnBack, 1, Int>::from_ints([keys_inp.len() as i32], &device)
                .into_primitive(),
        );

        let (ret_keys, ret_values) = radix_argsort(keys, values, num_points, 32);

        let ret_keys: Vec<i32> =
            Tensor::<BurnBack, 1, Int>::from_primitive(bitcast_tensor(ret_keys))
                .to_data()
                .value;

        let ret_values: Vec<i32> =
            Tensor::<BurnBack, 1, Int>::from_primitive(bitcast_tensor(ret_values))
                .to_data()
                .value;
        let inds = argsort(&keys_inp);

        let ref_keys: Vec<i32> = inds.iter().map(|&i| keys_inp[i]).collect();
        let ref_values: Vec<i32> = inds.iter().map(|&i| values_inp[i]).collect();

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
