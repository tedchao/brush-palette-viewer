use brush_kernel::bitcast_tensor;
use brush_kernel::create_dispatch_buffer;
use brush_kernel::create_tensor;
use brush_kernel::create_uniform_buffer;
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn_jit::JitBackend;
use burn_wgpu::CubeCount;
use burn_wgpu::JitTensor;
use burn_wgpu::WgpuRuntime;
use shaders::sort_count;
use shaders::sort_reduce;
use shaders::sort_scan;
use shaders::sort_scan_add;
use shaders::sort_scatter;
use tracing::info_span;

use brush_kernel::kernel_source_gen;

mod shaders;

const WG: u32 = shaders::sorting::WG;
const ELEMENTS_PER_THREAD: u32 = shaders::sorting::ELEMENTS_PER_THREAD;
const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;
const BIN_COUNT: u32 = shaders::sorting::BIN_COUNT;

kernel_source_gen!(SortCount {}, sort_count);
kernel_source_gen!(SortReduce {}, sort_reduce);
kernel_source_gen!(SortScanAdd {}, sort_scan_add);
kernel_source_gen!(SortScan {}, sort_scan);
kernel_source_gen!(SortScatter {}, sort_scatter);

pub fn radix_argsort(
    input_keys: JitTensor<WgpuRuntime, u32, 1>,
    input_values: JitTensor<WgpuRuntime, u32, 1>,
    n_sort: JitTensor<WgpuRuntime, u32, 1>,
    sorting_bits: u32,
) -> (
    JitTensor<WgpuRuntime, u32, 1>,
    JitTensor<WgpuRuntime, u32, 1>,
) {
    assert_eq!(input_keys.shape.dims[0], input_values.shape.dims[0]);
    assert!(sorting_bits <= 32);

    let _span = info_span!("Radix sort").entered();

    let client = &input_keys.client.clone();
    let max_n = input_keys.shape.dims[0] as u32;

    // compute buffer and dispatch sizes
    let device = &input_keys.device.clone();

    let max_needed_wgs = max_n.div_ceil(BLOCK_SIZE);
    let count_buf =
        create_tensor::<u32, 1, WgpuRuntime>([(max_needed_wgs as usize) * 16], device, client);
    let reduced_buf = create_tensor::<u32, 1, WgpuRuntime>([BLOCK_SIZE as usize], device, client);

    let output_keys = input_keys;
    let output_values = input_values;
    let output_keys_swap = create_tensor::<u32, 1, _>([max_n as usize], device, client);
    let output_values_swap = create_tensor::<u32, 1, _>([max_n as usize], device, client);

    let (mut last_out, mut last_out_values) = (&output_keys, &output_values);

    let dispatch_span = info_span!("Create dispatch buffers").entered();

    let num_wgs = create_dispatch_buffer(n_sort.clone(), [BLOCK_SIZE, 1, 1]);
    let num_reduce_wgs: Tensor<JitBackend<WgpuRuntime, f32, i32>, 1, Int> =
        Tensor::from_primitive(bitcast_tensor(create_dispatch_buffer(
            num_wgs.clone(),
            [BLOCK_SIZE, 1, 1],
        ))) * Tensor::from_ints([BIN_COUNT, 1, 1], device);
    let num_reduce_wgs: JitTensor<WgpuRuntime, u32, 1> =
        bitcast_tensor(num_reduce_wgs.into_primitive());
    drop(dispatch_span);

    for pass in 0..sorting_bits.div_ceil(4) {
        let (to, to_val) = if pass % 2 == 0 {
            (&output_keys_swap, &output_values_swap)
        } else {
            (&output_keys, &output_values)
        };

        let uniforms_buffer: JitTensor<WgpuRuntime, u32, 1> = create_uniform_buffer(
            shaders::sort_count::Uniforms { shift: pass * 4 },
            device,
            client,
        );

        client.execute(
            SortCount::task(),
            CubeCount::Dynamic(num_wgs.clone().handle.binding()),
            vec![
                uniforms_buffer.clone().handle.binding(),
                n_sort.clone().handle.binding(),
                last_out.handle.clone().binding(),
                count_buf.clone().handle.binding(),
            ],
        );

        client.execute(
            SortReduce::task(),
            CubeCount::Dynamic(num_reduce_wgs.clone().handle.binding()),
            vec![
                n_sort.clone().handle.binding(),
                count_buf.clone().handle.binding(),
                reduced_buf.clone().handle.binding(),
            ],
        );

        client.execute(
            SortScan::task(),
            CubeCount::Static(1, 1, 1),
            vec![
                n_sort.clone().handle.binding(),
                reduced_buf.clone().handle.binding(),
            ],
        );

        client.execute(
            SortScanAdd::task(),
            CubeCount::Dynamic(num_reduce_wgs.handle.clone().binding()),
            vec![
                n_sort.clone().handle.binding(),
                reduced_buf.clone().handle.binding(),
                count_buf.clone().handle.binding(),
            ],
        );

        client.execute(
            SortScatter::task(),
            CubeCount::Dynamic(num_wgs.clone().handle.binding()),
            vec![
                uniforms_buffer.handle.clone().binding(),
                n_sort.clone().handle.binding(),
                last_out.handle.clone().binding(),
                last_out_values.handle.clone().binding(),
                count_buf.handle.clone().binding(),
                to.handle.clone().binding(),
                to_val.handle.clone().binding(),
            ],
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
    use brush_kernel::bitcast_tensor;
    use burn::tensor::{Int, Tensor};
    use burn_wgpu::{JitBackend, WgpuRuntime};

    type Backend = JitBackend<WgpuRuntime, f32, i32>;

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
            let keys = Tensor::<Backend, 1, Int>::from_ints(keys_inp, &device).into_primitive();
            let keys = bitcast_tensor(keys);

            let values = Tensor::<Backend, 1, Int>::from_ints(values_inp.as_slice(), &device)
                .into_primitive();
            let num_points = bitcast_tensor(
                Tensor::<Backend, 1, Int>::from_ints([keys_inp.len() as i32], &device)
                    .into_primitive(),
            );
            let values = bitcast_tensor(values);
            let (ret_keys, ret_values) = radix_argsort(keys, values, num_points, 32);

            let ret_keys =
                Tensor::<Backend, 1, Int>::from_primitive(bitcast_tensor(ret_keys)).into_data();

            let ret_values =
                Tensor::<Backend, 1, Int>::from_primitive(bitcast_tensor(ret_values)).into_data();

            let inds = argsort(&keys_inp);

            let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i] as u32).collect();
            let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i] as u32).collect();

            for (((key, val), ref_key), ref_val) in ret_keys
                .as_slice::<i32>()
                .unwrap()
                .iter()
                .zip(ret_values.as_slice::<i32>().unwrap())
                .zip(ref_keys)
                .zip(ref_values)
            {
                assert_eq!(*key, ref_key as i32);
                assert_eq!(*val, ref_val as i32);
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
        let keys_inp: Vec<_> = content
            .split(',')
            .map(|field| field.trim().parse().unwrap())
            .collect();

        let values_inp: Vec<_> = keys_inp.iter().copied().map(|x| x * 2 + 5).collect();

        let device = Default::default();
        let keys = bitcast_tensor(
            Tensor::<Backend, 1, Int>::from_ints(keys_inp.as_slice(), &device).into_primitive(),
        );
        let values = bitcast_tensor(
            Tensor::<Backend, 1, Int>::from_ints(values_inp.as_slice(), &device).into_primitive(),
        );
        let num_points = bitcast_tensor(
            Tensor::<Backend, 1, Int>::from_ints([keys_inp.len() as i32], &device).into_primitive(),
        );

        let (ret_keys, ret_values) = radix_argsort(keys, values, num_points, 32);

        let ret_keys =
            Tensor::<Backend, 1, Int>::from_primitive(bitcast_tensor(ret_keys)).to_data();
        let ret_values =
            Tensor::<Backend, 1, Int>::from_primitive(bitcast_tensor(ret_values)).to_data();
        let inds = argsort(&keys_inp);

        let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i]).collect();
        let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i]).collect();

        for (((key, val), ref_key), ref_val) in ret_keys
            .as_slice::<i32>()
            .unwrap()
            .iter()
            .zip(ret_values.as_slice::<i32>().unwrap())
            .zip(ref_keys)
            .zip(ref_values)
        {
            assert_eq!(*key, ref_key as i32);
            assert_eq!(*val, ref_val as i32);
        }
    }
}
