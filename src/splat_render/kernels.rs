use std::mem::size_of;

use burn_compute::{
    channel::ComputeChannel,
    client::ComputeClient,
    server::{ComputeServer, Handle},
};

use burn::backend::wgpu::{
    Kernel, KernelSource, SourceKernel, SourceTemplate, WorkGroup, WorkgroupSize,
};

use bytemuck::NoUninit;
use glam::UVec3;

use super::generated_bindings::{self, sorting};

pub(crate) trait SplatKernel<S: ComputeServer<Kernel = Kernel>, C: ComputeChannel<S>>
where
    Self: Default + KernelSource + 'static,
{
    const BINDING_COUNT: usize;
    const WORKGROUP_SIZE: [u32; 3];
    type Uniforms: NoUninit;

    fn execute(
        client: &ComputeClient<S, C>,
        uniforms: Self::Uniforms,
        read_handles: &[&Handle<S>],
        write_handles: &[&Handle<S>],
        executions: [u32; 3],
    ) {
        let exec_vec = UVec3::from_array(executions);
        let group_size = UVec3::from_array(Self::WORKGROUP_SIZE);
        let execs = (exec_vec + group_size - 1) / group_size;
        let kernel = Kernel::Custom(Box::new(SourceKernel::new(
            Self::default(),
            WorkGroup::new(execs.x, execs.y, execs.z),
            WorkgroupSize::new(group_size.x, group_size.y, group_size.z),
        )));

        if size_of::<Self::Uniforms>() != 0 {
            let uniform_data = client.create(bytemuck::bytes_of(&uniforms));
            let total_handles = [&[&uniform_data], read_handles, write_handles].concat();
            assert_eq!(total_handles.len(), Self::BINDING_COUNT);
            client.execute(kernel, &total_handles);
        } else {
            let total_handles = [read_handles, write_handles].concat();
            assert_eq!(total_handles.len(), Self::BINDING_COUNT);
            client.execute(kernel, &total_handles);
        }
    }
}

#[macro_export]
macro_rules! kernel_source_gen {
    ($struct_name:ident, $module:ident) => {
        kernel_source_gen!($struct_name, $module, generated_bindings::$module::Uniforms);
    };

    ($struct_name:ident, $module:ident, $uniforms:ty) => {
        #[derive(Default, Debug)]
        pub(crate) struct $struct_name {}

        impl KernelSource for $struct_name {
            fn source(&self) -> SourceTemplate {
                SourceTemplate::new(generated_bindings::$module::SHADER_STRING)
            }
        }

        impl<S: ComputeServer<Kernel = Kernel>, C: ComputeChannel<S>> SplatKernel<S, C>
            for $struct_name
        {
            const BINDING_COUNT: usize =
                generated_bindings::$module::bind_groups::WgpuBindGroup0::LAYOUT_DESCRIPTOR
                    .entries
                    .len();
            type Uniforms = $uniforms;
            const WORKGROUP_SIZE: [u32; 3] =
                generated_bindings::$module::compute::MAIN_WORKGROUP_SIZE;
        }
    };
}

kernel_source_gen!(ProjectSplats, project_forward);
kernel_source_gen!(MapGaussiansToIntersect, map_gaussian_to_intersects);
kernel_source_gen!(GetTileBinEdges, get_tile_bin_edges, ());
kernel_source_gen!(Rasterize, rasterize);
kernel_source_gen!(RasterizeBackwards, rasterize_backwards);
kernel_source_gen!(ProjectBackwards, project_backwards);

kernel_source_gen!(PrefixSumScan, prefix_sum_scan, ());
kernel_source_gen!(PrefixSumScanSums, prefix_sum_scan_sums, ());
kernel_source_gen!(PrefixSumAddScannedSums, prefix_sum_add_scanned_sums, ());

kernel_source_gen!(SortCount, sort_count, sorting::Config);
kernel_source_gen!(SortReduce, sort_reduce, sorting::Config);
kernel_source_gen!(SortScanAdd, sort_scan_add, sorting::Config);
kernel_source_gen!(SortScan, sort_scan, sorting::Config);
kernel_source_gen!(SortScatter, sort_scatter, sorting::Config);

kernel_source_gen!(Zero, zero, ());
