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
use naga_oil::compose::ShaderDefValue;
use tracing::info_span;

use super::generated_bindings::{self, sorting};

pub(crate) trait SplatKernel<S: ComputeServer<Kernel = Kernel>, C: ComputeChannel<S>>
where
    Self: KernelSource + Sized + Copy + Clone + 'static,
{
    const SPAN_NAME: &'static str;
    const BINDING_COUNT: usize;
    const WORKGROUP_SIZE: [u32; 3];
    type Uniforms: NoUninit;

    fn execute(
        self,
        client: &ComputeClient<S, C>,
        uniforms: Self::Uniforms,
        read_handles: &[&Handle<S>],
        write_handles: &[&Handle<S>],
        executions: [u32; 3],
        sync: bool,
    ) {
        let _span = info_span!("Executing", "{}", Self::SPAN_NAME).entered();
        let exec_vec = UVec3::from_array(executions);
        let group_size = UVec3::from_array(Self::WORKGROUP_SIZE);
        let execs = (exec_vec + group_size - 1) / group_size;
        let kernel = Kernel::Custom(Box::new(SourceKernel::new(
            self,
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

        if sync {
            client.sync();
        }
    }
}

#[macro_export]
macro_rules! kernel_source_gen {
    ($struct_name:ident { $($field_name:ident),* }, $module:ident, $uniforms:ty) => {
        #[derive(Debug, Copy, Clone)]
        pub(crate) struct $struct_name {
            $(
                $field_name: bool,
            )*
        }

        impl $struct_name {
            pub fn new($($field_name: bool),*) -> Self {
                Self {
                    $(
                        $field_name,
                    )*
                }
            }

            fn create_shader_hashmap(&self) -> std::collections::HashMap<String, ShaderDefValue> {
                let map = std::collections::HashMap::new();
                $(
                    let mut map = map;

                    if self.$field_name {
                        map.insert(stringify!($field_name).to_owned().to_uppercase(), ShaderDefValue::Bool(true));
                    }
                )*
                map
            }
        }

        impl KernelSource for $struct_name {
            fn source(&self) -> SourceTemplate {
                let mut composer = naga_oil::compose::Composer::default();
                let shader_defs = self.create_shader_hashmap();
                generated_bindings::$module::load_shader_modules_embedded(
                    &mut composer,
                    &shader_defs,
                );
                let module = generated_bindings::$module::load_naga_module_embedded(
                    &mut composer,
                    shader_defs,
                );
                let info = wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::empty(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
                let shader_string = wgpu::naga::back::wgsl::write_string(
                    &module,
                    &info,
                    wgpu::naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
                )
                .expect("failed to convert naga module to source");

                SourceTemplate::new(shader_string)
            }
        }

        impl<S: ComputeServer<Kernel = Kernel>, C: ComputeChannel<S>> SplatKernel<S, C>
            for $struct_name
        {
            const SPAN_NAME: &'static str = stringify!($struct_name);
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

kernel_source_gen!(
    ProjectSplats {},
    project_forward,
    generated_bindings::project_forward::Uniforms
);
kernel_source_gen!(
    MapGaussiansToIntersect {},
    map_gaussian_to_intersects,
    generated_bindings::map_gaussian_to_intersects::Uniforms
);
kernel_source_gen!(GetTileBinEdges {}, get_tile_bin_edges, ());
kernel_source_gen!(
    Rasterize { forward_only },
    rasterize,
    generated_bindings::rasterize::Uniforms
);
kernel_source_gen!(
    RasterizeBackwards {},
    rasterize_backwards,
    generated_bindings::rasterize_backwards::Uniforms
);
kernel_source_gen!(
    ProjectBackwards {},
    project_backwards,
    generated_bindings::project_backwards::Uniforms
);

kernel_source_gen!(PrefixSumScan {}, prefix_sum_scan, ());
kernel_source_gen!(PrefixSumScanSums {}, prefix_sum_scan_sums, ());
kernel_source_gen!(PrefixSumAddScannedSums {}, prefix_sum_add_scanned_sums, ());

kernel_source_gen!(SortCount {}, sort_count, sorting::Config);
kernel_source_gen!(SortReduce {}, sort_reduce, sorting::Config);
kernel_source_gen!(SortScanAdd {}, sort_scan_add, sorting::Config);
kernel_source_gen!(SortScan {}, sort_scan, sorting::Config);
kernel_source_gen!(SortScatter {}, sort_scatter, sorting::Config);
