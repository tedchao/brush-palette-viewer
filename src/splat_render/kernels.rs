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

use super::generated_bindings;

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
            let total_handles = [read_handles, write_handles, &[&uniform_data]].concat();
            assert_eq!(total_handles.len(), Self::BINDING_COUNT);
            client.execute(kernel, &total_handles);
        } else {
            let total_handles = [read_handles, write_handles].concat();
            assert_eq!(total_handles.len(), Self::BINDING_COUNT);
            client.execute(kernel, &total_handles);
        }
    }
}

#[derive(Default, Debug)]
pub(crate) struct ProjectSplats {}

impl KernelSource for ProjectSplats {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(generated_bindings::project_forward::SHADER_STRING)
    }
}

impl<S: ComputeServer<Kernel = Kernel>, C: ComputeChannel<S>> SplatKernel<S, C> for ProjectSplats {
    const BINDING_COUNT: usize =
        generated_bindings::project_forward::bind_groups::WgpuBindGroup0::LAYOUT_DESCRIPTOR
            .entries
            .len();
    type Uniforms = generated_bindings::project_forward::Uniforms;
    const WORKGROUP_SIZE: [u32; 3] =
        generated_bindings::project_forward::compute::MAIN_WORKGROUP_SIZE;
}

#[derive(Default, Debug)]
pub(crate) struct MapGaussiansToIntersect {}

impl KernelSource for MapGaussiansToIntersect {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(generated_bindings::map_gaussian_to_intersects::SHADER_STRING)
    }
}

impl<S: ComputeServer<Kernel = Kernel>, C: ComputeChannel<S>> SplatKernel<S, C>
    for MapGaussiansToIntersect
{
    const BINDING_COUNT: usize =
    generated_bindings::map_gaussian_to_intersects::bind_groups::WgpuBindGroup0::LAYOUT_DESCRIPTOR
            .entries
            .len();
    type Uniforms = generated_bindings::map_gaussian_to_intersects::Uniforms;
    const WORKGROUP_SIZE: [u32; 3] =
        generated_bindings::map_gaussian_to_intersects::compute::MAIN_WORKGROUP_SIZE;
}

#[derive(Default, Debug)]
pub(crate) struct GetTileBinEdges {}

impl KernelSource for GetTileBinEdges {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(generated_bindings::get_tile_bin_edges::SHADER_STRING)
    }
}

impl<S: ComputeServer<Kernel = Kernel>, C: ComputeChannel<S>> SplatKernel<S, C>
    for GetTileBinEdges
{
    const BINDING_COUNT: usize =
        generated_bindings::get_tile_bin_edges::bind_groups::WgpuBindGroup0::LAYOUT_DESCRIPTOR
            .entries
            .len();
    type Uniforms = ();
    const WORKGROUP_SIZE: [u32; 3] =
        generated_bindings::get_tile_bin_edges::compute::MAIN_WORKGROUP_SIZE;
}

#[derive(Default, Debug)]
pub(crate) struct Rasterize {}

impl Rasterize {
    pub const BLOCK_WIDTH: u32 = generated_bindings::rasterize::BLOCK_WIDTH;
}

impl KernelSource for Rasterize {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(generated_bindings::rasterize::SHADER_STRING)
    }
}

impl<S: ComputeServer<Kernel = Kernel>, C: ComputeChannel<S>> SplatKernel<S, C> for Rasterize {
    const BINDING_COUNT: usize =
        generated_bindings::rasterize::bind_groups::WgpuBindGroup0::LAYOUT_DESCRIPTOR
            .entries
            .len();
    type Uniforms = generated_bindings::rasterize::Uniforms;
    const WORKGROUP_SIZE: [u32; 3] = generated_bindings::rasterize::compute::MAIN_WORKGROUP_SIZE;
}

#[derive(Default, Debug)]
pub(crate) struct RasterizeBackwards {}

impl KernelSource for RasterizeBackwards {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(generated_bindings::rasterize_backwards::SHADER_STRING)
    }
}

impl<S: ComputeServer<Kernel = Kernel>, C: ComputeChannel<S>> SplatKernel<S, C>
    for RasterizeBackwards
{
    const BINDING_COUNT: usize =
        generated_bindings::rasterize_backwards::bind_groups::WgpuBindGroup0::LAYOUT_DESCRIPTOR
            .entries
            .len();
    type Uniforms = generated_bindings::rasterize_backwards::Uniforms;
    const WORKGROUP_SIZE: [u32; 3] =
        generated_bindings::rasterize_backwards::compute::MAIN_WORKGROUP_SIZE;
}

#[derive(Default, Debug)]
pub(crate) struct ProjectBackwards {}

impl KernelSource for ProjectBackwards {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(generated_bindings::project_backwards::SHADER_STRING)
    }
}

impl<S: ComputeServer<Kernel = Kernel>, C: ComputeChannel<S>> SplatKernel<S, C>
    for ProjectBackwards
{
    const BINDING_COUNT: usize =
        generated_bindings::project_backwards::bind_groups::WgpuBindGroup0::LAYOUT_DESCRIPTOR
            .entries
            .len();
    type Uniforms = generated_bindings::project_backwards::Uniforms;
    const WORKGROUP_SIZE: [u32; 3] =
        generated_bindings::project_backwards::compute::MAIN_WORKGROUP_SIZE;
}
