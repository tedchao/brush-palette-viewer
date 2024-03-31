use burn::backend::wgpu::{DynamicKernel, DynamicKernelSource, SourceTemplate, WorkGroup};
use burn_compute::{
    channel::ComputeChannel,
    client::ComputeClient,
    server::{ComputeServer, Handle},
};
use burn_jit::compute::Kernel;
use bytemuck::NoUninit;
use glam::UVec3;

use super::gen;

pub(crate) trait SplatKernel<S: ComputeServer<Kernel = Box<dyn Kernel>>, C: ComputeChannel<S>>
where
    Self: Default + DynamicKernelSource + 'static,
{
    const DIM_READ: usize;
    const DIM_WRITE: usize;
    const WORKGROUP_SIZE: [u32; 3];
    type Uniforms: NoUninit;

    fn execute(
        client: &ComputeClient<S, C>,
        uniforms: Self::Uniforms,
        read_handles: [&Handle<S>; Self::DIM_READ],
        write_handles: [&Handle<S>; Self::DIM_WRITE],
        executions: [u32; 3],
    ) {
        let execs = (UVec3::from_array(executions).as_vec3()
            / UVec3::from_array(Self::WORKGROUP_SIZE).as_vec3())
        .ceil()
        .as_uvec3();
        let workgroup = WorkGroup::new(execs.x, execs.y, execs.z);

        let uniform_data = client.create(bytemuck::bytes_of(&uniforms));
        let total_handles: Vec<_> = read_handles
            .into_iter()
            .chain(write_handles)
            .chain([&uniform_data])
            .collect();

        client.execute(
            Box::new(DynamicKernel::new(Self::default(), workgroup)),
            &total_handles,
        );
    }
}

#[derive(Default, Debug)]
pub(crate) struct ProjectSplats {}

impl DynamicKernelSource for ProjectSplats {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::project_forward::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl<S: ComputeServer<Kernel = Box<dyn Kernel>>, C: ComputeChannel<S>> SplatKernel<S, C>
    for ProjectSplats
{
    const DIM_READ: usize = 3;
    const DIM_WRITE: usize = 6;
    type Uniforms = gen::project_forward::Uniforms;
    const WORKGROUP_SIZE: [u32; 3] = gen::project_forward::compute::MAIN_WORKGROUP_SIZE;
}

#[derive(Default, Debug)]
pub(crate) struct MapGaussiansToIntersect {}

impl DynamicKernelSource for MapGaussiansToIntersect {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::map_gaussian_to_intersects::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl<S: ComputeServer<Kernel = Box<dyn Kernel>>, C: ComputeChannel<S>> SplatKernel<S, C>
    for MapGaussiansToIntersect
{
    const DIM_READ: usize = 4;
    const DIM_WRITE: usize = 3;
    type Uniforms = gen::map_gaussian_to_intersects::Uniforms;
    const WORKGROUP_SIZE: [u32; 3] = gen::map_gaussian_to_intersects::compute::MAIN_WORKGROUP_SIZE;
}

#[derive(Default, Debug)]
pub(crate) struct GetTileBinEdges {}

impl DynamicKernelSource for GetTileBinEdges {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::get_tile_bin_edges::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl<S: ComputeServer<Kernel = Box<dyn Kernel>>, C: ComputeChannel<S>> SplatKernel<S, C>
    for GetTileBinEdges
{
    const DIM_READ: usize = 1;
    const DIM_WRITE: usize = 1;
    type Uniforms = gen::get_tile_bin_edges::Uniforms;
    const WORKGROUP_SIZE: [u32; 3] = gen::get_tile_bin_edges::compute::MAIN_WORKGROUP_SIZE;
}

#[derive(Default, Debug)]
pub(crate) struct Rasterize {}

impl DynamicKernelSource for Rasterize {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::rasterize::SHADER_STRING)
    }
    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl<S: ComputeServer<Kernel = Box<dyn Kernel>>, C: ComputeChannel<S>> SplatKernel<S, C>
    for Rasterize
{
    const DIM_READ: usize = 6;
    const DIM_WRITE: usize = 2;
    type Uniforms = gen::rasterize::Uniforms;
    const WORKGROUP_SIZE: [u32; 3] = gen::rasterize::compute::MAIN_WORKGROUP_SIZE;
}

#[derive(Default, Debug)]
pub(crate) struct RasterizeBackwards {}

impl DynamicKernelSource for RasterizeBackwards {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::rasterize_backwards::SHADER_STRING)
    }
    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl<S: ComputeServer<Kernel = Box<dyn Kernel>>, C: ComputeChannel<S>> SplatKernel<S, C>
    for RasterizeBackwards
{
    const DIM_READ: usize = 9;
    const DIM_WRITE: usize = 4;
    type Uniforms = gen::rasterize_backwards::Uniforms;
    const WORKGROUP_SIZE: [u32; 3] = gen::rasterize_backwards::compute::MAIN_WORKGROUP_SIZE;
}

#[derive(Default, Debug)]
pub(crate) struct ProjectBackwards {}

impl DynamicKernelSource for ProjectBackwards {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::project_backwards::SHADER_STRING)
    }
    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl<S: ComputeServer<Kernel = Box<dyn Kernel>>, C: ComputeChannel<S>> SplatKernel<S, C>
    for ProjectBackwards
{
    const DIM_READ: usize = 8;
    const DIM_WRITE: usize = 3;
    type Uniforms = gen::project_backwards::Uniforms;
    const WORKGROUP_SIZE: [u32; 3] = gen::project_backwards::compute::MAIN_WORKGROUP_SIZE;
}
