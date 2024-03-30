use burn::backend::wgpu::{DynamicKernelSource, SourceTemplate};
use derive_new::new;
use glam::{uvec3, UVec3};

use super::gen;


#[derive(new, Debug)]
pub(crate) struct ProjectSplats {}

impl DynamicKernelSource for ProjectSplats {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::project_forward::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl ProjectSplats {
    pub(crate) fn workgroup_size() -> UVec3 {
        uvec3(16, 1, 1)
    }
}

#[derive(new, Debug)]
pub(crate) struct MapGaussiansToIntersect {}

impl DynamicKernelSource for MapGaussiansToIntersect {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::map_gaussian_to_intersects::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl MapGaussiansToIntersect {
    pub(crate) fn workgroup_size() -> UVec3 {
        uvec3(16, 1, 1)
    }
}

#[derive(new, Debug)]
pub(crate) struct GetTileBinEdges {}

impl GetTileBinEdges {
    pub(crate) fn workgroup_size() -> UVec3 {
        uvec3(16, 1, 1)
    }
}

impl DynamicKernelSource for GetTileBinEdges {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::get_tile_bin_edges::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

#[derive(new, Debug)]
pub(crate) struct RasterizeForward {}

impl DynamicKernelSource for RasterizeForward {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::rasterize::SHADER_STRING)
    }
    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl RasterizeForward {
    pub(crate) fn workgroup_size() -> UVec3 {
        uvec3(16, 16, 1)
    }
}

#[derive(new, Debug)]
pub(crate) struct RasterizeBackwards {}

impl DynamicKernelSource for RasterizeBackwards {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::rasterize_backwards::SHADER_STRING)
    }
    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl RasterizeBackwards {
    pub(crate) fn workgroup_size() -> UVec3 {
        uvec3(16, 16, 1)
    }
}

#[derive(new, Debug)]
pub(crate) struct ProjectBackwards {}

impl DynamicKernelSource for ProjectBackwards {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::project_backwards::SHADER_STRING)
    }
    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl ProjectBackwards {
    pub(crate) fn workgroup_size() -> UVec3 {
        uvec3(16, 1, 1)
    }
}
