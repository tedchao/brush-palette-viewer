use brush_kernel::kernel_source_gen;
use brush_kernel::SplatKernel;

use super::shaders::{
    get_tile_bin_edges, map_gaussian_to_intersects, project_backwards, project_forward, rasterize,
    rasterize_backwards,
};

kernel_source_gen!(ProjectSplats {}, project_forward, project_forward::Uniforms);
kernel_source_gen!(
    MapGaussiansToIntersect {},
    map_gaussian_to_intersects,
    map_gaussian_to_intersects::Uniforms
);
kernel_source_gen!(GetTileBinEdges {}, get_tile_bin_edges, ());
kernel_source_gen!(Rasterize { raster_u32 }, rasterize, rasterize::Uniforms);
kernel_source_gen!(
    RasterizeBackwards {},
    rasterize_backwards,
    rasterize_backwards::Uniforms
);
kernel_source_gen!(
    ProjectBackwards {},
    project_backwards,
    project_backwards::Uniforms
);
