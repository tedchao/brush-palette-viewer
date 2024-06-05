use brush_kernel::kernel_source_gen;
use brush_kernel::SplatKernel;

use super::shaders::{
    get_tile_bin_edges, map_gaussian_to_intersects, prefix_sum_add_scanned_sums, prefix_sum_scan,
    prefix_sum_scan_sums, project_backwards, project_forward, rasterize, rasterize_backwards,
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

kernel_source_gen!(PrefixSumScan {}, prefix_sum_scan, ());
kernel_source_gen!(PrefixSumScanSums {}, prefix_sum_scan_sums, ());
kernel_source_gen!(PrefixSumAddScannedSums {}, prefix_sum_add_scanned_sums, ());
