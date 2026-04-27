//! Render entry point for palette-based 3DGS.
//!
//! C2.5b6: ProjectVisibleWeight runs FIRST, then PopulateProjectedForMap copies
//! geometry into a vanilla ProjectedSplat layout that MapGaussiansToIntersect
//! can read. This lets us reuse Brush's existing tile-binning kernel without
//! patching it.

use brush_kernel::{bytemuck, calc_cube_count_1d, create_meta_binding, create_tensor};
use brush_prefix_sum::prefix_sum;
use brush_render::{
    MainBackend, MainBackendBase,
    camera::Camera,
    shaders::{self, MapGaussiansToIntersect, ProjectSplats},
};
use brush_sort::radix_argsort;
use burn::prelude::*;
use burn::tensor::{
    DType, FloatDType, Int, IntDType, TensorMetadata, TensorPrimitive, Transaction,
    ops::{FloatTensor, FloatTensorOps, IntTensorOps},
};
use burn_cubecl::cubecl::server::KernelArguments;
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_fusion::stream::{Operation, OperationStreams};
use burn_fusion::FusionHandle;
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use burn_wgpu::WgpuRuntime;
use glam::uvec2;

use crate::PaletteSplats;
use crate::shaders::{
    OurTileOffsets, PaletteRemix, PopulateProjectedForMap, ProjectVisibleWeight, RasterizeWeight,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OurTileOffsetsUniforms {
    num_intersections: u32,
    pad_a: u32,
    pad_b: u32,
    pad_c: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PaletteProjectUniforms {
    viewmat:         [[f32; 4]; 4],
    focal:           [f32; 2],
    img_size:        [u32; 2],
    tile_bounds:     [u32; 2],
    pixel_center:    [f32; 2],
    camera_position: [f32; 4],
    sh_degree:       u32,
    total_splats:    u32,
    num_visible:     u32,
    k_full:          u32,
    p_factor:        u32,
    q_factor:        u32,
    pad_a:           u32,
    pad_b:           u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RasterWeightUniforms {
    tile_bounds: [u32; 2],
    img_size:    [u32; 2],
    k_full:      u32,
    pad_a:       u32,
    pad_b:       u32,
    pad_c:       u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RemixUniforms {
    img_w:  u32,
    img_h:  u32,
    k_full: u32,
    pad_a:  u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PopulateUniforms {
    num_visible: u32,
    pad_a: u32,
    pad_b: u32,
    pad_c: u32,
}

fn calc_tile_bounds(img_size: glam::UVec2) -> glam::UVec2 {
    uvec2(
        img_size.x.div_ceil(shaders::helpers::TILE_WIDTH),
        img_size.y.div_ceil(shaders::helpers::TILE_WIDTH),
    )
}

pub async fn render_palette(
    palette_splats: &PaletteSplats<MainBackend>,
    camera: &Camera,
    img_size: glam::UVec2,
    palette_override: Option<&[[f32; 3]]>,
) -> Tensor<MainBackend, 3> {
    log::info!(
        "render_palette: img_size={}x{}, n_splats={}, k_full={}, p={}, q={}",
        img_size.x, img_size.y,
        palette_splats.num_splats(),
        palette_splats.k_full,
        palette_splats.p,
        palette_splats.q,
    );

    // ── Unwrap inputs MainBackend (Fusion) → MainBackendBase ──────────────
    let transforms_fusion    = palette_splats.splats.transforms.val().into_primitive().tensor();
    let raw_opacities_fusion = palette_splats.splats.raw_opacities.val().into_primitive().tensor();
    let low_shs_w_fusion     = palette_splats.low_shs_w.clone().into_primitive().tensor();
    let high_shs_a_fusion    = palette_splats.high_shs_a.clone().into_primitive().tensor();
    let high_shs_b_fusion    = palette_splats.high_shs_b.clone().into_primitive().tensor();

    let client = transforms_fusion.client.clone();
    
    
    use brush_render::MainBackendBase as _MBB;
    let palette = if let Some(override_vals) = palette_override {
        let k_full = palette_splats.k_full as usize;
        assert_eq!(
            override_vals.len(), k_full,
            "palette_override has {} colors; expected {}", override_vals.len(), k_full
        );
        let mut flat = Vec::with_capacity(k_full * 3);
        for c in override_vals {
            flat.push(c[0]);
            flat.push(c[1]);
            flat.push(c[2]);
        }
        let dev_for_palette = palette_splats.palette.device();
        let host_tensor = burn::tensor::Tensor::<MainBackend, 2>::from_data(
            burn::tensor::TensorData::new(flat, [k_full, 3]),
            &dev_for_palette,
        );
        let host_fusion = host_tensor.into_primitive().tensor();
        let host_client = host_fusion.client.clone();
        into_contiguous(host_client.resolve_tensor_float::<_MBB>(host_fusion))
    } else {
        let palette_fusion = palette_splats.palette.clone().into_primitive().tensor();
        into_contiguous(client.clone().resolve_tensor_float::<_MBB>(palette_fusion))
    };


    let transforms    = into_contiguous(client.clone().resolve_tensor_float::<MainBackendBase>(transforms_fusion));
    let raw_opacities = into_contiguous(client.clone().resolve_tensor_float::<MainBackendBase>(raw_opacities_fusion));
    //let palette       = into_contiguous(client.clone().resolve_tensor_float::<MainBackendBase>(palette_fusion));
    let low_shs_w     = into_contiguous(client.clone().resolve_tensor_float::<MainBackendBase>(low_shs_w_fusion));
    let high_shs_a    = into_contiguous(client.clone().resolve_tensor_float::<MainBackendBase>(high_shs_a_fusion));
    let high_shs_b    = into_contiguous(client.clone().resolve_tensor_float::<MainBackendBase>(high_shs_b_fusion));

    let base_device = transforms.device.clone();
    let base_client = transforms.client.clone();

    // ── Geometry uniforms ─────────────────────────────────────────────────
    let tile_bounds = calc_tile_bounds(img_size);
    let total_splats = transforms.shape()[0];
    let project_uniforms = shaders::helpers::ProjectUniforms {
        viewmat: glam::Mat4::from(camera.world_to_local()).to_cols_array_2d(),
        camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
        focal: camera.focal(img_size).into(),
        pixel_center: camera.center(img_size).into(),
        img_size: img_size.into(),
        tile_bounds: tile_bounds.into(),
        sh_degree: 0,
        total_splats: total_splats as u32,
        num_visible: 0,
        pad_a: 0,
    };

    // ── Step 1: ProjectSplats ─────────────────────────────────────────────
    let num_visible_buffer       = <MainBackendBase as IntTensorOps<MainBackendBase>>::int_zeros([1].into(), &base_device, IntDType::U32);
    let num_intersections_buffer = <MainBackendBase as IntTensorOps<MainBackendBase>>::int_zeros([1].into(), &base_device, IntDType::U32);
    let intersect_counts         = <MainBackendBase as IntTensorOps<MainBackendBase>>::int_zeros([total_splats].into(), &base_device, IntDType::U32);
    let max_radius               = <MainBackendBase as FloatTensorOps<MainBackendBase>>::float_zeros([total_splats].into(), &base_device, FloatDType::F32);
    let global_from_presort_gid  = create_tensor::<1>([total_splats], &base_device, DType::U32);
    let depths                   = create_tensor::<1>([total_splats], &base_device, DType::F32);

    unsafe {
        base_client.launch_unchecked(
            ProjectSplats::task(false),
            calc_cube_count_1d(total_splats as u32, ProjectSplats::WORKGROUP_SIZE[0]),
            KernelArguments::new()
                .with_buffers(vec![
                    transforms.handle.clone().binding(),
                    raw_opacities.handle.clone().binding(),
                    global_from_presort_gid.handle.clone().binding(),
                    depths.handle.clone().binding(),
                    num_visible_buffer.handle.clone().binding(),
                    intersect_counts.handle.clone().binding(),
                    num_intersections_buffer.handle.clone().binding(),
                    max_radius.handle.clone().binding(),
                ])
                .with_info(create_meta_binding(project_uniforms)),
        );
    }
    let _ = max_radius;

    // ── Step 2: readback ──────────────────────────────────────────────────
    let (num_visible, num_intersections) = if total_splats == 0 {
        (0u32, 0u32)
    } else {
        let data = Transaction::default()
            .register(Tensor::<MainBackendBase, 1, Int>::from_primitive(num_visible_buffer))
            .register(Tensor::<MainBackendBase, 1, Int>::from_primitive(num_intersections_buffer))
            .execute_async()
            .await
            .expect("readback failed");
        (
            data[0].clone().into_vec::<u32>().expect("nv")[0],
            data[1].clone().into_vec::<u32>().expect("ni")[0],
        )
    };
    log::info!("ProjectSplats done: num_visible={}, num_intersections={}", num_visible, num_intersections);

    let num_visible_sz = (num_visible as usize).max(1);

    // ── Step 3: Depth sort ────────────────────────────────────────────────
    let global_from_compact_gid = {
        let depths_sl = <MainBackendBase as FloatTensorOps<MainBackendBase>>::float_slice(depths, &[(0..num_visible_sz).into()]);
        let gfp_sl    = <MainBackendBase as IntTensorOps<MainBackendBase>>::int_slice(global_from_presort_gid, &[(0..num_visible_sz).into()]);
        let (_, gfc) = radix_argsort(depths_sl, gfp_sl, 32);
        gfc
    };

    // ── Step 4–5: gather + prefix sum ────────────────────────────────────
    let compact_counts = <MainBackendBase as IntTensorOps<MainBackendBase>>::int_gather(
        0, intersect_counts, global_from_compact_gid.clone(),
    );
    let cum_tiles_hit = prefix_sum(compact_counts);

    // ── Step 6: ProjectVisibleWeight (MOVED UP from later) ────────────────
    let projected_weight = create_tensor::<2>([num_visible_sz, 16], &base_device, DType::F32);

    let palette_uniforms = PaletteProjectUniforms {
        viewmat: glam::Mat4::from(camera.world_to_local()).to_cols_array_2d(),
        focal: camera.focal(img_size).into(),
        img_size: img_size.into(),
        tile_bounds: tile_bounds.into(),
        pixel_center: camera.center(img_size).into(),
        camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
        sh_degree: 3,
        total_splats: total_splats as u32,
        num_visible,
        k_full: palette_splats.k_full,
        p_factor: palette_splats.p,
        q_factor: palette_splats.q,
        pad_a: 0, pad_b: 0,
    };

    unsafe {
        base_client.launch_unchecked(
            ProjectVisibleWeight::task(),
            calc_cube_count_1d(num_visible.max(1), ProjectVisibleWeight::WORKGROUP_SIZE[0]),
            KernelArguments::new()
                .with_buffers(vec![
                    transforms.handle.clone().binding(),
                    raw_opacities.handle.clone().binding(),
                    global_from_compact_gid.handle.clone().binding(),
                    low_shs_w.handle.clone().binding(),
                    high_shs_a.handle.clone().binding(),
                    high_shs_b.handle.clone().binding(),
                    projected_weight.handle.clone().binding(),
                ])
                .with_info(create_meta_binding(palette_uniforms)),
        );
    }
    log::info!("ProjectVisibleWeight dispatched");

    // ── Step 7: PopulateProjectedForMap (NEW) ─────────────────────────────
    // Copy xy/conic/opac from projected_weight into a vanilla ProjectedSplat
    // layout buffer that MapGaussiansToIntersect can read.
    let proj_size = std::mem::size_of::<shaders::helpers::ProjectedSplat>() / std::mem::size_of::<f32>();
    let projected_splats_dummy = create_tensor::<2>([num_visible_sz, proj_size], &base_device, DType::F32);

    let pop_u = PopulateUniforms {
        num_visible,
        pad_a: 0, pad_b: 0, pad_c: 0,
    };
    unsafe {
        base_client.launch_unchecked(
            PopulateProjectedForMap::task(),
            calc_cube_count_1d(num_visible.max(1), PopulateProjectedForMap::WORKGROUP_SIZE[0]),
            KernelArguments::new()
                .with_buffers(vec![
                    projected_weight.handle.clone().binding(),
                    projected_splats_dummy.handle.clone().binding(),
                ])
                .with_info(create_meta_binding(pop_u)),
        );
    }
    log::info!("PopulateProjectedForMap dispatched");

    // ── Step 8: MapGaussiansToIntersect ───────────────────────────────────
    let num_tiles = tile_bounds.x * tile_bounds.y;
    let buffer_size = (num_intersections as usize).max(1);
    let tile_id_from_isect      = create_tensor::<1>([buffer_size], &base_device, DType::U32);
    let compact_gid_from_isect  = create_tensor::<1>([buffer_size], &base_device, DType::U32);

    let map_uniforms = shaders::map_gaussians_to_intersect::Uniforms {
        tile_bounds: tile_bounds.into(),
        num_visible,
        pad_a: 0,
    };

    base_client.launch(
        MapGaussiansToIntersect::task(),
        calc_cube_count_1d(num_visible, MapGaussiansToIntersect::WORKGROUP_SIZE[0]),
        KernelArguments::new()
            .with_buffers(vec![
                projected_splats_dummy.handle.clone().binding(),
                cum_tiles_hit.handle.clone().binding(),
                tile_id_from_isect.handle.clone().binding(),
                compact_gid_from_isect.handle.clone().binding(),
            ])
            .with_info(create_meta_binding(map_uniforms)),
    );
    let _ = projected_splats_dummy;

    // ── Step 9: tile sort ─────────────────────────────────────────────────
    let bits = u32::BITS - num_tiles.leading_zeros();
    let (tile_id_from_isect, compact_gid_from_isect) =
        radix_argsort(tile_id_from_isect, compact_gid_from_isect, bits);

    // ── Step 10: tile offsets (our shader) ────────────────────────────────
    let tile_offsets = <MainBackendBase as IntTensorOps<MainBackendBase>>::int_zeros(
        [tile_bounds.y as usize, tile_bounds.x as usize, 2].into(),
        &base_device, IntDType::U32,
    );
    let our_uniforms = OurTileOffsetsUniforms {
        num_intersections, pad_a: 0, pad_b: 0, pad_c: 0,
    };
    unsafe {
        base_client.launch_unchecked(
            OurTileOffsets::task(),
            calc_cube_count_1d(num_intersections.max(1), OurTileOffsets::WORKGROUP_SIZE[0]),
            KernelArguments::new()
                .with_buffers(vec![
                    tile_id_from_isect.handle.clone().binding(),
                    tile_offsets.handle.clone().binding(),
                ])
                .with_info(create_meta_binding(our_uniforms)),
        );
    }

    // ── Step 11: RasterizeWeight ──────────────────────────────────────────
    let h = img_size.y as usize;
    let w = img_size.x as usize;
    const MAX_K_FULL: usize = 8;
    let weight_image = <MainBackendBase as FloatTensorOps<MainBackendBase>>::float_zeros(
        [h, w, MAX_K_FULL].into(), &base_device, FloatDType::F32,
    );

    let raster_uniforms = RasterWeightUniforms {
        tile_bounds: tile_bounds.into(),
        img_size:    img_size.into(),
        k_full:      palette_splats.k_full,
        pad_a: 0, pad_b: 0, pad_c: 0,
    };
    let n_pixels = (w * h) as u32;
    let total_threads = num_tiles * (shaders::helpers::TILE_WIDTH * shaders::helpers::TILE_WIDTH);
    unsafe {
        base_client.launch_unchecked(
            RasterizeWeight::task(),
            calc_cube_count_1d(total_threads, RasterizeWeight::WORKGROUP_SIZE[0]),
            KernelArguments::new()
                .with_buffers(vec![
                    compact_gid_from_isect.handle.clone().binding(),
                    tile_offsets.handle.clone().binding(),
                    projected_weight.handle.clone().binding(),
                    weight_image.handle.clone().binding(),
                ])
                .with_info(create_meta_binding(raster_uniforms)),
        );
    }
    log::info!("RasterizeWeight dispatched");

    // ── Step 12: PaletteRemix → output RGB image ──────────────────────────
    let out_img: FloatTensor<MainBackendBase> =
        <MainBackendBase as FloatTensorOps<MainBackendBase>>::float_zeros(
            [h, w, 1].into(), &base_device, FloatDType::F32,
    );

    let remix_uniforms = RemixUniforms {
        img_w: img_size.x,
        img_h: img_size.y,
        k_full: palette_splats.k_full,
        pad_a: 0,
    };
    unsafe {
        base_client.launch_unchecked(
            PaletteRemix::task(),
            calc_cube_count_1d(n_pixels, PaletteRemix::WORKGROUP_SIZE[0] * PaletteRemix::WORKGROUP_SIZE[1]),
            KernelArguments::new()
                .with_buffers(vec![
                    weight_image.handle.clone().binding(),
                    palette.handle.clone().binding(),
                    out_img.handle.clone().binding(),
                ])
                .with_info(create_meta_binding(remix_uniforms)),
        );
    }
    log::info!("PaletteRemix dispatched");

    // ── Step 13: Rewrap MainBackendBase output back into MainBackend (Fusion) ─
    #[derive(Debug)]
    struct BindOp {
        desc: CustomOpIr,
        out_img: FloatTensor<MainBackendBase>,
    }
    impl Operation<FusionCubeRuntime<WgpuRuntime>> for BindOp {
        fn execute(
            &self,
            h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
        ) {
            let (_, outputs) = self.desc.as_fixed::<0, 1>();
            let [out_img] = outputs;
            h.register_float_tensor::<MainBackendBase>(&out_img.id, self.out_img.clone());
        }
    }

    let out_img_ir = TensorIr::uninit(client.create_empty_handle(), out_img.shape(), DType::F32);
    let stream = OperationStreams::default();
    let desc = CustomOpIr::new("palette_render_bind", &[], &[out_img_ir]);
    let op = BindOp { desc: desc.clone(), out_img };

    let outputs = client.register(stream, OperationIr::Custom(desc), op).outputs();
    let [out_fusion] = outputs;

    Tensor::from_primitive(TensorPrimitive::Float(out_fusion))
}