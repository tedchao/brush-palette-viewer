use super::kernels::SplatKernel;
use super::prefix_sum::prefix_sum;
use super::radix_sort::radix_argsort;
use crate::camera::Camera;
use crate::gaussian_splats::{num_sh_coeffs, sh_basis_from_coeffs};
use crate::splat_render::create_tensor;
use crate::splat_render::dim_check::DimCheck;
use crate::splat_render::kernels::{
    GetTileBinEdges, MapGaussiansToIntersect, ProjectBackwards, ProjectSplats, Rasterize,
    RasterizeBackwards,
};
use burn::backend::autodiff::NodeID;
use burn::tensor::ops::FloatTensorOps;
use burn::tensor::ops::IntTensorOps;
use burn::tensor::{Shape, Tensor};

use burn_wgpu::JitTensor;
use tracing::info_span;

use super::{
    bitcast_tensor, read_buffer_as_u32, shaders, Aux, Backend, BurnBack, BurnRuntime, FloatTensor,
};
use burn::backend::{
    autodiff::{
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    wgpu::into_contiguous,
    Autodiff,
};
use glam::{uvec2, Vec3};

fn render_forward(
    camera: &Camera,
    img_size: glam::UVec2,
    means: JitTensor<BurnRuntime, f32, 2>,
    _xy_dummy: JitTensor<BurnRuntime, f32, 2>,
    log_scales: JitTensor<BurnRuntime, f32, 2>,
    quats: JitTensor<BurnRuntime, f32, 2>,
    sh_coeffs: JitTensor<BurnRuntime, f32, 2>,
    raw_opacity: JitTensor<BurnRuntime, f32, 1>,
    background: glam::Vec3,
    raster_u32: bool,
) -> (JitTensor<BurnRuntime, f32, 3>, Aux<BurnBack>) {
    let _span = info_span!("render_gaussians").entered();
    let setup = info_span!("setup").entered();

    let (means, log_scales, quats, sh_coeffs, raw_opacities) = (
        into_contiguous(means.clone()),
        into_contiguous(log_scales.clone()),
        into_contiguous(quats.clone()),
        into_contiguous(sh_coeffs.clone()),
        into_contiguous(raw_opacity.clone()),
    );

    DimCheck::new()
        .check_dims(&means, ["D".into(), 3.into()])
        .check_dims(&log_scales, ["D".into(), 3.into()])
        .check_dims(&quats, ["D".into(), 4.into()])
        .check_dims(&sh_coeffs, ["D".into(), "C".into()])
        .check_dims(&raw_opacities, ["D".into()]);

    let sh_degree = sh_basis_from_coeffs(sh_coeffs.shape.dims[1] / 3);

    // Divide screen into blocks.
    let tile_width = shaders::helpers::TILE_WIDTH;
    let tile_bounds = uvec2(
        img_size.x.div_ceil(tile_width),
        img_size.y.div_ceil(tile_width),
    );

    let client = means.client.clone();
    let device = means.device.clone();

    let sync = || {
        if false {
            client.sync()
        }
    };

    let num_points = means.shape.dims[0];

    // Projected gaussian values.
    let xys = create_tensor::<f32, 2>(&client, &device, [num_points, 2]);
    let depths = create_tensor::<f32, 1>(&client, &device, [num_points]);
    let colors = create_tensor::<f32, 2>(&client, &device, [num_points, 4]);
    let radii = create_tensor::<u32, 1>(&client, &device, [num_points]);
    let conic_comps = create_tensor::<f32, 2>(&client, &device, [num_points, 4]);

    // A note on some confusing naming that'll be used throughout this function:
    // Gaussians are stored in various states of buffers, eg. at the start they're all in one big bufffer,
    // then we sparsely store some results, then sort gaussian based on depths, etc.
    // Overall this means there's lots of indices flying all over the place, and it's hard to keep track
    // what is indexing what. So, for some sanity, try to match a few "gaussian ids" (gid) variable names.
    // - Global Gaussin ID - global_gid
    // - Compacted Gaussian ID - compact_gid
    // - Compacted Gaussian ID sorted by depth - depthsort_gid
    // - Per tile intersection depth sorted ID - tiled_gid
    // - Sorted by tile per tile intersection depth sorted ID - sorted_tiled_gid
    // Then, various buffers map between these, which are named x_from_y_gid, eg.
    //  global_from_compact_gid or compact_from_depthsort_gid.

    // Tile rendering setup.
    let num_tiles_hit = BurnBack::int_zeros(Shape::new([num_points]), &device);
    let num_visible = bitcast_tensor(BurnBack::int_zeros(Shape::new([1]), &device));
    let global_from_compact_gid = create_tensor::<u32, 1>(&client, &device, [num_points]);

    // TODO: This should just be an int_arrange, but that atm in burn falls back to a
    // CPU operation and is way too slow. I could also make this a special case in the radix sort.
    let arranged_ids = create_tensor::<u32, 1>(&client, &device, [num_points]);

    drop(setup);

    {
        let _span = info_span!("ProjectSplats").entered();

        ProjectSplats::new().execute(
            &client,
            shaders::project_forward::Uniforms::new(
                camera.world_to_local(),
                camera.focal(img_size).into(),
                camera.center(img_size).into(),
                img_size.into(),
                tile_bounds.into(),
                tile_width,
                0.01,
                sh_degree as u32,
            ),
            &[
                means.handle.binding(),
                log_scales.handle.binding(),
                quats.handle.binding(),
                sh_coeffs.handle.binding(),
                raw_opacities.handle.binding(),
            ],
            &[
                arranged_ids.handle.clone().binding(),
                global_from_compact_gid.handle.clone().binding(),
                xys.handle.clone().binding(),
                depths.handle.clone().binding(),
                colors.handle.clone().binding(),
                radii.handle.clone().binding(),
                conic_comps.handle.clone().binding(),
                num_tiles_hit.handle.clone().binding(),
                num_visible.handle.clone().binding(),
            ],
            [num_points as u32, 1, 1],
        );

        sync();
    }

    let depth_sort_span = info_span!("DepthSort").entered();
    // Interpret depth as u32. This is fine with radix sort, as long as depth > 0.0,
    // which we know to be the case given how we cull.
    let (_, compact_from_depthsort_gid) = radix_argsort(
        client.clone(),
        bitcast_tensor(depths.clone()),
        arranged_ids,
        num_visible.clone(),
        32,
    );
    sync();

    drop(depth_sort_span);

    let cum_hit_span = info_span!("TilesPermute").entered();

    // Permute the number of tiles hit for the sorted gaussians.
    // This means num_tiles_hit is not stored per compact_gid, but per depthsort_gid.
    let num_tiles_hit = bitcast_tensor(BurnBack::int_gather(
        0,
        bitcast_tensor(num_tiles_hit),
        bitcast_tensor(compact_from_depthsort_gid.clone()),
    ));
    // Calculate cumulative sums as offsets for the intersections buffer.
    let cum_tiles_hit = prefix_sum(&client, num_tiles_hit);

    let num_intersects = bitcast_tensor(BurnBack::int_slice(
        bitcast_tensor(cum_tiles_hit.clone()),
        #[allow(clippy::single_range_in_vec_init)]
        [num_points - 1..num_points],
    ));

    let num_tiles = tile_bounds[0] * tile_bounds[1];

    // TODO: How do we actually properly deal with this :/
    let max_intersects =
        read_buffer_as_u32(&client, num_intersects.clone().handle.binding())[0] as usize;
    // let max_intersects = (num_points * (num_tiles as usize)).min(256 * 4 * 65535);

    // Each intersection maps to a gaussian.
    let tile_id_from_isect = create_tensor::<u32, 1>(&client, &device, [max_intersects]);
    let depthsort_gid_from_isect = create_tensor::<u32, 1>(&client, &device, [max_intersects]);

    sync();
    drop(cum_hit_span);

    {
        let _span = info_span!("MapGaussiansToIntersect").entered();

        // Dispatch one thread per point.
        // TODO: Really want to do an indirect dispatch here for num_visible.
        MapGaussiansToIntersect::new().execute(
            &client,
            shaders::map_gaussian_to_intersects::Uniforms::new(tile_bounds.into()),
            &[
                compact_from_depthsort_gid.handle.clone().binding(),
                xys.handle.clone().binding(),
                conic_comps.handle.clone().binding(),
                radii.handle.clone().binding(),
                cum_tiles_hit.handle.clone().binding(),
                num_visible.handle.clone().binding(),
            ],
            &[
                tile_id_from_isect.handle.clone().binding(),
                depthsort_gid_from_isect.handle.clone().binding(),
            ],
            [num_points as u32, 1, 1],
        );
        sync();
    }

    // We're sorting by tile ID, but we know beforehand what the maximum value
    // can be. We don't need to sort all the leading 0 bits!
    let bits = u32::BITS - num_tiles.leading_zeros();

    let tile_sort_span = info_span!("Tile sort").entered();
    let (tile_id_from_isect, depthsort_gid_from_isect) = radix_argsort(
        client.clone(),
        bitcast_tensor(tile_id_from_isect),
        bitcast_tensor(depthsort_gid_from_isect),
        num_intersects.clone(),
        bits,
    );
    sync();
    drop(tile_sort_span);

    let tile_edge_span = info_span!("GetTileBinEdges").entered();

    let tile_bins = BurnBack::int_zeros(
        Shape::new([tile_bounds[0] as usize, tile_bounds[1] as usize, 2]),
        &device,
    );
    GetTileBinEdges::new().execute(
        &client,
        (),
        &[
            tile_id_from_isect.handle.binding(),
            num_intersects.handle.clone().binding(),
        ],
        &[tile_bins.handle.clone().binding()],
        [
            (max_intersects as u32).div_ceil(shaders::get_tile_bin_edges::VERTICAL_GROUPS),
            shaders::get_tile_bin_edges::VERTICAL_GROUPS,
            1,
        ],
    );
    sync();
    drop(tile_edge_span);

    let tile_edge_span = info_span!("Rasterize").entered();

    let out_img = if raster_u32 {
        // Channels are packed into 1 byte - aka one u32 element.
        create_tensor(
            &client,
            &device,
            [img_size.y as usize, img_size.x as usize, 1],
        )
    } else {
        create_tensor(
            &client,
            &device,
            [img_size.y as usize, img_size.x as usize, 4],
        )
    };

    let final_index =
        create_tensor::<u32, 2>(&client, &device, [img_size.y as usize, img_size.x as usize]);

    let mut out_binds = vec![out_img.handle.clone().binding()];

    if !raster_u32 {
        out_binds.push(final_index.handle.clone().binding());
    }

    Rasterize::new(raster_u32).execute(
        &client,
        shaders::rasterize::Uniforms::new(img_size.into(), background.into()),
        &[
            depthsort_gid_from_isect.handle.clone().binding(),
            compact_from_depthsort_gid.handle.clone().binding(),
            tile_bins.handle.clone().binding(),
            xys.handle.clone().binding(),
            conic_comps.handle.clone().binding(),
            colors.handle.clone().binding(),
        ],
        out_binds.as_slice(),
        [img_size.x, img_size.y, 1],
    );
    sync();
    drop(tile_edge_span);

    (
        out_img,
        Aux {
            num_visible: Tensor::from_primitive(bitcast_tensor(num_visible)),
            num_intersects: Tensor::from_primitive(bitcast_tensor(num_intersects)),
            tile_bins: Tensor::from_primitive(bitcast_tensor(tile_bins)),
            cum_tiles_hit: Tensor::from_primitive(bitcast_tensor(cum_tiles_hit)),
            radii_compact: Tensor::from_primitive(bitcast_tensor(radii)),
            conic_comps: Tensor::from_primitive(bitcast_tensor(conic_comps)),
            colors: Tensor::from_primitive(colors),
            depths: Tensor::from_primitive(depths),
            xys: Tensor::from_primitive(bitcast_tensor(xys)),
            final_index: Tensor::from_primitive(bitcast_tensor(final_index)),
            depthsort_gid_from_isect: Tensor::from_primitive(bitcast_tensor(
                depthsort_gid_from_isect,
            )),
            compact_from_depthsort_gid: Tensor::from_primitive(bitcast_tensor(
                compact_from_depthsort_gid,
            )),
            global_from_compact_gid: Tensor::from_primitive(bitcast_tensor(
                global_from_compact_gid,
            )),
        },
    )
}

impl Backend for BurnBack {
    fn render_gaussians(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self, 2>,
        xy_dummy: FloatTensor<Self, 2>,
        log_scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
        sh_coeffs: FloatTensor<Self, 2>,
        raw_opacity: FloatTensor<Self, 1>,
        background: glam::Vec3,
        render_u32_buffer: bool,
    ) -> (FloatTensor<Self, 3>, Aux<BurnBack>) {
        render_forward(
            camera,
            img_size,
            means,
            xy_dummy,
            log_scales,
            quats,
            sh_coeffs,
            raw_opacity,
            background,
            render_u32_buffer,
        )
    }
}

#[derive(Debug, Clone)]
struct GaussianBackwardState {
    cam: Camera,
    background: Vec3,

    // Splat inputs.
    means: NodeID,
    log_scales: NodeID,
    quats: NodeID,
    raw_opac: NodeID,
    sh_degree: usize,
    out_img: Tensor<BurnBack, 3>,
    aux: Aux<BurnBack>,
}

#[derive(Debug)]
struct RenderBackwards;

impl<C: CheckpointStrategy> Backend for Autodiff<BurnBack, C> {
    fn render_gaussians(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self, 2>,
        xy_dummy: FloatTensor<Self, 2>,
        log_scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
        sh_coeffs: FloatTensor<Self, 2>,
        raw_opacity: FloatTensor<Self, 1>,
        background: glam::Vec3,
        render_u32_buffer: bool,
    ) -> (FloatTensor<Self, 3>, Aux<Self>) {
        let (out_img, aux) = render_forward(
            camera,
            img_size,
            means.clone().primitive,
            xy_dummy.clone().primitive,
            log_scales.clone().primitive,
            quats.clone().primitive,
            sh_coeffs.clone().primitive,
            raw_opacity.clone().primitive,
            background,
            render_u32_buffer,
        );

        let prep_nodes = RenderBackwards
            .prepare::<C>([
                means.node.clone(),
                xy_dummy.node.clone(),
                log_scales.node.clone(),
                quats.node.clone(),
                sh_coeffs.node.clone(),
                raw_opacity.node.clone(),
            ])
            .compute_bound()
            .stateful();

        let orig_aux = aux.clone();

        let wrap_aux = Aux::<Self> {
            num_visible: Tensor::from_inner(aux.num_visible),
            num_intersects: Tensor::from_inner(aux.num_intersects),
            tile_bins: Tensor::from_inner(aux.tile_bins),
            radii_compact: Tensor::from_inner(aux.radii_compact),
            depthsort_gid_from_isect: Tensor::from_inner(aux.depthsort_gid_from_isect),
            compact_from_depthsort_gid: Tensor::from_inner(aux.compact_from_depthsort_gid),
            depths: Tensor::from_inner(aux.depths),
            xys: Tensor::from_inner(aux.xys),
            cum_tiles_hit: Tensor::from_inner(aux.cum_tiles_hit),
            conic_comps: Tensor::from_inner(aux.conic_comps),
            colors: Tensor::from_inner(aux.colors),
            final_index: Tensor::from_inner(aux.final_index),
            global_from_compact_gid: Tensor::from_inner(aux.global_from_compact_gid),
        };

        let sh_degree = sh_basis_from_coeffs(sh_coeffs.primitive.shape.dims[1] / 3);

        // Prepare a stateful operation with each variable node and corresponding graph.
        //
        // Each node can be fetched with `ops.parents` in the same order as defined here.
        match prep_nodes {
            OpsKind::Tracked(mut prep) => {
                // Grads need floating point buffer.
                let state = GaussianBackwardState {
                    means: prep.checkpoint(&means),
                    log_scales: prep.checkpoint(&log_scales),
                    quats: prep.checkpoint(&quats),
                    raw_opac: prep.checkpoint(&raw_opacity),
                    aux: orig_aux,
                    cam: camera.clone(),
                    background,
                    out_img: Tensor::from_primitive(out_img.clone()),
                    sh_degree,
                };

                (prep.finish(state, out_img), wrap_aux)
            }
            OpsKind::UnTracked(prep) => {
                // When no node is tracked, we can just compute the original operation without
                // keeping any state.
                (prep.finish(out_img), wrap_aux)
            }
        }
    }
}

impl Backward<BurnBack, 3, 6> for RenderBackwards {
    type State = GaussianBackwardState;

    fn backward(
        self,
        ops: Ops<Self::State, 6>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let _span = info_span!("render_gaussians backwards").entered();

        let state = ops.state;
        let aux = state.aux;

        let img_dimgs = state.out_img.dims();
        let img_size = glam::uvec2(img_dimgs[1] as u32, img_dimgs[0] as u32);

        let v_output = grads.consume::<BurnBack, 3>(&ops.node);
        DimCheck::new().check_dims(&v_output, [img_size.y.into(), img_size.x.into(), 4.into()]);

        let means = checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 2>>(state.means);
        let quats = checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 2>>(state.quats);
        let log_scales =
            checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 2>>(state.log_scales);
        let raw_opac =
            checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 1>>(state.raw_opac);

        let client = &v_output.client;
        let device = &v_output.device;

        let num_points = means.shape.dims[0];

        let max_intersects = aux.depthsort_gid_from_isect.shape().dims[0];
        let v_xy_scatter = BurnBack::float_zeros(Shape::new([max_intersects, 2]), device);
        let v_conic_scatter = BurnBack::float_zeros(Shape::new([max_intersects, 4]), device);
        let v_colors_scatter = BurnBack::float_zeros(Shape::new([max_intersects, 4]), device);

        let hit_ids = BurnBack::int_zeros(Shape::new([num_points]), device);

        RasterizeBackwards::new().execute(
            client,
            shaders::rasterize_backwards::Uniforms::new(img_size.into(), state.background.into()),
            &[
                aux.depthsort_gid_from_isect
                    .into_primitive()
                    .handle
                    .binding(),
                aux.compact_from_depthsort_gid
                    .clone()
                    .into_primitive()
                    .handle
                    .binding(),
                aux.tile_bins.into_primitive().handle.binding(),
                aux.xys.into_primitive().handle.binding(),
                aux.cum_tiles_hit.clone().into_primitive().handle.binding(),
                aux.conic_comps.clone().into_primitive().handle.binding(),
                aux.colors.clone().into_primitive().handle.binding(),
                aux.final_index.into_primitive().handle.binding(),
                state.out_img.into_primitive().handle.binding(),
                v_output.handle.binding(),
            ],
            &[
                v_xy_scatter.handle.clone().binding(),
                v_conic_scatter.handle.clone().binding(),
                v_colors_scatter.handle.clone().binding(),
                hit_ids.handle.clone().binding(),
            ],
            [img_size.x, img_size.y, 1],
        );

        // TODO: Can't this be done for just visible points
        let v_means = BurnBack::float_zeros(Shape::new([num_points, 3]), device);
        let v_scales = BurnBack::float_zeros(Shape::new([num_points, 3]), device);

        let v_quats = BurnBack::float_zeros(Shape::new([num_points, 4]), device);
        let v_coeffs = BurnBack::float_zeros(
            Shape::new([num_points, num_sh_coeffs(state.sh_degree) * 3]),
            device,
        );
        let v_opac = BurnBack::float_zeros(Shape::new([num_points]), device);
        let v_xys = BurnBack::float_zeros(Shape::new([num_points, 2]), device);

        ProjectBackwards::new().execute(
            client,
            shaders::project_backwards::Uniforms::new(
                state.cam.world_to_local(),
                state.cam.focal(img_size).into(),
                img_size.into(),
                state.sh_degree as u32,
            ),
            &[
                means.handle.binding(),
                log_scales.handle.binding(),
                quats.handle.binding(),
                raw_opac.handle.binding(),
                aux.conic_comps.into_primitive().handle.binding(),
                aux.cum_tiles_hit.into_primitive().handle.clone().binding(),
                v_xy_scatter.handle.binding(),
                v_conic_scatter.handle.binding(),
                v_colors_scatter.handle.binding(),
                aux.num_visible.into_primitive().handle.clone().binding(),
                aux.global_from_compact_gid
                    .into_primitive()
                    .handle
                    .clone()
                    .binding(),
                aux.compact_from_depthsort_gid
                    .clone()
                    .into_primitive()
                    .handle
                    .clone()
                    .binding(),
            ],
            &[
                v_means.handle.clone().binding(),
                v_xys.handle.clone().binding(),
                v_scales.handle.clone().binding(),
                v_quats.handle.clone().binding(),
                v_coeffs.handle.clone().binding(),
                v_opac.handle.clone().binding(),
            ],
            [num_points as u32, 1, 1],
        );

        // Register gradients for parent nodes.
        let [mean_parent, xys_parent, log_scales_parent, quats_parent, coeffs_parent, raw_opacity_parent] =
            ops.parents;

        if let Some(node) = mean_parent {
            grads.register::<BurnBack, 2>(node.id, v_means);
        }

        if let Some(node) = log_scales_parent {
            grads.register::<BurnBack, 2>(node.id, v_scales);
        }

        if let Some(node) = quats_parent {
            grads.register::<BurnBack, 2>(node.id, v_quats);
        }

        if let Some(node) = coeffs_parent {
            grads.register::<BurnBack, 2>(node.id, v_coeffs);
        }

        if let Some(node) = raw_opacity_parent {
            grads.register::<BurnBack, 1>(node.id, v_opac);
        }

        if let Some(node) = xys_parent {
            grads.register::<BurnBack, 2>(node.id, v_xys);
        }
    }
}

pub fn render<B: Backend>(
    camera: &Camera,
    img_size: glam::UVec2,
    means: Tensor<B, 2>,
    xy_dummy: Tensor<B, 2>,
    log_scales: Tensor<B, 2>,
    quats: Tensor<B, 2>,
    sh_coeffs: Tensor<B, 2>,
    raw_opacity: Tensor<B, 1>,
    background: glam::Vec3,
    render_u32_buffer: bool,
) -> (Tensor<B, 3>, Aux<B>) {
    let (img, aux) = B::render_gaussians(
        camera,
        img_size,
        means.clone().into_primitive(),
        xy_dummy.clone().into_primitive(),
        log_scales.clone().into_primitive(),
        quats.clone().into_primitive(),
        sh_coeffs.clone().into_primitive(),
        raw_opacity.clone().into_primitive(),
        background,
        render_u32_buffer,
    );
    (Tensor::from_primitive(img), aux)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Read;

    use crate::camera::{focal_to_fov, fov_to_focal};

    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use burn::tensor::{Data, Float};
    use burn_wgpu::WgpuDevice;

    type DiffBack = Autodiff<BurnBack>;

    use safetensors::tensor::TensorView;
    use safetensors::SafeTensors;

    use anyhow::{Context, Result};

    // TODO: Add some reference renders.
    #[test]
    fn renders_at_all() {
        // Check if rendering doesn't hard crash or anything.
        // These are some zero-sized gaussians, so we know
        // what the result should look like.
        let cam = Camera::new(glam::vec3(0.0, 0.0, 0.0), glam::Quat::IDENTITY, 0.5, 0.5);
        let img_size = glam::uvec2(32, 32);
        let device = WgpuDevice::BestAvailable;

        let num_points = 8;
        let means = Tensor::<DiffBack, 2, _>::zeros([num_points, 3], &device);
        let xy_dummy = Tensor::<DiffBack, 2, _>::zeros([num_points, 2], &device);

        let log_scales = Tensor::ones([num_points, 3], &device) * 2.0;
        let quats = Tensor::from_data(glam::Quat::IDENTITY.to_array(), &device)
            .unsqueeze_dim(0)
            .repeat(0, num_points);
        let sh_coeffs = Tensor::ones([num_points, 4], &device);
        let raw_opacity = Tensor::zeros([num_points], &device);
        let (output, _) = render(
            &cam,
            img_size,
            means,
            xy_dummy,
            log_scales,
            quats,
            sh_coeffs,
            raw_opacity,
            glam::vec3(0.123, 0.123, 0.123),
            false,
        );

        let rgb = output.clone().slice([0..32, 0..32, 0..3]);
        let alpha = output.clone().slice([0..32, 0..32, 3..4]);
        // TODO: Maybe use all_close from burn - but that seems to be
        // broken atm.
        assert_approx_eq!(rgb.clone().mean().to_data().value[0], 0.123, 1e-5);
        assert_approx_eq!(alpha.clone().mean().to_data().value[0], 0.0);
    }

    fn float_from_u8(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect()
    }

    // Nb: this only handles float tensors, good enough :)
    fn safe_tensor_to_burn1<B: Backend>(t: TensorView, device: &B::Device) -> Tensor<B, 1, Float> {
        Tensor::from_data(
            Data::new(float_from_u8(t.data()), Shape::new([t.shape()[0]])).convert(),
            device,
        )
    }

    fn safe_tensor_to_burn2<B: Backend>(t: TensorView, device: &B::Device) -> Tensor<B, 2, Float> {
        Tensor::from_data(
            Data::new(
                float_from_u8(t.data()),
                Shape::new([t.shape()[0], t.shape()[1]]),
            )
            .convert(),
            device,
        )
    }

    fn safe_tensor_to_burn3<B: Backend>(t: TensorView, device: &B::Device) -> Tensor<B, 3, Float> {
        Tensor::from_data(
            Data::new(
                float_from_u8(t.data()),
                Shape::new([t.shape()[0], t.shape()[1], t.shape()[2]]),
            )
            .convert(),
            device,
        )
    }

    #[test]
    fn test_reference() -> Result<()> {
        let device = WgpuDevice::BestAvailable;
        #[cfg(feature = "rerun")]
        let rec = rerun::RecordingStreamBuilder::new("visualize training").spawn()?;

        let crab_img = image::open("./test_cases/crab.png")?;
        // Convert the image to RGB format
        // Get the raw buffer
        let raw_buffer = crab_img.to_rgb8().into_raw();
        let crab_tens: Tensor<DiffBack, 3> = Tensor::from_floats(
            raw_buffer
                .iter()
                .map(|&b| b as f32 / 255.0)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([crab_img.height() as usize, crab_img.width() as usize, 3]);

        for path in ["basic_case", "mix_case"] {
            let mut buffer = Vec::new();
            let _ =
                File::open(format!("./test_cases/{path}.safetensors"))?.read_to_end(&mut buffer)?;
            let tensors = SafeTensors::deserialize(&buffer)?;

            let means =
                safe_tensor_to_burn2::<DiffBack>(tensors.tensor("means")?, &device).require_grad();
            let num_points = means.dims()[0];

            let xy_dummy = Tensor::zeros([num_points, 2], &device).require_grad();

            let log_scales =
                safe_tensor_to_burn2::<DiffBack>(tensors.tensor("scales")?, &device).require_grad();

            let coeffs = safe_tensor_to_burn3::<DiffBack>(tensors.tensor("coeffs")?, &device)
                .reshape([num_points, 3])
                .require_grad();

            let quats =
                safe_tensor_to_burn2::<DiffBack>(tensors.tensor("quats")?, &device).require_grad();
            let opacities = safe_tensor_to_burn1::<DiffBack>(tensors.tensor("opacities")?, &device)
                .require_grad();

            let xys_ref = safe_tensor_to_burn2::<DiffBack>(tensors.tensor("xys")?, &device);
            let conics_ref = safe_tensor_to_burn2::<DiffBack>(tensors.tensor("conics")?, &device);
            let img_ref = safe_tensor_to_burn3::<DiffBack>(tensors.tensor("out_img")?, &device);
            let img_dims = img_ref.dims();

            let fov = std::f32::consts::PI * 0.5;

            let focal = fov_to_focal(fov, img_dims[1] as u32);
            let fov_x = focal_to_fov(focal, img_dims[1] as u32);
            let fov_y = focal_to_fov(focal, img_dims[0] as u32);

            let cam = Camera::new(
                glam::vec3(0.0, 0.0, -8.0),
                glam::Quat::IDENTITY,
                fov_x,
                fov_y,
            );

            let (out, aux_out) = render(
                &cam,
                glam::uvec2(img_dims[1] as u32, img_dims[0] as u32),
                means.clone(),
                xy_dummy.clone(),
                log_scales.clone(),
                quats.clone(),
                coeffs.clone(),
                opacities.clone(),
                glam::vec3(0.0, 0.0, 0.0),
                false,
            );

            let out_rgb = out.clone().slice([0..img_dims[0], 0..img_dims[1], 0..3]);

            #[cfg(feature = "rerun")]
            {
                use ndarray::Array;
                rec.log(
                    "img/image",
                    &rerun::Image::try_from(
                        Array::from_shape_vec(out_rgb.dims(), out_rgb.to_data().value)?
                            .map(|x| (*x * 255.0).clamp(0.0, 255.0) as u8),
                    )?,
                )?;

                rec.log(
                    "img/ref",
                    &rerun::Image::try_from(
                        Array::from_shape_vec(img_ref.dims(), img_ref.to_data().value)?
                            .map(|x| (*x * 255.0).clamp(0.0, 255.0) as u8),
                    )?,
                )?;

                rec.log(
                    "img/dif",
                    &rerun::Tensor::try_from(Array::from_shape_vec(
                        img_ref.dims(),
                        (img_ref.clone() - out_rgb.clone()).to_data().value,
                    )?)?,
                )?;
            }

            let num_visible = aux_out.num_visible.to_data().value[0] as usize;
            let perm = aux_out.global_from_compact_gid.clone();

            let xys = aux_out.xys.slice([0..num_visible]);
            let xys_ref = xys_ref.select(0, perm.clone()).slice([0..num_visible]);

            let conics = aux_out.conic_comps.slice([0..num_visible, 0..3]);
            let conics_ref = conics_ref.select(0, perm.clone()).slice([0..num_visible]);

            let grads = (out_rgb.clone() - crab_tens.clone())
                .powf_scalar(2.0)
                .mean()
                .backward();

            let v_opacities_ref =
                safe_tensor_to_burn1::<DiffBack>(tensors.tensor("v_opacities")?, &device).inner();
            let v_opacities = opacities.grad(&grads).context("opacities grad")?;

            let v_coeffs_ref =
                safe_tensor_to_burn3::<DiffBack>(tensors.tensor("v_coeffs")?, &device)
                    .reshape([num_points, 3])
                    .inner();
            let v_coeffs = coeffs.grad(&grads).context("coeffs grad")?;

            let v_quats = quats.grad(&grads).context("quats grad")?;
            let v_quats_ref =
                safe_tensor_to_burn2::<DiffBack>(tensors.tensor("v_quats")?, &device).inner();

            let v_scales = log_scales.grad(&grads).context("scales grad")?;
            let v_scales_ref =
                safe_tensor_to_burn2::<DiffBack>(tensors.tensor("v_scales")?, &device).inner();

            let v_means_ref =
                safe_tensor_to_burn2::<DiffBack>(tensors.tensor("v_means")?, &device).inner();
            let v_means = means.grad(&grads).context("means grad")?;

            let v_xys_ref =
                safe_tensor_to_burn2::<DiffBack>(tensors.tensor("v_xy")?, &device).inner();

            let v_xys = xy_dummy.grad(&grads).context("no xys grad")?;

            assert!(xys.all_close(xys_ref, Some(1e-5), Some(1e-5)));
            assert!(conics.all_close(conics_ref, Some(1e-5), Some(1e-6)));
            assert!(out_rgb.all_close(img_ref, Some(1e-5), Some(1e-6)));

            assert!(v_xys.all_close(v_xys_ref, Some(1e-5), Some(1e-6)));
            assert!(v_opacities.all_close(v_opacities_ref, Some(1e-5), Some(1e-6)));
            assert!(v_coeffs.all_close(v_coeffs_ref, Some(1e-5), Some(1e-6)));
            assert!(v_quats.all_close(v_quats_ref, Some(1e-5), Some(1e-6)));
            assert!(v_scales.all_close(v_scales_ref, Some(1e-5), Some(1e-6)));
            assert!(v_means.all_close(v_means_ref, Some(1e-5), Some(1e-6)));
        }
        Ok(())
    }

    // // TODO: This doesn't work yet for some reason. Are the gradients wrong? Or?
    // #[test]
    // fn test_mean_grads() {
    //     let cam = Camera::new(glam::vec3(0.0, 0.0, -5.0), glam::Quat::IDENTITY, 0.5, 0.5);
    //     let num_points = 1;

    //     let img_size = glam::uvec2(16, 16);
    //     let device = WgpuDevice::BestAvailable;

    //     let means = Tensor::<DiffBack, 2, _>::zeros([num_points, 3], &device).require_grad();
    //     let log_scales = Tensor::ones([num_points, 3], &device).require_grad();
    //     let quats = Tensor::from_data(glam::Quat::IDENTITY.to_array(), &device)
    //         .unsqueeze_dim(0)
    //         .repeat(0, num_points)
    //         .require_grad();
    //     let sh_coeffs = Tensor::zeros([num_points, 4], &device).require_grad();
    //     let raw_opacity = Tensor::zeros([num_points], &device).require_grad();

    //     let mut dloss_dmeans_stat = Tensor::zeros([num_points], &device);

    //     // Calculate a stochasic gradient estimation by perturbing random dimensions.
    //     let num_iters = 50;

    //     for _ in 0..num_iters {
    //         let eps = 1e-4;

    //         let flip_vec = Tensor::<DiffBack, 1>::random(
    //             [num_points],
    //             burn::tensor::Distribution::Uniform(-1.0, 1.0),
    //             &device,
    //         );
    //         let seps = flip_vec * eps;

    //         let l1 = render(
    //             &cam,
    //             img_size,
    //             means.clone(),
    //             log_scales.clone(),
    //             quats.clone(),
    //             sh_coeffs.clone(),
    //             raw_opacity.clone() - seps.clone(),
    //             glam::Vec3::ZERO,
    //         )
    //         .0
    //         .mean();

    //         let l2 = render(
    //             &cam,
    //             img_size,
    //             means.clone(),
    //             log_scales.clone(),
    //             quats.clone(),
    //             sh_coeffs.clone(),
    //             raw_opacity.clone() + seps.clone(),
    //             glam::Vec3::ZERO,
    //         )
    //         .0
    //         .mean();

    //         let df = l2 - l1;
    //         dloss_dmeans_stat = dloss_dmeans_stat + df * (seps * 2.0).recip();
    //     }

    //     dloss_dmeans_stat = dloss_dmeans_stat / (num_iters as f32);
    //     let dloss_dmeans_stat = dloss_dmeans_stat.into_data().value;

    //     let loss = render(
    //         &cam,
    //         img_size,
    //         means.clone(),
    //         log_scales.clone(),
    //         quats.clone(),
    //         sh_coeffs.clone(),
    //         raw_opacity.clone(),
    //         glam::Vec3::ZERO,
    //     )
    //     .0
    //     .mean();
    //     // calculate numerical gradients.
    //     // Compare to reference value.

    //     // Check if rendering doesn't hard crash or anything.
    //     // These are some zero-sized gaussians, so we know
    //     // what the result should look like.
    //     let grads = loss.backward();

    //     // Get the gradient of the rendered image.
    //     let dloss_dmeans = (Tensor::<BurnBack, 1>::from_primitive(
    //         grads.get(&raw_opacity.clone().into_primitive()).unwrap(),
    //     ))
    //     .into_data()
    //     .value;

    //     println!("Stat grads {dloss_dmeans_stat:.5?}");
    //     println!("Calc grads {dloss_dmeans:.5?}");

    //     // TODO: These results don't make sense at all currently, which is either
    //     // mildly bad news or very bad news :)
    // }
}
