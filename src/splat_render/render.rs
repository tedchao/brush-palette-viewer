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

use super::{bitcast_tensor, shaders, Aux, Backend, BurnBack, BurnRuntime, FloatTensor};
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
    log_scales: JitTensor<BurnRuntime, f32, 2>,
    quats: JitTensor<BurnRuntime, f32, 2>,
    sh_coeffs: JitTensor<BurnRuntime, f32, 2>,
    raw_opacity: JitTensor<BurnRuntime, f32, 1>,
    background: glam::Vec3,
    forward_only: bool,
) -> (JitTensor<BurnRuntime, f32, 3>, Aux<BurnBack>) {
    let _span = info_span!("render_gaussians").entered();

    let (means, log_scales, quats, sh_coeffs, raw_opacities) = (
        into_contiguous(means.clone()),
        into_contiguous(log_scales.clone()),
        into_contiguous(quats.clone()),
        into_contiguous(sh_coeffs.clone()),
        into_contiguous(raw_opacity.clone()),
    );

    DimCheck::new()
        .check_dims(&means, ["D".into(), 4.into()])
        .check_dims(&log_scales, ["D".into(), 4.into()])
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

    let num_points = means.shape.dims[0];

    // Projected gaussian values.
    let xys = create_tensor::<f32, 2>(&client, &device, [num_points, 2]);
    let depths = create_tensor::<f32, 1>(&client, &device, [num_points]);
    let colors = create_tensor::<f32, 2>(&client, &device, [num_points, 4]);
    let radii = create_tensor::<u32, 1>(&client, &device, [num_points]);
    let cov2ds = create_tensor::<f32, 2>(&client, &device, [num_points, 4]);

    // Tile rendering setup.
    let num_tiles_hit = BurnBack::int_zeros(Shape::new([num_points]), &device);
    let num_visible = bitcast_tensor(BurnBack::int_zeros(Shape::new([1]), &device));
    let compact_ids = create_tensor::<u32, 1>(&client, &device, [num_points]);
    let remap_ids = create_tensor::<u32, 1>(&client, &device, [num_points]);

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
            camera.position.into(),
        ),
        &[
            means.handle.binding(),
            log_scales.handle.binding(),
            quats.handle.binding(),
            sh_coeffs.handle.binding(),
            raw_opacities.handle.binding(),
        ],
        &[
            compact_ids.handle.clone().binding(),
            remap_ids.handle.clone().binding(),
            xys.handle.clone().binding(),
            depths.handle.clone().binding(),
            colors.handle.clone().binding(),
            radii.handle.clone().binding(),
            cov2ds.handle.clone().binding(),
            num_tiles_hit.handle.clone().binding(),
            num_visible.handle.clone().binding(),
        ],
        [num_points as u32, 1, 1],
    );

    // Interpret depth as u32. This is fine with radix sort, as long as depth > 0.0,
    // which we know to be the case given how we cull.
    let (_, sorted_compact_ids) = radix_argsort(
        client.clone(),
        bitcast_tensor(depths),
        compact_ids,
        num_visible.clone(),
        32,
    );

    // Gather the number of tiles hit for the sorted gaussians.
    let num_tiles_hit = bitcast_tensor(BurnBack::int_gather(
        0,
        bitcast_tensor(num_tiles_hit),
        bitcast_tensor(sorted_compact_ids.clone()),
    ));

    // Calculate cumulative sums as offsets for the intersections buffer.
    let cum_tiles_hit = prefix_sum(&client, num_tiles_hit);

    // TODO: How do we actually properly deal with this :/
    // TODO: Look into some more ways of reducing intersections.
    // Ideally render gaussians in chunks, but that might be hell with the backward pass.
    let max_intersects = num_points * 3;

    // Each intersection maps to a gaussian.
    let tile_ids_unsorted = create_tensor::<u32, 1>(&client, &device, [max_intersects]);
    let gaussian_per_intersect_unsorted =
        create_tensor::<u32, 1>(&client, &device, [max_intersects]);

    // Dispatch one thread per point.
    // TODO: Really want to do an indirect dispatch here.
    MapGaussiansToIntersect::new().execute(
        &client,
        shaders::map_gaussian_to_intersects::Uniforms::new(tile_bounds.into()),
        &[
            sorted_compact_ids.handle.clone().binding(),
            xys.handle.clone().binding(),
            cov2ds.handle.clone().binding(),
            radii.handle.clone().binding(),
            cum_tiles_hit.handle.clone().binding(),
            num_visible.handle.clone().binding(),
        ],
        &[
            tile_ids_unsorted.handle.clone().binding(),
            gaussian_per_intersect_unsorted.handle.clone().binding(),
        ],
        [num_points as u32, 1, 1],
    );

    let num_intersects = bitcast_tensor(BurnBack::int_gather(
        0,
        bitcast_tensor(cum_tiles_hit),
        BurnBack::int_add_scalar(bitcast_tensor(num_visible.clone()), -1),
    ));

    // We're sorting by tile ID, but we know beforehand what the maximum value
    // can be. We don't need to sort all the leading 0 bits!
    let num_tiles: u32 = tile_bounds[0] * tile_bounds[1];
    let bits = u32::BITS - num_tiles.leading_zeros();
    let (tile_ids_sorted, gaussian_per_intersect) = radix_argsort(
        client.clone(),
        bitcast_tensor(tile_ids_unsorted),
        bitcast_tensor(gaussian_per_intersect_unsorted),
        num_intersects.clone(),
        bits,
    );

    let tile_bins = BurnBack::int_zeros(
        Shape::new([tile_bounds[0] as usize, tile_bounds[1] as usize, 2]),
        &device,
    );
    GetTileBinEdges::new().execute(
        &client,
        (),
        &[
            tile_ids_sorted.handle.binding(),
            num_intersects.handle.clone().binding(),
        ],
        &[tile_bins.handle.clone().binding()],
        [max_intersects as u32, 1, 1],
    );

    let out_img = if forward_only {
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

    let final_index = if forward_only {
        None
    } else {
        Some(create_tensor::<u32, 2>(
            &client,
            &device,
            [img_size.y as usize, img_size.x as usize],
        ))
    };

    let mut out_handles = vec![out_img.handle.clone().binding()];

    if let Some(f) = final_index.as_ref() {
        out_handles.push(f.handle.clone().binding());
    }

    Rasterize::new(forward_only).execute(
        &client,
        shaders::rasterize::Uniforms::new(img_size.into(), background.into()),
        &[
            gaussian_per_intersect.handle.clone().binding(),
            tile_bins.handle.clone().binding(),
            xys.handle.clone().binding(),
            cov2ds.handle.clone().binding(),
            colors.handle.binding(),
        ],
        &out_handles,
        [img_size.x, img_size.y, 1],
    );

    (
        out_img,
        Aux {
            num_visible: Tensor::from_primitive(bitcast_tensor(num_visible)),
            num_intersects: Tensor::from_primitive(bitcast_tensor(num_intersects)),
            tile_bins: Tensor::from_primitive(bitcast_tensor(tile_bins)),
            radii: Tensor::from_primitive(bitcast_tensor(radii)),
            cov2ds: Tensor::from_primitive(bitcast_tensor(cov2ds)),
            gaussian_per_intersect: Tensor::from_primitive(bitcast_tensor(gaussian_per_intersect)),
            xys: Tensor::from_primitive(bitcast_tensor(xys)),
            final_index: final_index.map(|f| Tensor::from_primitive(bitcast_tensor(f))),
            remap_ids: Tensor::from_primitive(bitcast_tensor(remap_ids)),
        },
    )
}

impl Backend for BurnBack {
    fn render_gaussians(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self, 2>,
        log_scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
        colors: FloatTensor<Self, 2>,
        raw_opacity: FloatTensor<Self, 1>,
        background: glam::Vec3,
    ) -> (FloatTensor<Self, 3>, Aux<BurnBack>) {
        render_forward(
            camera,
            img_size,
            means,
            log_scales,
            quats,
            colors,
            raw_opacity,
            background,
            true,
        )
    }
}

#[derive(Debug, Clone)]
struct GaussianBackwardState {
    cam: Camera,
    background: Vec3,

    // Splat inputs.
    means: NodeID,
    scales: NodeID,
    quats: NodeID,
    colors: NodeID,
    raw_opacities: NodeID,
    sh_degree: usize,
    out_img: FloatTensor<BurnBack, 3>,
    aux: Aux<BurnBack>,
}

#[derive(Debug)]
struct RenderBackwards;

impl<C: CheckpointStrategy> Backend for Autodiff<BurnBack, C> {
    fn render_gaussians(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self, 2>,
        log_scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
        colors: FloatTensor<Self, 2>,
        raw_opacity: FloatTensor<Self, 1>,
        background: glam::Vec3,
    ) -> (FloatTensor<Self, 3>, Aux<BurnBack>) {
        let prep_nodes = RenderBackwards
            .prepare::<C>([
                means.node.clone(),
                log_scales.node.clone(),
                quats.node.clone(),
                colors.node.clone(),
                raw_opacity.node.clone(),
            ])
            .compute_bound()
            .stateful();

        // let forward_only = matches!(prep_nodes, OpsKind::UnTracked(_));
        let (out_img, aux) = render_forward(
            camera,
            img_size,
            means.clone().primitive,
            log_scales.clone().primitive,
            quats.clone().primitive,
            colors.clone().primitive,
            raw_opacity.clone().primitive,
            background,
            false,
        );

        let sh_degree = sh_basis_from_coeffs(colors.primitive.shape.dims[1] / 3);
        // Prepare a stateful operation with each variable node and corresponding graph.
        //
        // Each node can be fetched with `ops.parents` in the same order as defined here.
        match prep_nodes {
            OpsKind::Tracked(mut prep) => {
                let state = GaussianBackwardState {
                    // TODO: Respect checkpointing in this.
                    means: prep.checkpoint(&means),
                    scales: prep.checkpoint(&log_scales),
                    quats: prep.checkpoint(&quats),
                    colors: prep.checkpoint(&colors),
                    raw_opacities: prep.checkpoint(&raw_opacity),
                    aux: aux.clone(),
                    cam: camera.clone(),
                    background,
                    out_img: out_img.clone(),
                    sh_degree,
                };

                (prep.finish(state, out_img), aux)
            }
            OpsKind::UnTracked(prep) => {
                // When no node is tracked, we can just compute the original operation without
                // keeping any state.
                (prep.finish(out_img), aux)
            }
        }
    }
}

// Implement the backward trait for the given backend B, the node gradient being of rank D
// with three other gradients to calculate (means, colors, and opacity).
impl Backward<BurnBack, 3, 5> for RenderBackwards {
    // Our state that we must build during the forward pass to compute the backward pass.
    // (means)
    type State = GaussianBackwardState;

    fn backward(
        self,
        ops: Ops<Self::State, 5>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let _span = info_span!("render_gaussians backwards").entered();

        // // Get the nodes of each variable.
        let state = ops.state;

        let img_dimgs = state.out_img.shape.dims;
        let img_size = glam::uvec2(img_dimgs[1] as u32, img_dimgs[0] as u32);

        let v_output = grads.consume::<BurnBack, 3>(&ops.node);
        let camera = state.cam;

        DimCheck::new().check_dims(&v_output, [img_size.y.into(), img_size.x.into(), 4.into()]);

        let client = &v_output.client;
        let device = &v_output.device;

        let means = checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 2>>(state.means);
        let quats = checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 2>>(state.quats);
        let log_scales =
            checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 2>>(state.scales);
        let colors = checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 2>>(state.colors);

        let num_points = means.shape.dims[0];

        // TODO: Can't this be done for just visible points
        let v_xy = BurnBack::float_zeros(Shape::new([num_points, 2]), device);
        let v_conic = BurnBack::float_zeros(Shape::new([num_points, 4]), device);
        let v_colors = BurnBack::float_zeros(Shape::new([num_points, 4]), device);

        let aux = state.aux;

        RasterizeBackwards::new().execute(
            client,
            shaders::rasterize_backwards::Uniforms::new(img_size.into(), state.background.into()),
            &[
                aux.gaussian_per_intersect.into_primitive().handle.binding(),
                aux.tile_bins.into_primitive().handle.binding(),
                aux.xys.into_primitive().handle.binding(),
                aux.cov2ds.clone().into_primitive().handle.binding(),
                colors.handle.binding(),
                aux.radii.into_primitive().handle.binding(),
                aux.final_index.unwrap().into_primitive().handle.binding(),
                state.out_img.handle.binding(),
                v_output.handle.binding(),
            ],
            &[
                v_xy.handle.clone().binding(),
                v_conic.handle.clone().binding(),
                v_colors.handle.clone().binding(),
            ],
            [img_size.x, img_size.y, 1],
        );

        // TODO: Can't this be done for just visible points
        let v_means_agg = BurnBack::float_zeros(Shape::new([num_points, 4]), device);
        let v_scales_agg = BurnBack::float_zeros(Shape::new([num_points, 4]), device);
        let v_quats_agg = BurnBack::float_zeros(Shape::new([num_points, 4]), device);
        let v_coeffs_agg = BurnBack::float_zeros(
            Shape::new([num_points, num_sh_coeffs(state.sh_degree) * 3]),
            device,
        );
        let v_opac_agg = BurnBack::float_zeros(Shape::new([num_points]), device);

        ProjectBackwards::new().execute(
            client,
            shaders::project_backwards::Uniforms::new(
                camera.world_to_local(),
                camera.center(img_size).into(),
                img_size.into(),
            ),
            &[
                means.handle.binding(),
                log_scales.handle.binding(),
                quats.handle.binding(),
                aux.cov2ds.into_primitive().handle.binding(),
                v_xy.handle.binding(),
                v_conic.handle.binding(),
                v_colors.handle.binding(),
                aux.num_visible.into_primitive().handle.clone().binding(),
                aux.remap_ids.into_primitive().handle.clone().binding(),
            ],
            &[
                v_means_agg.handle.clone().binding(),
                v_scales_agg.handle.clone().binding(),
                v_quats_agg.handle.clone().binding(),
                v_coeffs_agg.handle.clone().binding(),
                v_opac_agg.handle.clone().binding(),
            ],
            [num_points as u32, 1, 1],
        );

        // Register gradients for parent nodes.
        // TODO: Optimise cases where only some gradients are tracked.
        let [mean_parent, log_scales_parent, quats_parent, coeffs_parent, raw_opacity_parent] =
            ops.parents;

        if let Some(node) = mean_parent {
            grads.register::<BurnBack, 2>(node.id, v_means_agg);
        }

        if let Some(node) = log_scales_parent {
            grads.register::<BurnBack, 2>(node.id, v_scales_agg);
        }

        if let Some(node) = quats_parent {
            grads.register::<BurnBack, 2>(node.id, v_quats_agg);
        }

        if let Some(node) = coeffs_parent {
            grads.register::<BurnBack, 2>(node.id, v_coeffs_agg);
        }

        if let Some(node) = raw_opacity_parent {
            grads.register::<BurnBack, 1>(node.id, v_opac_agg);
        }
    }
}

pub fn render<B: Backend>(
    camera: &Camera,
    img_size: glam::UVec2,
    means: Tensor<B, 2>,
    log_scales: Tensor<B, 2>,
    quats: Tensor<B, 2>,
    sh_coeffs: Tensor<B, 2>,
    raw_opacity: Tensor<B, 1>,
    background: glam::Vec3,
) -> (Tensor<B, 3>, Aux<BurnBack>) {
    let (img, aux) = B::render_gaussians(
        camera,
        img_size,
        means.clone().into_primitive(),
        log_scales.clone().into_primitive(),
        quats.clone().into_primitive(),
        sh_coeffs.clone().into_primitive(),
        raw_opacity.clone().into_primitive(),
        background,
    );
    (Tensor::from_primitive(img), aux)
}
