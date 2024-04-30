use super::kernels::SplatKernel;
use super::prefix_sum::prefix_sum;
use super::radix_sort::radix_argsort;
use crate::camera::Camera;
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
    scales: JitTensor<BurnRuntime, f32, 2>,
    quats: JitTensor<BurnRuntime, f32, 2>,
    colors: JitTensor<BurnRuntime, f32, 2>,
    opacity: JitTensor<BurnRuntime, f32, 1>,
    background: glam::Vec3,
    forward_only: bool,
) -> (JitTensor<BurnRuntime, f32, 3>, Aux<BurnBack>) {
    let _span = info_span!("render_gaussians").entered();

    let (means, scales, quats, colors, opacity) = (
        into_contiguous(means.clone()),
        into_contiguous(scales.clone()),
        into_contiguous(quats.clone()),
        into_contiguous(colors.clone()),
        into_contiguous(opacity.clone()),
    );

    DimCheck::new()
        .check_dims(&means, ["D".into(), 4.into()])
        .check_dims(&scales, ["D".into(), 4.into()])
        .check_dims(&quats, ["D".into(), 4.into()])
        .check_dims(&colors, ["D".into(), 4.into()])
        .check_dims(&opacity, ["D".into()]);

    // Divide screen into blocks.
    let tile_width = shaders::helpers::TILE_WIDTH;
    let tile_bounds = uvec2(
        img_size.x.div_ceil(tile_width),
        img_size.y.div_ceil(tile_width),
    );

    let client = means.client.clone();
    let device = means.device.clone();

    let num_points = means.shape.dims[0];

    let xys = create_tensor::<f32, 2>(&client, &device, [num_points, 2]);
    let depths = create_tensor::<f32, 1>(&client, &device, [num_points]);
    let radii = create_tensor::<u32, 1>(&client, &device, [num_points]);
    let cov2ds = create_tensor::<f32, 2>(&client, &device, [num_points, 4]);
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
        ),
        &[
            means.handle.binding(),
            scales.handle.binding(),
            quats.handle.binding(),
            opacity.handle.clone().binding(),
        ],
        &[
            compact_ids.handle.clone().binding(),
            remap_ids.handle.clone().binding(),
            xys.handle.clone().binding(),
            depths.handle.clone().binding(),
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
    // Ideally render gaussians in chunks, but that might be hell with the backward pass.
    let max_intersects = num_points * 3;

    // Each intersection maps to a gaussian.
    let tile_ids_unsorted = create_tensor::<u32, 1>(&client, &device, [max_intersects]);
    let gaussian_ids_depth_sorted = create_tensor::<u32, 1>(&client, &device, [max_intersects]);

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
            gaussian_ids_depth_sorted.handle.clone().binding(),
        ],
        [num_points as u32, 1, 1],
    );

    let num_intersect = bitcast_tensor(BurnBack::int_gather(
        0,
        bitcast_tensor(cum_tiles_hit),
        bitcast_tensor(num_visible),
    ));

    // We're sorting by tile ID, but we know beforehand what the maximum value
    // can be. We don't need to sort all the leading 0 bits!
    let num_tiles: u32 = tile_bounds[0] * tile_bounds[1];
    let bits = u32::BITS - num_tiles.leading_zeros();
    let (tile_ids_sorted, gaussian_ids_sorted) = radix_argsort(
        client.clone(),
        bitcast_tensor(tile_ids_unsorted),
        bitcast_tensor(gaussian_ids_depth_sorted),
        num_intersect.clone(),
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
            num_intersect.handle.clone().binding(),
        ],
        &[tile_bins.handle.clone().binding()],
        [max_intersects as u32, 1, 1],
    );

    let out_img = if forward_only {
        create_tensor(
            &client,
            &device,
            [img_size.y as usize, img_size.x as usize, 4],
        )
    } else {
        // Channels are packed into 1 byte - aka one u32 element.
        create_tensor(
            &client,
            &device,
            [img_size.y as usize, img_size.x as usize, 1],
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
            gaussian_ids_sorted.handle.clone().binding(),
            remap_ids.handle.clone().binding(),
            tile_bins.handle.clone().binding(),
            xys.handle.clone().binding(),
            cov2ds.handle.clone().binding(),
            colors.handle.binding(),
            opacity.handle.binding(),
        ],
        &out_handles,
        [img_size.x, img_size.y, 1],
    );

    // TODO: Deal with sparsity in backwards pass.
    (
        out_img,
        Aux {
            num_intersects: max_intersects as u32,
            tile_bins: Tensor::from_primitive(bitcast_tensor(tile_bins)),
            radii: Tensor::from_primitive(bitcast_tensor(radii)),
            cov2ds: Tensor::from_primitive(bitcast_tensor(cov2ds)),
            gaussian_ids_sorted: Tensor::from_primitive(bitcast_tensor(gaussian_ids_sorted)),
            xys: Tensor::from_primitive(bitcast_tensor(xys)),
            final_index: final_index.map(|f| Tensor::from_primitive(bitcast_tensor(f))),
        },
    )
}

impl Backend for BurnBack {
    fn render_gaussians(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self, 2>,
        scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
        colors: FloatTensor<Self, 2>,
        opacity: FloatTensor<Self, 1>,
        background: glam::Vec3,
    ) -> (FloatTensor<Self, 3>, Aux<BurnBack>) {
        render_forward(
            camera, img_size, means, scales, quats, colors, opacity, background, true,
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
    opacity: NodeID,

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
        scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
        colors: FloatTensor<Self, 2>,
        opacity: FloatTensor<Self, 1>,
        background: glam::Vec3,
    ) -> (FloatTensor<Self, 3>, Aux<BurnBack>) {
        let prep_nodes = RenderBackwards
            .prepare::<C>([
                means.node.clone(),
                scales.node.clone(),
                quats.node.clone(),
                colors.node.clone(),
                opacity.node.clone(),
            ])
            .compute_bound()
            .stateful();

        let forward_only = matches!(prep_nodes, OpsKind::UnTracked(_));
        let (out_img, aux) = render_forward(
            camera,
            img_size,
            means.clone().primitive,
            scales.clone().primitive,
            quats.clone().primitive,
            colors.clone().primitive,
            opacity.clone().primitive,
            background,
            forward_only,
        );

        // Prepare a stateful operation with each variable node and corresponding graph.
        //
        // Each node can be fetched with `ops.parents` in the same order as defined here.
        match prep_nodes {
            OpsKind::Tracked(mut prep) => {
                let state = GaussianBackwardState {
                    // TODO: Respect checkpointing in this.
                    means: prep.checkpoint(&means),
                    scales: prep.checkpoint(&scales),
                    quats: prep.checkpoint(&quats),
                    colors: prep.checkpoint(&colors),
                    opacity: prep.checkpoint(&opacity),
                    aux: aux.clone(),
                    cam: camera.clone(),
                    background,
                    out_img: out_img.clone(),
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
        let scales = checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 2>>(state.scales);
        let colors = checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 2>>(state.colors);
        let opacity = checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 1>>(state.opacity);

        let num_points = means.shape.dims[0];

        // TODO: Can't this be done for just visible points
        let v_xy = BurnBack::float_zeros(Shape::new([num_points, 2]), device);
        let v_conic = BurnBack::float_zeros(Shape::new([num_points, 4]), device);
        let v_colors = BurnBack::float_zeros(Shape::new([num_points, 4]), device);
        let v_opacity = BurnBack::float_zeros(Shape::new([num_points]), device);

        let aux = state.aux;

        RasterizeBackwards::new().execute(
            client,
            shaders::rasterize_backwards::Uniforms::new(img_size.into(), state.background.into()),
            &[
                aux.gaussian_ids_sorted.into_primitive().handle.binding(),
                aux.tile_bins.into_primitive().handle.binding(),
                aux.xys.into_primitive().handle.binding(),
                aux.cov2ds.clone().into_primitive().handle.binding(),
                colors.handle.binding(),
                opacity.handle.binding(),
                aux.final_index.unwrap().into_primitive().handle.binding(),
                state.out_img.handle.binding(),
                v_output.handle.binding(),
            ],
            &[
                v_xy.handle.clone().binding(),
                v_conic.handle.clone().binding(),
                v_colors.handle.clone().binding(),
                v_opacity.handle.clone().binding(),
            ],
            [img_size.x, img_size.y, 1],
        );

        // TODO: Can't this be done for just visible points
        let v_means = create_tensor(client, device, [num_points, 4]);
        let v_scales = create_tensor(client, device, [num_points, 4]);
        let v_quats = create_tensor(client, device, [num_points, 4]);

        ProjectBackwards::new().execute(
            client,
            shaders::project_backwards::Uniforms::new(
                camera.world_to_local(),
                camera.center(img_size).into(),
                img_size.into(),
            ),
            &[
                means.handle.binding(),
                scales.handle.binding(),
                quats.handle.binding(),
                aux.radii.into_primitive().handle.binding(),
                aux.cov2ds.into_primitive().handle.binding(),
                v_xy.handle.binding(),
                v_conic.handle.binding(),
                v_opacity.handle.clone().binding(),
            ],
            &[
                v_means.handle.clone().binding(),
                v_scales.handle.clone().binding(),
                v_quats.handle.clone().binding(),
            ],
            [num_points as u32, 1, 1],
        );

        // Register gradients for parent nodes.
        // TODO: Optimise cases where only some gradients are tracked.
        let [mean_parent, scales_parent, quats_parent, colors_parent, opacity_parent] = ops.parents;

        if let Some(node) = mean_parent {
            grads.register::<BurnBack, 2>(node.id, v_means);
        }

        if let Some(node) = scales_parent {
            grads.register::<BurnBack, 2>(node.id, v_scales);
        }

        if let Some(node) = quats_parent {
            grads.register::<BurnBack, 2>(node.id, v_quats);
        }

        if let Some(node) = colors_parent {
            grads.register::<BurnBack, 2>(node.id, v_colors);
        }

        if let Some(node) = opacity_parent {
            grads.register::<BurnBack, 1>(node.id, v_opacity);
        }
    }
}

pub fn render<B: Backend>(
    camera: &Camera,
    img_size: glam::UVec2,
    means: Tensor<B, 2>,
    scales: Tensor<B, 2>,
    quats: Tensor<B, 2>,
    colors: Tensor<B, 2>,
    opacity: Tensor<B, 1>,
    background: glam::Vec3,
) -> (Tensor<B, 3>, Aux<BurnBack>) {
    let (img, aux) = B::render_gaussians(
        camera,
        img_size,
        means.clone().into_primitive(),
        scales.clone().into_primitive(),
        quats.clone().into_primitive(),
        colors.clone().into_primitive(),
        opacity.clone().into_primitive(),
        background,
    );
    (Tensor::from_primitive(img), aux)
}
