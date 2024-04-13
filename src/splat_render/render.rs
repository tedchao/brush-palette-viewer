use super::kernels::SplatKernel;
use super::prefix_sum::prefix_sum;
use super::radix_sort::radix_argsort;
use crate::camera::Camera;
use crate::splat_render::dim_check::DimCheck;
use crate::splat_render::kernels::{
    GetTileBinEdges, MapGaussiansToIntersect, ProjectBackwards, ProjectSplats, Rasterize,
    RasterizeBackwards,
};
use crate::splat_render::{create_buffer, create_tensor, read_buffer_to_u32};
use burn::backend::autodiff::NodeID;
use burn::tensor::ops::FloatTensorOps;
use burn::tensor::ops::IntTensorOps;
use burn::tensor::{Shape, Tensor};

use super::{generated_bindings, Aux, Backend, BurnBack, FloatTensor};
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

pub fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &data[i]);
    indices
}

impl Backend for BurnBack {
    fn render_gaussians(
        camera: &Camera,
        means: FloatTensor<Self, 2>,
        scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
        colors: FloatTensor<Self, 2>,
        opacity: FloatTensor<Self, 1>,
        background: glam::Vec3,
    ) -> (FloatTensor<Self, 3>, Aux<Self>) {
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
        let tile_width = generated_bindings::helpers::TILE_WIDTH;
        let img_size = [camera.width, camera.height];
        let tile_bounds = uvec2(
            camera.width.div_ceil(tile_width),
            camera.height.div_ceil(tile_width),
        );

        let client = means.client.clone();
        let device = means.device.clone();

        let num_points = means.shape.dims[0];
        let radii = create_tensor(&client, &device, [num_points]);
        let depths = create_buffer::<f32, 1>(&client, [num_points]);
        let xys = create_tensor(&client, &device, [num_points, 2]);
        let cov2ds = create_tensor(&client, &device, [num_points, 4]);
        let num_tiles_hit = create_tensor::<i32, 1>(&client, &device, [num_points]);

        ProjectSplats::execute(
            &client,
            generated_bindings::project_forward::Uniforms::new(
                camera.viewmatrix(),
                camera.focal().into(),
                camera.center().into(),
                img_size,
                tile_bounds.into(),
                tile_width,
                0.01,
            ),
            &[&means.handle, &scales.handle, &quats.handle],
            &[
                &xys.handle,
                &depths,
                &radii.handle,
                &cov2ds.handle,
                &num_tiles_hit.handle,
            ],
            [num_points as u32, 1, 1],
        );

        let cum_tiles_hit = prefix_sum(&client, &num_tiles_hit);

        // TODO: This is the only real CPU <-> GPU bottleneck, get around this somehow?
        #[allow(clippy::single_range_in_vec_init)]
        let last_elem = BurnBack::int_slice(cum_tiles_hit.clone(), [num_points - 1..num_points]);

        let num_intersects = *read_buffer_to_u32(&client, &last_elem.handle)
            .last()
            .unwrap() as usize;

        // Each intersection maps to a gaussian.
        let isect_ids_unsorted = create_tensor::<i32, 1>(&client, &device, [num_intersects]);
        let gaussian_ids_unsorted = create_tensor::<i32, 1>(&client, &device, [num_intersects]);

        // Dispatch one thread per point.
        MapGaussiansToIntersect::execute(
            &client,
            generated_bindings::map_gaussian_to_intersects::Uniforms::new(tile_bounds.into()),
            &[&xys.handle, &radii.handle, &cum_tiles_hit.handle, &depths],
            &[&isect_ids_unsorted.handle, &gaussian_ids_unsorted.handle],
            [num_points as u32, 1, 1],
        );

        let (isect_ids_sorted, gaussian_ids_sorted) = radix_argsort(
            client.clone(),
            isect_ids_unsorted.clone(),
            gaussian_ids_unsorted,
        );

        let tile_bins = BurnBack::int_zeros(
            Shape::new([tile_bounds[0] as usize, tile_bounds[1] as usize, 2]),
            &device,
        );
        GetTileBinEdges::execute(
            &client,
            (),
            &[&isect_ids_sorted.handle],
            &[&tile_bins.handle],
            [num_intersects as u32, 1, 1],
        );

        let out_img = create_tensor(
            &client,
            &device,
            [camera.height as usize, camera.width as usize, 4],
        );

        let final_index = create_tensor(
            &client,
            &device,
            [camera.height as usize, camera.width as usize],
        );

        Rasterize::execute(
            &client,
            generated_bindings::rasterize::Uniforms::new(img_size, background.into()),
            &[
                &gaussian_ids_sorted.handle,
                &tile_bins.handle,
                &xys.handle,
                &cov2ds.handle,
                &colors.handle,
                &opacity.handle,
            ],
            &[&out_img.handle, &final_index.handle],
            [camera.width, camera.height, 1],
        );

        (
            out_img,
            Aux {
                num_intersects: num_intersects as u32,
                tile_bins: Tensor::from_primitive(tile_bins.clone()),
                radii: Tensor::from_primitive(radii),
                cov2ds: Tensor::from_primitive(cov2ds),
                gaussian_ids_sorted: Tensor::from_primitive(gaussian_ids_sorted),
                xys: Tensor::from_primitive(xys),
                final_index: Tensor::from_primitive(final_index),
            },
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

        let (out_img, aux) = BurnBack::render_gaussians(
            camera,
            means.clone().primitive,
            scales.clone().primitive,
            quats.clone().primitive,
            colors.clone().primitive,
            opacity.clone().primitive,
            background,
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
        // // Get the nodes of each variable.
        let state = ops.state;
        let v_output = grads.consume::<BurnBack, 3>(&ops.node);
        let camera = state.cam;

        DimCheck::new().check_dims(
            &v_output,
            [camera.height.into(), camera.width.into(), 4.into()],
        );

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
        let img_size = [camera.width, camera.height];

        RasterizeBackwards::execute(
            client,
            generated_bindings::rasterize_backwards::Uniforms::new(
                img_size,
                state.background.into(),
            ),
            &[
                &aux.gaussian_ids_sorted.into_primitive().handle,
                &aux.tile_bins.into_primitive().handle,
                &aux.xys.into_primitive().handle,
                &aux.cov2ds.clone().into_primitive().handle,
                &colors.handle,
                &opacity.handle,
                &aux.final_index.into_primitive().handle,
                &state.out_img.handle,
                &v_output.handle,
            ],
            &[
                &v_xy.handle,
                &v_conic.handle,
                &v_colors.handle,
                &v_opacity.handle,
            ],
            [camera.width, camera.height, 1],
        );

        // TODO: Can't this be done for just visible points
        let v_means = create_tensor(client, device, [num_points, 4]);
        let v_scales = create_tensor(client, device, [num_points, 4]);
        let v_quats = create_tensor(client, device, [num_points, 4]);

        ProjectBackwards::execute(
            client,
            generated_bindings::project_backwards::Uniforms::new(
                camera.viewmatrix(),
                camera.center().into(),
                img_size,
            ),
            &[
                &means.handle,
                &scales.handle,
                &quats.handle,
                &aux.radii.into_primitive().handle,
                &aux.cov2ds.into_primitive().handle,
                &v_xy.handle,
                &v_conic.handle,
                &v_opacity.handle,
            ],
            &[&v_means.handle, &v_scales.handle, &v_quats.handle],
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
    means: Tensor<B, 2>,
    scales: Tensor<B, 2>,
    quats: Tensor<B, 2>,
    colors: Tensor<B, 2>,
    opacity: Tensor<B, 1>,
    background: glam::Vec3,
) -> (Tensor<B, 3>, Aux<BurnBack>) {
    let (img, aux) = B::render_gaussians(
        camera,
        means.clone().into_primitive(),
        scales.clone().into_primitive(),
        quats.clone().into_primitive(),
        colors.clone().into_primitive(),
        opacity.clone().into_primitive(),
        background,
    );
    (Tensor::from_primitive(img), aux)
}
