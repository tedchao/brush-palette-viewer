use super::kernels::SplatKernel;
use crate::camera::Camera;
use crate::splat_render::gen::rasterize;
use crate::splat_render::kernels::{
    GetTileBinEdges, MapGaussiansToIntersect, ProjectBackwards, ProjectSplats, Rasterize,
    RasterizeBackwards,
};
use crate::splat_render::{create_buffer, create_tensor, read_buffer_to_u32};
use burn::backend::autodiff::NodeID;
use burn::tensor::ops::IntTensor;
use burn::tensor::Tensor;

use super::{gen, Backend, BurnBack, FloatTensor};
use crate::splat_render::BufferAlloc;
use burn::{
    backend::{
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        wgpu::{
            into_contiguous, FloatElement, GraphicsApi, IntElement, JitBackend, JitTensor,
            WgpuRuntime,
        },
        Autodiff,
    },
    tensor::Shape,
};
use glam::{uvec2, Vec3};

pub fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &data[i]);
    indices
}

impl<G: GraphicsApi, F: FloatElement, I: IntElement> Backend for JitBackend<WgpuRuntime<G, F, I>> {
    fn render_gaussians(
        _camera: &Camera,
        _means: FloatTensor<Self, 2>,
        _scales: FloatTensor<Self, 2>,
        _quats: FloatTensor<Self, 2>,
        _colors: FloatTensor<Self, 2>,
        _opacity: FloatTensor<Self, 1>,
        _background: glam::Vec3,
    ) -> FloatTensor<Self, 3> {
        // Implement inference only version. This shouldn't be hard, but burn makes it a bit annoying!
        todo!();
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

    // Calculated state.
    radii: FloatTensor<BurnBack, 1>,
    compensation: FloatTensor<BurnBack, 1>,

    xys: FloatTensor<BurnBack, 2>,
    conics: FloatTensor<BurnBack, 2>,
    out_img: FloatTensor<BurnBack, 3>,

    gaussian_ids_sorted: IntTensor<BurnBack, 1>,
    tile_bins: IntTensor<BurnBack, 2>,
    final_index: IntTensor<BurnBack, 2>,
}

#[derive(Debug)]
struct RenderBackwards;

impl<C: CheckpointStrategy> Backend for Autodiff<BurnBack, C> {
    fn render_gaussians(
        camera: &Camera,
        means_diff: FloatTensor<Self, 2>,
        scales_diff: FloatTensor<Self, 2>,
        quats_diff: FloatTensor<Self, 2>,
        colors_diff: FloatTensor<Self, 2>,
        opacity_diff: FloatTensor<Self, 1>,
        background: glam::Vec3,
    ) -> FloatTensor<Self, 3> {
        let prep_nodes = RenderBackwards
            .prepare::<C>(
                [
                    means_diff.node.clone(),
                    scales_diff.node.clone(),
                    quats_diff.node.clone(),
                    colors_diff.node.clone(),
                    opacity_diff.node.clone(),
                ],
                [
                    means_diff.graph.clone(),
                    scales_diff.graph.clone(),
                    quats_diff.graph.clone(),
                    colors_diff.graph.clone(),
                    opacity_diff.graph.clone(),
                ],
            )
            .compute_bound()
            .stateful();

        let (means, scales, quats, colors, opacity) = (
            into_contiguous(means_diff.clone().primitive),
            into_contiguous(scales_diff.clone().primitive),
            into_contiguous(quats_diff.clone().primitive),
            into_contiguous(colors_diff.clone().primitive),
            into_contiguous(opacity_diff.clone().primitive),
        );

        assert!(
            means.device == scales.device
                && means.device == quats.device
                && means.device == colors.device
                && means.device == opacity.device
        );

        let num_points = means.shape.dims[0];
        let batches = [
            means.shape.dims[0],
            scales.shape.dims[0],
            quats.shape.dims[0],
            colors.shape.dims[0],
            opacity.shape.dims[0],
        ];
        assert!(batches.iter().all(|&x| x == num_points));

        // Atm these have to be dim=4, to be compatible
        // with wgsl alignment.
        assert!(means.shape.dims[1] == 4);
        assert!(scales.shape.dims[1] == 4);
        // 4D quaternions.
        assert!(quats.shape.dims[1] == 4);

        // Divide screen into blocks.
        let block_width = rasterize::GROUP_DIM;
        let img_size = [camera.width, camera.height];
        let tile_bounds = uvec2(
            camera.height.div_ceil(block_width),
            camera.height.div_ceil(block_width),
        );

        let intrins = glam::vec4(
            camera.focal().x,
            camera.focal().y,
            camera.center().x,
            camera.center().y,
        );

        let client = means.client.clone();
        let device = means.device.clone();

        let depths = create_buffer::<f32, 1>(&client, [num_points], BufferAlloc::Empty);
        let xys = create_tensor(&client, &device, [num_points, 2], BufferAlloc::Empty);
        let conics = create_tensor(&client, &device, [num_points, 4], BufferAlloc::Empty);
        let radii = create_tensor(&client, &device, [num_points], BufferAlloc::Empty);
        let compensation = create_tensor(&client, &device, [num_points], BufferAlloc::Empty);
        let num_tiles_hit = create_buffer::<u32, 1>(&client, [num_points], BufferAlloc::Empty);

        ProjectSplats::execute(
            &client,
            gen::project_forward::Uniforms::new(
                num_points as u32,
                camera.viewmatrix(),
                intrins,
                img_size,
                tile_bounds.into(),
                1.0,
                0.001,
                block_width,
            ),
            [&means.handle, &scales.handle, &quats.handle],
            [
                &xys.handle,
                &depths,
                &radii.handle,
                &conics.handle,
                &compensation.handle,
                &num_tiles_hit,
            ],
            [num_points as u32, 1, 1],
        );

        // TODO: CPU emulation for now. Investigate if we can do without a cumulative sum?

        // Read num tiles to CPU.
        let num_tiles_hit = read_buffer_to_u32(&client, &num_tiles_hit);
        // Calculate cumulative sum.
        let cum_tiles_hit: Vec<u32> = num_tiles_hit
            .into_iter()
            .scan(0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect();
        let num_intersects = *cum_tiles_hit.last().unwrap() as usize;

        // Reupload to GPU.
        let cum_tiles_hit = client.create(bytemuck::cast_slice::<u32, u8>(&cum_tiles_hit));

        // Each intersection maps to a gaussian.
        let isect_ids_unsorted =
            create_buffer::<u32, 1>(&client, [num_intersects], BufferAlloc::Empty);
        let gaussian_ids_unsorted =
            create_buffer::<u32, 1>(&client, [num_intersects], BufferAlloc::Empty);

        // Dispatch one thread per point.
        MapGaussiansToIntersect::execute(
            &client,
            gen::map_gaussian_to_intersects::Uniforms::new(
                num_points as u32,
                tile_bounds.into(),
                block_width,
            ),
            [&xys.handle, &depths, &radii.handle, &cum_tiles_hit],
            [&isect_ids_unsorted, &gaussian_ids_unsorted],
            [num_points as u32, 1, 1],
        );

        // TODO: WGSL Radix sort.
        let isect_ids_unsorted = read_buffer_to_u32(&client, &isect_ids_unsorted);
        let gaussian_ids_unsorted = read_buffer_to_u32(&client, &gaussian_ids_unsorted);
        let sorted_indices = argsort(&isect_ids_unsorted);

        let isect_ids_sorted: Vec<_> = sorted_indices
            .iter()
            .copied()
            .map(|x| isect_ids_unsorted[x])
            .collect();
        let gaussian_ids_sorted: Vec<_> = sorted_indices
            .iter()
            .copied()
            .map(|x| gaussian_ids_unsorted[x])
            .collect();

        let isect_ids_sorted = client.create(bytemuck::cast_slice::<u32, u8>(&isect_ids_sorted));
        let tile_bins = create_tensor(
            &client,
            &device,
            [(tile_bounds[0] * tile_bounds[1]) as usize, 2],
            BufferAlloc::Zeros,
        );

        GetTileBinEdges::execute(
            &client,
            gen::get_tile_bin_edges::Uniforms::new(num_intersects as u32),
            [&isect_ids_sorted],
            [&tile_bins.handle],
            [num_intersects as u32, 1, 1],
        );

        let out_img = create_tensor(
            &client,
            &device,
            [camera.height as usize, camera.width as usize, 4],
            BufferAlloc::Zeros,
        );
        let final_index = create_tensor(
            &client,
            &device,
            [camera.height as usize, camera.width as usize],
            BufferAlloc::Zeros,
        );

        let gaussian_ids_sorted = JitTensor::new(
            client.clone(),
            device.clone(),
            Shape::new([num_intersects]),
            client.create(bytemuck::cast_slice::<u32, u8>(&gaussian_ids_sorted)),
        );

        Rasterize::execute(
            &client,
            gen::rasterize::Uniforms::new(tile_bounds.into(), background.into(), img_size),
            [
                &gaussian_ids_sorted.handle,
                &tile_bins.handle,
                &xys.handle,
                &conics.handle,
                &colors.handle,
                &opacity.handle,
            ],
            [&out_img.handle, &final_index.handle],
            [camera.height, camera.width, 1],
        );

        // Prepare a stateful operation with each variable node and corresponding graph.
        //
        // Each node can be fetched with `ops.parents` in the same order as defined here.
        match prep_nodes {
            OpsKind::Tracked(mut prep) => {
                let state = GaussianBackwardState {
                    // TODO: Respect checkpointing in this.
                    means: prep.checkpoint(&means_diff),
                    scales: prep.checkpoint(&scales_diff),
                    quats: prep.checkpoint(&quats_diff),
                    colors: prep.checkpoint(&colors_diff),
                    opacity: prep.checkpoint(&opacity_diff),
                    radii,
                    compensation,
                    cam: camera.clone(),
                    background,
                    out_img: out_img.clone(),
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    conics,
                    final_index,
                };

                prep.finish(state, out_img)
            }
            OpsKind::UnTracked(prep) => {
                // When no node is tracked, we can just compute the original operation without
                // keeping any state.
                prep.finish(out_img)
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
        let block_size = 16;

        let state = ops.state;
        let v_output = grads.consume::<BurnBack, 3>(&ops.node);
        let camera = state.cam;
        let tile_bounds = uvec2(
            camera.height.div_ceil(block_size),
            camera.height.div_ceil(block_size),
        );

        assert!(v_output.shape.dims == [camera.height as usize, camera.width as usize, 4]);

        let client = v_output.client.clone();
        let device = v_output.device.clone();

        let intrins = glam::vec4(
            camera.focal().x,
            camera.focal().y,
            camera.center().x,
            camera.center().y,
        );

        let create_empty_f32 = |dim| -> FloatTensor<BurnBack, 2> {
            let shape = Shape::new(dim);
            JitTensor::new(
                client.clone(),
                device.clone(),
                shape.clone(),
                client.create(&vec![0; shape.num_elements() * core::mem::size_of::<f32>()]),
            )
        };
        let create_empty_f32_1d = |dim| -> FloatTensor<BurnBack, 1> {
            let shape = Shape::new(dim);
            JitTensor::new(
                client.clone(),
                device.clone(),
                shape.clone(),
                client.create(&vec![0; shape.num_elements() * core::mem::size_of::<f32>()]),
            )
        };

        let means = checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 2>>(state.means);
        let quats = checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 2>>(state.quats);
        let scales = checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 2>>(state.scales);
        let colors = checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 2>>(state.colors);
        let opacity = checkpointer.retrieve_node_output::<FloatTensor<BurnBack, 1>>(state.opacity);

        let num_points = means.shape.dims[0];
        let v_opacity = create_empty_f32_1d([num_points]);

        let v_colors = create_empty_f32([num_points, 4]);
        let v_conic = create_empty_f32([num_points, 4]);
        let v_xy = create_empty_f32([num_points, 2]);

        RasterizeBackwards::execute(
            &client,
            gen::rasterize_backwards::Uniforms::new(
                [camera.height, camera.width],
                tile_bounds.into(),
                state.background.into(),
            ),
            [
                &state.gaussian_ids_sorted.handle,
                &state.tile_bins.handle,
                &state.xys.handle,
                &state.conics.handle,
                &colors.handle,
                &opacity.handle,
                &state.final_index.handle,
                &state.out_img.handle,
                &v_output.handle,
            ],
            [
                &v_opacity.handle,
                &v_conic.handle,
                &v_xy.handle,
                &v_colors.handle,
            ],
            [camera.height, camera.width, 1],
        );

        let v_means = create_empty_f32([num_points, 4]);
        let v_scales = create_empty_f32([num_points, 4]);
        let v_quats = create_empty_f32([num_points, 4]);

        ProjectBackwards::execute(
            &client,
            gen::project_backwards::Uniforms::new(
                num_points as u32,
                1.0,
                camera.viewmatrix(),
                intrins,
                [camera.height, camera.width],
            ),
            [
                &means.handle,
                &scales.handle,
                &quats.handle,
                &state.radii.handle,
                &state.conics.handle,
                &state.compensation.handle,
                &v_xy.handle,
                &v_conic.handle,
            ],
            [&v_means.handle, &v_scales.handle, &v_quats.handle],
            [num_points as u32, 1, 1],
        );

        // Register gradients for parent nodes.
        // TODO: Optimise cases where only some gradients are tracked.
        let [mean_parent, scales_parent, quats_parent, colors_parent, opacity_parent] = ops.parents;

        if let Some(node) = mean_parent {
            grads.register::<BurnBack, 2>(node, v_means);
        }

        if let Some(node) = scales_parent {
            grads.register::<BurnBack, 2>(node, v_scales);
        }

        if let Some(node) = quats_parent {
            grads.register::<BurnBack, 2>(node, v_quats);
        }

        if let Some(node) = colors_parent {
            grads.register::<BurnBack, 2>(node, v_colors);
        }

        if let Some(node) = opacity_parent {
            grads.register::<BurnBack, 1>(node, v_opacity);
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
) -> Tensor<B, 3> {
    let img = B::render_gaussians(
        camera,
        means.clone().into_primitive(),
        scales.clone().into_primitive(),
        quats.clone().into_primitive(),
        colors.clone().into_primitive(),
        opacity.clone().into_primitive(),
        background,
    );
    Tensor::from_primitive(img)
}
