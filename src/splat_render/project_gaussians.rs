use std::io::Write;

use burn::{
    backend::{
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
            NodeID,
        },
        wgpu::{
            into_contiguous, DynamicKernel, DynamicKernelSource, FloatElement, GraphicsApi,
            IntElement, JitBackend, JitTensor, SourceTemplate, WgpuRuntime, WorkGroup,
        },
        Autodiff,
    },
    tensor::Shape,
};
use derive_new::new;
use glam::{uvec3, UVec3};
use crate::camera::Camera;
use super::{gen, Backend, FloatTensor};

#[derive(new, Debug)]
struct ProjectSplats {}

#[derive(new, Debug)]
struct MapGaussiansToIntersect {}

#[derive(new, Debug)]
struct GetTileBinEdges {}

#[derive(new, Debug)]
struct RasterizeForward {}

impl DynamicKernelSource for ProjectSplats {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::project_forward::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl DynamicKernelSource for MapGaussiansToIntersect {
    fn source(&self) -> SourceTemplate {
        todo!();
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

fn get_workgroup(executions: UVec3, group_size: UVec3) -> WorkGroup {
    let execs = (executions.as_vec3() / group_size.as_vec3()).ceil().as_uvec3();
    WorkGroup::new(execs.x, execs.y, execs.z)
}

pub fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &data[i]);
    indices
}

impl<G: GraphicsApi, F: FloatElement, I: IntElement> Backend for JitBackend<WgpuRuntime<G, F, I>> {
    fn render_gaussians(
        camera: &Camera,
        means: FloatTensor<Self, 2>,
        scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
        colors: FloatTensor<Self, 2>,
        opacity: FloatTensor<Self, 2>,
    ) -> FloatTensor<Self, 3> {
        println!("Project gaussians!");

        // Prolly gonna crash so yeah just
        // flush beforehand.
        std::io::stdout().flush().unwrap();

        // Preprocess tensors.
        assert!(means.device == scales.device && means.device == quats.device);
        let (means, scales, quats) = (
            into_contiguous(means),
            into_contiguous(scales),
            into_contiguous(quats),
        );
        let num_points = means.shape.dims[0];
        assert!(means.shape.dims[0] == num_points && scales.shape.dims[0] == num_points);

        let block_size = 16;
        let tile_bounds = [
            (camera.width + block_size - 1) / block_size,
            (camera.height + block_size - 1) / block_size,
        ];

        let intrins = 
            // TODO: Does this need the tan business?
            glam::vec4(
                (camera.width as f32) / (2.0 * camera.fovx.tan()),
                (camera.height as f32) / (2.0 * camera.fovy.tan()),
                (camera.width as f32) / 2.0,
                (camera.height as f32) / 2.0,
            );

        let client = means.client.clone();
        let device = means.device.clone();

        let create_empty_f32 = |dim| {
            let shape = Shape::new(dim);
            JitTensor::new(
                client.clone(),
                device.clone(),
                shape.clone(),
                client.empty(shape.num_elements() * core::mem::size_of::<F>()),
            )
        };

        let create_empty_i32 = |dim| -> JitTensor<WgpuRuntime<G, F, I>, I, 2> {
            let shape = Shape::new(dim);
            JitTensor::new(
                client.clone(),
                device.clone(),
                shape.clone(),
                client.empty(shape.num_elements() * core::mem::size_of::<I>()),
            )
        };

        let aux_data = client.create(bytemuck::bytes_of(&gen::helpers::InfoBinding {
            viewmat: camera.transform.inverse(),
            projmat: camera.proj_mat,
            intrins,
            img_size: [camera.width, camera.height],
            tile_bounds,
            glob_scale: 1.0,
            num_points: num_points as u32,
            clip_thresh: 0.001,
            block_width: block_size,
        }));

        // Create the output tensor primitive.

        // TODO: We might only need the data on the client for this not the tensor wrapper?
        let radii = create_empty_i32([num_points, 1]);
        let num_tiles_hit = create_empty_i32([num_points, 1]);
        let covs3d = create_empty_f32([num_points, 6]);
        let conics = create_empty_f32([num_points, 3]);
        let depths = create_empty_f32([num_points, 1]);
        let xys = create_empty_f32([num_points, 2]);
        let compensation = create_empty_f32([num_points, 1]);

        let workgroup = get_workgroup(uvec3(num_points as u32, 1, 1), uvec3(16, 1, 1));
        client.execute(
            Box::new(DynamicKernel::new(ProjectSplats::new(), workgroup)),
            &[
                // Input tensors.
                &means.handle,  // 0
                &scales.handle, // 1
                &quats.handle,  // 2
                // Output tensors.
                &covs3d.handle,        // 3
                &xys.handle,           // 4
                &depths.handle,        // 5
                &radii.handle,         // 6
                &conics.handle,        // 7
                &compensation.handle,  // 8
                &num_tiles_hit.handle, // 9
                // Aux data.
                &aux_data
            ],
        );

        // TODO: CPU emulation for now. Investigate if we can do without a cumulative sum?
        let num_tiles_hit = bytemuck::cast_vec::<u8, u32>(client.read(&num_tiles_hit.handle).read());

        let cum_tiles_hit: Vec<u32> = num_tiles_hit
            .into_iter()
            .scan(0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        let num_intersects = cum_tiles_hit.last().unwrap();

        // Mapping gaussians to sorted unique intersection IDs and tile bins used for fast rasterization.
        // We return both dsorted an unsorted versions of intersect IDs and gaussian IDs for testing purposes.

        // Returns:
        //     A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor}:

        //     - **isect_ids_sorted** (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        //     - **gaussian_ids_sorted** (Tensor): sorted Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        //     - **tile_bins** (Tensor): range of gaussians hit per tile.

        let cum_tiles_hit = client.create(bytemuck::cast_slice::<u32, u8>(&cum_tiles_hit));

        // TODO Num points -> num_total_intersects.

        let isect_ids = create_empty_i32([num_points, 1]);
        let gaussian_ids = create_empty_i32([num_points, 1]);

        let workgroup = get_workgroup(uvec3(num_points as u32, 1, 1), uvec3(16, 1, 1));
        client.execute(
            Box::new(DynamicKernel::new(MapGaussiansToIntersect::new(), workgroup)),
            &[
                // Input tensors.
                &xys.handle,
                &depths.handle,
                &radii.handle,
                &cum_tiles_hit,

                &isect_ids.handle,
                &gaussian_ids.handle,

                &aux_data
            ],
        );

        // TODO: WGSL Radix sort.
        let isect_ids = bytemuck::cast_vec::<u8, u32>(client.read(&isect_ids.handle).read());
        let gaussian_ids = bytemuck::cast_vec::<u8, u32>(client.read(&gaussian_ids.handle).read());
        let sorted_indices = argsort(&isect_ids);
        let isect_ids = sorted_indices.iter().copied().map(|x| isect_ids[x]).collect();
        let gaussian_ids_sorted = sorted_indices.iter().copied().map(|x| gaussian_ids[x]).collect();

        // Map sorted intersection IDs to tile bins which give the range of unique gaussian IDs belonging to each tile.
        // Expects that intersection IDs are sorted by increasing tile ID.
        // Indexing into tile_bins[tile_idx] returns the range (lower,upper) of gaussian IDs that hit tile_idx.
        // Args:
        //     num_intersects (int): total number of gaussian intersects.
        //     isect_ids_sorted (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        //     tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).
    
        // Returns:
        //     A Tensor:
        //     - **tile_bins** (Tensor): range of gaussians IDs hit per tile.
        
        // TODO: Num intersects.
        let workgroup = get_workgroup(uvec3(num_points as u32, 1, 1), uvec3(16, 1, 1));
        client.execute(
            Box::new(DynamicKernel::new(GetTileBinEdges::new(), workgroup)),
            &[
                // Input tensors.
                &isect_ids_sorted.handle,
                // Aux data.
                &aux_data
            ],
        );

        client.execute(
            Box::new(DynamicKernel::new(RasterizeForward::new(), workgroup)),
            &[
                // Input tensors.
                &gaussian_ids_sorted.handle,
                &tile_bins.handle,
                &xys.handle,
                &conics.handle,
                &colors.handle,
                &opacity.handle,
                // Aux data.
                &aux_data
            ],
        );

        out_img
    }
}

// Create our zero-sized type that will implement the Backward trait.
#[derive(Debug)]
struct ProjectSplatsBackward;

// Implement the backward trait for the given backend B, the node gradient being of rank D
// with three other gradients to calculate (means, colors, and opacity).
impl<B: Backend> Backward<B, 2, 3> for ProjectSplatsBackward {
    // Our state that we must build during the forward pass to compute the backward pass.
    //
    // Note that we could improve the performance further by only keeping the state of
    // tensors that are tracked, improving memory management, but for simplicity, we avoid
    // that part.

    //           (means, colors, opacity, ??, ???)
    type State = (NodeID, NodeID, NodeID);

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        // Get the nodes of each variable.
        let [node_means, node_colors, node_opacity] = ops.parents;

        // Fetch the gradient for the current node.
        // let grad = grads.consume::<B, 3>(&ops.node);

        // Set our state.
        let (means_state, colors_state, opacity_state) = ops.state;
        let means = checkpointer.retrieve_node_output(means_state);
        let colors = checkpointer.retrieve_node_output(colors_state);
        let opacity = checkpointer.retrieve_node_output(opacity_state);

        // TODO: Some actual gradients ahh
        let grad_colors = colors;
        let grad_means = means;
        let grad_opacity = opacity;

        // Register the gradient for each variable based on whether they are marked as
        // `tracked`.
        if let Some(node) = node_means {
            grads.register::<B, 2>(node, grad_means);
        }
        if let Some(node) = node_colors {
            grads.register::<B, 2>(node, grad_colors);
        }
        if let Some(node) = node_opacity {
            grads.register::<B, 1>(node, grad_opacity);
        }
    }
}

impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    fn render_gaussians(
        camera: &Camera,
        means: FloatTensor<Self, 2>,
        scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
    ) -> FloatTensor<Self, 2> {
        // Prepare a stateful operation with each variable node and corresponding graph.
        //
        // Each node can be fetched with `ops.parents` in the same order as defined here.
        match ProjectSplatsBackward
            .prepare::<C>(
                [means.node.clone(), scales.node.clone(), quats.node.clone()],
                [
                    means.graph.clone(),
                    scales.graph.clone(),
                    quats.graph.clone(),
                ],
            )
            // Marks the operation as compute bound, meaning it will save its
            // state instead of recomputing itself during checkpointing
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                // When at least one node is tracked, we should register our backward step.

                // The state consists of what will be needed for this operation's backward pass.
                // Since we need the parents' outputs, we must checkpoint their ids to retrieve their node
                // output at the beginning of the backward. We can also save utilitary data such as the bias shape
                // If we also need this operation's output, we can either save it in the state or recompute it
                // during the backward pass. Here we choose to save it in the state because it's a compute bound operation.
                let means_state = prep.checkpoint(&means);
                let scale_state = prep.checkpoint(&scales);
                let quat_state = prep.checkpoint(&quats);

                let output = B::render_gaussians(
                    camera,
                    means.primitive.clone(),
                    scales.primitive.clone(),
                    quats.primitive,
                );

                let state = (means_state, scale_state, quat_state);
                prep.finish(state, output)
            }
            OpsKind::UnTracked(prep) => {
                // When no node is tracked, we can just compute the original operation without
                // keeping any state.
                let output =
                    B::render_gaussians(camera, means.primitive, scales.primitive, quats.primitive);
                prep.finish(output)
            }
        }
    }
}
