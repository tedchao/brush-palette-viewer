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

use crate::camera::Camera;

use super::{gen, Backend, FloatTensor};

// Define our kernel type with workgroup information.
#[derive(new, Debug)]
struct RenderSplats2D {}

// Implement the dynamic kernel trait for our kernel type.
impl DynamicKernelSource for RenderSplats2D {
    fn source(&self) -> SourceTemplate {
        // Extend our raw kernel with workgroup size information using the
        // `SourceTemplate` trait.
        SourceTemplate::new(gen::project_forward::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl<G: GraphicsApi, F: FloatElement, I: IntElement> Backend for JitBackend<WgpuRuntime<G, F, I>> {
    fn project_splats(
        camera: &Camera,
        means: FloatTensor<Self, 2>,
        scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
    ) -> FloatTensor<Self, 2> {
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

        let block_width = 16;
        let tile_bounds = [
            (camera.width + block_width - 1) / block_width,
            (camera.height + block_width - 1) / block_width,
        ];

        let info = gen::project_forward::InfoBinding {
            viewmat: camera.transform,
            projmat: camera.proj_mat,
            intrins: camera.intrins(),
            img_size: [camera.width, camera.height],
            tile_bounds,
            glob_scale: 1.0,
            num_points: num_points as u32,
            clip_thresh: 0.001,
            block_width,
        };

        let client = means.client.clone();
        let device = means.device.clone();

        let create_empty = |dim| {
            let shape = Shape::new(dim);
            JitTensor::new(
                client.clone(),
                device.clone(),
                shape.clone(),
                client.empty(shape.num_elements() * core::mem::size_of::<F>()),
            )
        };

        // Create the output tensor primitive.
        let radii = create_empty([num_points, 1]);
        let num_tiles_hit = create_empty([num_points, 1]);
        let covs3d = create_empty([num_points, 6]);
        let conics = create_empty([num_points, 3]);
        let depths = create_empty([num_points, 1]);
        let xys = create_empty([num_points, 2]);
        let compensation = create_empty([num_points, 1]);

        // Imagine ExampleStruct had a bunch of arguments.
        let bytes = bytemuck::bytes_of(&info);
        let info_data = means.client.create(bytes);

        // Create the kernel.
        let kernel = RenderSplats2D::new();

        // Declare the wgsl workgroup with the number of blocks in x, y and z.
        let blocks_needed_in_x = f32::ceil(num_points as f32 / block_width as f32) as u32;
        let workgroup = WorkGroup::new(blocks_needed_in_x, 1, 1);

        // Execute lazily the kernel with the launch information and the given buffers.
        println!("Execpute WGSL compute shader.");
        means.client.execute(
            Box::new(DynamicKernel::new(kernel, workgroup)),
            &[
                // Input tensors.
                &means.handle,  // 0
                &scales.handle, // 1
                &quats.handle,  // 2
                // Output tensors.
                &radii.handle,         // 3
                &num_tiles_hit.handle, // 4
                &covs3d.handle,        // 5
                &conics.handle,        // 6
                &depths.handle,        // 7
                &xys.handle,           // 8
                &compensation.handle,  // 9
                // Aux data.
                &info_data, // 10
            ],
        );

        println!("Execputed WGSL compute shader.");

        // Return the output tensor.
        // TODO: How to return the rest??
        xys
        // ProjectionOutput {
        //     covs3d,
        //     xys,
        //     depths,
        //     radii,
        //     conics,
        //     compensation,
        //     num_tiles_hit,
        // }
    }
}

// Create our zero-sized type that will implement the Backward trait.
#[derive(Debug)]
struct RenderSplatsBackward;

// Implement the backward trait for the given backend B, the node gradient being of rank D
// with three other gradients to calculate (means, colors, and opacity).
impl<B: Backend> Backward<B, 2, 3> for RenderSplatsBackward {
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
    fn project_splats(
        camera: &Camera,
        means: FloatTensor<Self, 2>,
        scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
    ) -> FloatTensor<Self, 2> {
        // Prepare a stateful operation with each variable node and corresponding graph.
        //
        // Each node can be fetched with `ops.parents` in the same order as defined here.
        match RenderSplatsBackward
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

                let output = B::project_splats(
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
                    B::project_splats(camera, means.primitive, scales.primitive, quats.primitive);
                prep.finish(output)
            }
        }
    }
}
