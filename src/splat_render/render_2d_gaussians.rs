use burn::{
    backend::{
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
            NodeID,
        },
        wgpu::{
            into_contiguous, DynamicKernel, DynamicKernelSource, FloatElement, JitTensor,
            SourceTemplate, WorkGroup,
        },
    },
    tensor::Shape,
};
use derive_new::new;

use super::{Backend, FloatTensor};

// Define our kernel type with workgroup information.
#[derive(new, Debug)]
struct RenderSplats2D {
    workgroup_size_x: usize,
    workgroup_size_y: usize,
}

// Implement the dynamic kernel trait for our kernel type.
impl DynamicKernelSource for RenderSplats2D {
    fn source(&self) -> SourceTemplate {
        let source = SourceTemplate::new(include_str!("./forward.wgsl"));
        // Extend our raw kernel with workgroup size information using the
        // `SourceTemplate` trait.
        source
            .register("workgroup_size_x", self.workgroup_size_x.to_string())
            .register("workgroup_size_y", self.workgroup_size_y.to_string())
            .register("int", "i32")
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

pub(crate) fn forward<B: Backend, F: FloatElement>(
    resolution: (u32, u32),
    xys: FloatTensor<B, 2>,
    shs: FloatTensor<B, 3>,
    opacity: FloatTensor<B, 1>,
) -> FloatTensor<B, 3> {
    // Define workgroup size, hardcoded for simplicity.
    let workgroup_size_x = 16;
    let workgroup_size_y = 16;
    assert!(xys.device == shs.device && xys.device == opacity.device);

    // For simplicity, make sure each tensor is continuous.
    let (means, colors, opacity) = (
        into_contiguous(xys),
        into_contiguous(shs),
        into_contiguous(opacity),
    );

    // Get the matmul relevant shapes.
    let num_splats = means.shape.dims[0];
    assert!(colors.shape.dims[0] == num_splats && opacity.shape.dims[0] == num_splats);

    let shape_out = Shape::new([resolution.0 as usize, resolution.1 as usize, 3]);

    // Create a buffer for the output tensor.
    let buffer = means
        .client
        .empty(shape_out.num_elements() * core::mem::size_of::<F>());

    // Create the output tensor primitive.
    let output = JitTensor::new(
        means.client.clone(),
        means.device.clone(),
        shape_out,
        buffer,
    );

    // Create the kernel.
    let kernel = RenderSplats2D::new(workgroup_size_x, workgroup_size_y);

    // Build info buffer with tensor information needed by the kernel, such as shapes and strides.
    // let info = build_info(&[&means, &rhs, &output]);
    // let info_handle = means.client.create(bytemuck::cast_slice(&info));

    // Declare the wgsl workgroup with the number of blocks in x, y and z.
    let blocks_needed_in_x = f32::ceil(num_splats as f32 / workgroup_size_x as f32) as u32;
    let workgroup = WorkGroup::new(blocks_needed_in_x, 1, 1);

    // Execute lazily the kernel with the launch information and the given buffers.
    means.client.execute(
        Box::new(DynamicKernel::new(kernel, workgroup)),
        &[
            &means.handle,
            &colors.handle,
            &opacity.handle,
            &output.handle,
        ],
    );

    // Return the output tensor.
    output
}

// Implement our custom backend trait for any backend that also implements our custom backend trait.
//
// Note that we could implement the backend trait only for the Wgpu backend instead of any backend that
// also implements our own API. This would allow us to call any function only implemented for Wgpu
// and potentially call a custom kernel crafted only for this task.
// impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {

//     fn project_gaussians() {
//         todo!()
//     }
// }

// Create our zero-sized type that will implement the Backward trait.
#[derive(Debug)]
struct RenderSplatsBackward;

// Implement the backward trait for the given backend B, the node gradient being of rank D
// with three other gradients to calculate (means, colors, and opacity).
impl<B: Backend> Backward<B, 3, 3> for RenderSplatsBackward {
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

fn render_splats_2d_backward<B: Backend, C: CheckpointStrategy>(
    resolution: (u32, u32),
    means: FloatTensor<B, 2>,
    shs: FloatTensor<B, 3>,
    opacity: FloatTensor<B, 1>,
) -> FloatTensor<B, 3> {
    // Prepare a stateful operation with each variable node and corresponding graph.
    //
    // Each node can be fetched with `ops.parents` in the same order as defined here.
    match RenderSplatsBackward
        .prepare::<C>(
            [means.node.clone(), shs.node.clone(), opacity.node.clone()],
            [
                means.graph.clone(),
                shs.graph.clone(),
                opacity.graph.clone(),
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
            let colors_state = prep.checkpoint(&shs);
            let opacity_state = prep.checkpoint(&opacity);

            let output = B::render_splats_2d(
                resolution,
                means.primitive.clone(),
                shs.primitive.clone(),
                opacity.primitive,
            );

            let state = (means_state, colors_state, opacity_state);
            prep.finish(state, output)
        }
        OpsKind::UnTracked(prep) => {
            // When no node is tracked, we can just compute the original operation without
            // keeping any state.
            let output = B::render_splats_2d(
                resolution,
                means.primitive,
                shs.primitive,
                opacity.primitive,
            );
            prep.finish(output)
        }
    }
}
