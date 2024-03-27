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

impl DynamicKernelSource for ProjectSplats {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::project_forward::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl ProjectSplats  {
    fn workgroup_size() -> UVec3 {
        uvec3(16, 1, 1)
    }
}

#[derive(new, Debug)]
struct MapGaussiansToIntersect {
}

impl DynamicKernelSource for MapGaussiansToIntersect {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::map_gaussian_to_intersects::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl MapGaussiansToIntersect  {
    fn workgroup_size() -> UVec3 {
        uvec3(16, 1, 1)
    }
}


#[derive(new, Debug)]
struct GetTileBinEdges {
}

impl GetTileBinEdges  {
    fn workgroup_size() -> UVec3 {
        uvec3(16, 1, 1)
    }
}

impl DynamicKernelSource for GetTileBinEdges {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::get_tile_bin_edges::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

#[derive(new, Debug)]
struct RasterizeForward {
}

impl DynamicKernelSource for RasterizeForward {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::rasterize::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl RasterizeForward  {
    fn workgroup_size() -> UVec3 {
        uvec3(16, 16, 1)
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
        opacity: FloatTensor<Self, 1>,
        background: glam::Vec3,
    ) -> FloatTensor<Self, 3> {
        println!("Project gaussians!");

        // Preprocess tensors.
        assert!(means.device == scales.device && means.device == quats.device && means.device == colors.device && means.device == opacity.device);
        let (means, scales, quats, colors, opacity) = (
            into_contiguous(means),
            into_contiguous(scales),
            into_contiguous(quats),
            into_contiguous(colors),
            into_contiguous(opacity),
        );

        let num_points = means.shape.dims[0];
        let batches = [means.shape.dims[0], scales.shape.dims[0], quats.shape.dims[0], colors.shape.dims[0], opacity.shape.dims[0]];
        assert!(batches.iter().all(|&x| x == num_points));

        // Divide screen into block
        let block_size = 16;
        let img_size = [camera.width, camera.height];
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

        // TODO: Must be a faster way to create with nulls.
        let create_empty_f32 = |dim|-> JitTensor<WgpuRuntime<G, F, I>, F, 2> {
            let shape = Shape::new(dim);
            JitTensor::new(
                client.clone(),
                device.clone(),
                shape.clone(),
                client.create(&vec![0; shape.num_elements() * core::mem::size_of::<F>()]),
            )
        };
        let create_empty_i32 = |dim| -> JitTensor<WgpuRuntime<G, F, I>, I, 2> {
            let shape = Shape::new(dim);
            JitTensor::new(
                client.clone(),
                device.clone(),
                shape.clone(),
                client.create(&vec![0; shape.num_elements() * core::mem::size_of::<F>()]),
            )
        };

        let to_vec_i = |tensor: &JitTensor<WgpuRuntime<G, F, I>, I, 2>| -> Vec<u32> {
            let data = client.read(&tensor.handle).read();
            data.into_iter().array_chunks::<4>().map(u32::from_le_bytes).collect()
        };

        // TODO: We might only need the data on the client for this not the tensor wrapper?
        let covs3d = create_empty_f32([num_points, 6]);
        let xys = create_empty_f32([num_points, 2]);
        let depths = create_empty_f32([num_points, 1]);
        let radii = create_empty_i32([num_points, 1]);
        let conics = create_empty_f32([num_points, 3]);
        let compensation = create_empty_f32([num_points, 1]);
        let num_tiles_hit = create_empty_i32([num_points, 1]);

        let workgroup = get_workgroup(uvec3(num_points as u32, 1, 1), ProjectSplats::workgroup_size());
        client.execute(
            Box::new(DynamicKernel::new(ProjectSplats::new(), workgroup)),
            &[
                // Input tensors.
                &means.handle,
                &scales.handle,
                &quats.handle,
                // Output tensors.
                &covs3d.handle,
                &xys.handle,
                &depths.handle,
                &radii.handle,
                &conics.handle,
                &compensation.handle,
                &num_tiles_hit.handle,
                // Aux data.
                &client.create(bytemuck::bytes_of(&gen::project_forward::Uniforms::new(
                    num_points as u32,
                    camera.transform.inverse(),
                    intrins,
                    img_size,
                    tile_bounds,
                    1.0,
                    0.001,
                    block_size,
                ))),
            ],
        );

        // TODO: CPU emulation for now. Investigate if we can do without a cumulative sum?

        // Read num tiles to CPU.
        let num_tiles_hit = to_vec_i(&num_tiles_hit);
        // Calculate cumulative sum.
        let cum_tiles_hit: Vec<u32> = num_tiles_hit
            .into_iter()
            .scan(0, |acc, x| {
                let acc_val: u32 = *acc;
                *acc = acc_val.saturating_add(x);
                Some(*acc)
            })
            .collect();
        let num_intersections = *cum_tiles_hit.last().unwrap();
        // Reupload to GPU.
        let cum_tiles_hit = client.create(bytemuck::cast_slice::<u32, u8>(&cum_tiles_hit));

        let isect_ids = create_empty_i32([num_points, 2]);
        let gaussian_ids = create_empty_i32([num_points, 1]);

        let workgroup = get_workgroup(uvec3(num_intersections, 1, 1), MapGaussiansToIntersect::workgroup_size());
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

                &client.create(bytemuck::bytes_of(&gen::map_gaussian_to_intersects::Uniforms::new(
                    num_intersections,
                    tile_bounds,
                    block_size,
                ))),
            ],
        );

        // TODO: WGSL Radix sort.
        // let isect_ids = bytemuck::cast_vec::<u32, u64>(to_vec_i(&isect_ids));
        // let gaussian_ids = to_vec_i(&gaussian_ids);
        // let sorted_indices = argsort(&isect_ids);
        // let isect_ids_sorted: Vec<_> = sorted_indices.iter().copied().map(|x| isect_ids[x]).collect();
        // let gaussian_ids_sorted: Vec<_> = sorted_indices.iter().copied().map(|x| gaussian_ids[x]).collect();

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
        let workgroup = get_workgroup(uvec3(num_intersections, 1, 1), GetTileBinEdges::workgroup_size());
        // let isect_ids_sorted = client.create(bytemuck::cast_slice::<u32, u8>(&isect_ids_sorted));
        // let gaussian_ids_sorted = client.create(bytemuck::cast_slice::<u32, u8>(&gaussian_ids_sorted));

        let tile_bins = create_empty_i32([num_points, 2]);

        client.execute(
            Box::new(DynamicKernel::new(GetTileBinEdges::new(), workgroup)),
            &[
                &isect_ids.handle,
                &tile_bins.handle,
                // Uniforms.
                &client.create(bytemuck::bytes_of(&gen::get_tile_bin_edges::Uniforms::new(
                    num_intersections,
                ))),
            ],
        );

        let out_img_shape = Shape::new([camera.height as usize, camera.width as usize, 4]);
        let out_img = JitTensor::new(
            client.clone(),
            device.clone(),
            out_img_shape.clone(),
            client.empty(out_img_shape.num_elements() * core::mem::size_of::<f32>()),
        );

        let workgroup = get_workgroup(uvec3(camera.width, camera.height, 1), RasterizeForward::workgroup_size());
        client.execute(
            Box::new(DynamicKernel::new(RasterizeForward::new(), workgroup)),
            &[
                // Input tensors.
                &gaussian_ids.handle,
                &tile_bins.handle,
                &xys.handle,
                &conics.handle,
                &colors.handle,
                &opacity.handle,

                // Output data
                &out_img.handle,

                // Aux data.
                &client.create(bytemuck::bytes_of(&gen::rasterize::Uniforms::new(
                    tile_bounds,
                    background.into(),
                    img_size
                ))),
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
impl<B: Backend> Backward<B, 3, 1> for ProjectSplatsBackward {
    // Our state that we must build during the forward pass to compute the backward pass.
    //
    // Note that we could improve the performance further by only keeping the state of
    // tensors that are tracked, improving memory management, but for simplicity, we avoid
    // that part.

    // (means)
    type State = (NodeID,);

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        // // Get the nodes of each variable.
        // let image_gradient = grads.consume::<B, 3>(&ops.node);

        // Read from the state we have setup.
        let (means_state,) = ops.state;
        let means = checkpointer.retrieve_node_output(means_state);

        // Return gradient of means, atm just as itself. So d/dx means == means? errr... ?
        let grad_means = means;

        // Get the parent nodes we're registering gradients for.
        let [mean_parent] = ops.parents;

        if let Some(node) = mean_parent {
            grads.register::<B, 2>(node, grad_means);
        }
    }
}

impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    fn render_gaussians(
        camera: &Camera,
        means: FloatTensor<Self, 2>,
        scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
        colors: FloatTensor<Self, 2>,
        opacity: FloatTensor<Self, 1>,
        background: glam::Vec3,
    ) -> FloatTensor<Self, 3> {
        let prep_nodes = ProjectSplatsBackward
            .prepare::<C>(
                [means.node.clone()],
                [
                    means.graph.clone(),
                ],
            )
            // Marks the operation as compute bound, meaning it will save its
            // state instead of recomputing itself during checkpointing
            .compute_bound()
            .stateful();

        // Prepare a stateful operation with each variable node and corresponding graph.
        //
        // Each node can be fetched with `ops.parents` in the same order as defined here.
        match prep_nodes {
            OpsKind::Tracked(mut prep) => {
                // When at least one node is tracked, we should register our backward step.

                // The state consists of what will be needed for this operation's backward pass.
                // Since we need the parents' outputs, we must checkpoint their ids to retrieve their node
                // output at the beginning of the backward. We can also save utilitary data such as the bias shape
                // If we also need this operation's output, we can either save it in the state or recompute it
                // during the backward pass. Here we choose to save it in the state because it's a compute bound operation.
                let means_state = prep.checkpoint(&means);

                let output = B::render_gaussians(
                    camera,
                    means.primitive,
                    scales.primitive,
                    quats.primitive,
                    colors.primitive,
                    opacity.primitive,
                    background
                );

                let state = (means_state,);
                prep.finish(state, output)
            }
            OpsKind::UnTracked(prep) => {
                // When no node is tracked, we can just compute the original operation without
                // keeping any state.
                let output = B::render_gaussians(
                        camera,
                        means.primitive,
                        scales.primitive,
                        quats.primitive,
                        colors.primitive,
                        opacity.primitive,
                        background
                    );
                prep.finish(output)
            }
        }
    }
}
