use burn::{
    backend::{
        wgpu::{AutoGraphicsApi, JitBackend, WgpuRuntime},
        Autodiff,
    },
    tensor::Shape,
};
use burn_compute::{
    channel::ComputeChannel,
    client::ComputeClient,
    server::{ComputeServer, Handle},
};
use burn_jit::{compute::Kernel, JitElement, Runtime};
use burn_wgpu::JitTensor;

use crate::camera::Camera;

mod gen;
mod kernels;
pub mod render;

type BurnRuntime = WgpuRuntime<AutoGraphicsApi, f32, i32>;
type BurnBack = JitBackend<BurnRuntime>;
type BurnClient =
    ComputeClient<<BurnRuntime as Runtime>::Server, <BurnRuntime as Runtime>::Channel>;

type BufferHandle = Handle<<BurnRuntime as Runtime>::Server>;

pub type FloatTensor<B, const D: usize> =
    <B as burn::tensor::backend::Backend>::FloatTensorPrimitive<D>;

pub type IntTensor<B, const D: usize> =
    <B as burn::tensor::backend::Backend>::IntTensorPrimitive<D>;

pub type HandleType<S> = Handle<S>;

struct ProjectionOutput<B: Backend> {
    covs3d: FloatTensor<B, 2>,
    xys: FloatTensor<B, 2>,
    depths: FloatTensor<B, 2>,
    radii: FloatTensor<B, 2>,
    conics: FloatTensor<B, 2>,
    compensation: FloatTensor<B, 2>,
    num_tiles_hit: FloatTensor<B, 2>,
}

/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend {
    // Render splats
    // Project splats processing step. This produces
    // a whole bunch of gradients that we store.
    // The return just happens to be the xy screenspace points
    // which we use to 'carry' the gradients'.
    fn render_gaussians(
        cam: &Camera,
        means: FloatTensor<Self, 2>,
        scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
        colors: FloatTensor<Self, 2>,
        opacity: FloatTensor<Self, 1>,
        background: glam::Vec3,
    ) -> FloatTensor<Self, 3>;
}

// TODO: In rust 1.80 having a trait bound here on the inner backend would be great.
// For now all code using it will need to specify this bound itself.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}
impl AutodiffBackend for Autodiff<BurnBack> {}

enum BufferAlloc {
    Empty,
    Zeros,
}

fn create_buffer<E: JitElement, const D: usize>(
    client: &BurnClient,
    shape: [usize; D],
    alloc: BufferAlloc,
) -> BufferHandle {
    let shape = Shape::new(shape);
    let bufsize = shape.num_elements() * core::mem::size_of::<E>();
    match alloc {
        BufferAlloc::Empty => client.empty(bufsize),
        BufferAlloc::Zeros => {
            // TODO: Does burn not have a fast path for zero allocating a buffer?
            // Or are buffers zero allocated in the first place?
            let zeros = vec![0; bufsize];
            client.create(&zeros)
        }
    }
}

fn create_tensor<E: JitElement, const D: usize>(
    client: &BurnClient,
    device: &<BurnRuntime as Runtime>::Device,
    shape: [usize; D],
    alloc: BufferAlloc,
) -> JitTensor<BurnRuntime, E, D> {
    JitTensor::new(
        client.clone(),
        device.clone(),
        Shape::new(shape),
        create_buffer::<E, D>(client, shape, alloc),
    )
}

fn read_buffer_to_u32<S: ComputeServer<Kernel = Box<dyn Kernel>>, C: ComputeChannel<S>>(
    client: &ComputeClient<S, C>,
    tensor: &Handle<S>,
) -> Vec<u32> {
    let data = client.read(tensor).read();
    data.into_iter()
        .array_chunks::<4>()
        .map(u32::from_le_bytes)
        .collect()
}

fn read_buffer_to_f32<S: ComputeServer<Kernel = Box<dyn Kernel>>, C: ComputeChannel<S>>(
    client: &ComputeClient<S, C>,
    tensor: &Handle<S>,
) -> Vec<f32> {
    let data = client.read(tensor).read();
    data.into_iter()
        .array_chunks::<4>()
        .map(f32::from_le_bytes)
        .collect()
}
