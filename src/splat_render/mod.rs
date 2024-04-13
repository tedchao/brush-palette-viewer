use burn::prelude::Int;
use burn::{
    backend::{
        wgpu::{AutoGraphicsApi, JitBackend, WgpuRuntime},
        Autodiff,
    },
    tensor::{Shape, Tensor},
};
use burn_compute::{
    channel::ComputeChannel,
    client::ComputeClient,
    server::{ComputeServer, Handle},
};
use burn_jit::{JitElement, Runtime};
use burn_wgpu::JitTensor;

use crate::camera::Camera;

mod dim_check;
mod generated_bindings;
mod kernels;
mod prefix_sum;
mod radix_sort;

pub mod render;

type BurnRuntime = WgpuRuntime<AutoGraphicsApi, f32, i32>;
pub(crate) type BurnBack = JitBackend<BurnRuntime>;
type BurnClient =
    ComputeClient<<BurnRuntime as Runtime>::Server, <BurnRuntime as Runtime>::Channel>;

type BufferHandle = Handle<<BurnRuntime as Runtime>::Server>;

pub type FloatTensor<B, const D: usize> =
    <B as burn::tensor::backend::Backend>::FloatTensorPrimitive<D>;

pub type IntTensor<B, const D: usize> =
    <B as burn::tensor::backend::Backend>::IntTensorPrimitive<D>;

pub type HandleType<S> = Handle<S>;

#[derive(Debug, Clone)]
pub(crate) struct Aux<B: Backend> {
    pub tile_bins: Tensor<B, 3, Int>,
    pub radii: Tensor<B, 1>,
    pub gaussian_ids_sorted: Tensor<B, 1, Int>,
    pub xys: Tensor<B, 2>,
    pub cov2ds: Tensor<B, 2>,
    pub final_index: Tensor<B, 2, Int>,
    pub num_intersects: u32,
}

#[derive(Default)]
pub(crate) struct RenderArgs {
    pub sync_kernels: bool,
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
        args: RenderArgs,
    ) -> (FloatTensor<Self, 3>, Aux<BurnBack>);
}

// TODO: In rust 1.80 having a trait bound here on the inner backend would be great.
// For now all code using it will need to specify this bound itself.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}
impl AutodiffBackend for Autodiff<BurnBack> {}

fn create_buffer<E: JitElement, const D: usize>(
    client: &BurnClient,
    shape: [usize; D],
) -> BufferHandle {
    let shape = Shape::new(shape);
    let bufsize = shape.num_elements() * core::mem::size_of::<E>();
    client.empty(bufsize)
}

fn create_tensor<E: JitElement, const D: usize>(
    client: &BurnClient,
    device: &<BurnRuntime as Runtime>::Device,
    shape: [usize; D],
) -> JitTensor<BurnRuntime, E, D> {
    JitTensor::new(
        client.clone(),
        device.clone(),
        Shape::new(shape),
        create_buffer::<E, D>(client, shape),
    )
}

pub(crate) fn read_buffer_to_u32<S: ComputeServer, C: ComputeChannel<S>>(
    client: &ComputeClient<S, C>,
    tensor: &Handle<S>,
) -> Vec<u32> {
    let data = client.read(tensor).read();
    data.into_iter()
        .array_chunks::<4>()
        .map(u32::from_le_bytes)
        .collect()
}

fn read_buffer_to_f32<S: ComputeServer, C: ComputeChannel<S>>(
    client: &ComputeClient<S, C>,
    tensor: &Handle<S>,
) -> Vec<f32> {
    let data = client.read(tensor).read();
    data.into_iter()
        .array_chunks::<4>()
        .map(f32::from_le_bytes)
        .collect()
}

fn assert_buffer_is_finite<S: ComputeServer, C: ComputeChannel<S>>(
    client: &ComputeClient<S, C>,
    tensor: &Handle<S>,
) {
    for (i, elem) in read_buffer_to_f32(client, tensor).iter().enumerate() {
        if !elem.is_finite() {
            panic!("Elem {elem} at {i} is invalid!");
        }
    }
}

fn div_round_up(x: usize, y: usize) -> usize {
    (x + y - 1) / y
}
