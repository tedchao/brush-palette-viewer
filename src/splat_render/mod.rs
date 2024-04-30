use crate::camera::Camera;
use burn::prelude::Int;
use burn::{
    backend::wgpu::{AutoGraphicsApi, JitBackend, WgpuRuntime},
    tensor::{Shape, Tensor},
};
use burn_compute::server::Binding;
use burn_compute::{
    channel::ComputeChannel,
    client::ComputeClient,
    server::{ComputeServer, Handle},
};
use burn_jit::{JitElement, Runtime};
use burn_wgpu::JitTensor;
use tracing::info_span;

mod dim_check;
mod kernels;
mod prefix_sum;
mod radix_sort;
pub mod render;
mod shaders;

pub type BurnRuntime = WgpuRuntime<AutoGraphicsApi>;
pub type BurnBack = JitBackend<BurnRuntime, f32, i32>;

type BurnClient =
    ComputeClient<<BurnRuntime as Runtime>::Server, <BurnRuntime as Runtime>::Channel>;
type BufferHandle = Handle<<BurnRuntime as Runtime>::Server>;

type FloatTensor<B, const D: usize> =
    <B as burn::tensor::backend::Backend>::FloatTensorPrimitive<D>;
type IntTensor<B, const D: usize> = <B as burn::tensor::backend::Backend>::IntTensorPrimitive<D>;

#[derive(Debug, Clone)]
pub(crate) struct Aux<B: Backend> {
    pub tile_bins: Tensor<B, 3, Int>,
    pub radii: Tensor<B, 1>,
    pub gaussian_ids_sorted: Tensor<B, 1, Int>,
    pub xys: Tensor<B, 2>,
    pub cov2ds: Tensor<B, 2>,
    pub final_index: Option<Tensor<B, 2, Int>>,
    pub num_intersects: u32,
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
        img_size: glam::UVec2,
        means: FloatTensor<Self, 2>,
        scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
        colors: FloatTensor<Self, 2>,
        opacity: FloatTensor<Self, 1>,
        background: glam::Vec3,
    ) -> (FloatTensor<Self, 3>, Aux<BurnBack>);
}

pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}

fn create_tensor<E: JitElement, const D: usize>(
    client: &BurnClient,
    device: &<BurnRuntime as Runtime>::Device,
    shape: [usize; D],
) -> JitTensor<BurnRuntime, E, D> {
    let _span = info_span!("Create tensor").entered();

    let shape = Shape::new(shape);
    let bufsize = shape.num_elements() * core::mem::size_of::<E>();
    let buffer = client.empty(bufsize);
    JitTensor::new(client.clone(), device.clone(), shape, buffer)
}

fn bitcast_tensor<const D: usize, EIn: JitElement, EOut: JitElement>(
    tensor: JitTensor<BurnRuntime, EIn, D>,
) -> JitTensor<BurnRuntime, EOut, D> {
    JitTensor::new(tensor.client, tensor.device, tensor.shape, tensor.handle)
}

pub(crate) fn read_buffer_to_u32<S: ComputeServer, C: ComputeChannel<S>>(
    client: &ComputeClient<S, C>,
    binding: Binding<S>,
) -> Vec<u32> {
    let data = client.read(binding).read();
    data.chunks_exact(4)
        .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

fn read_buffer_to_f32<S: ComputeServer, C: ComputeChannel<S>>(
    client: &ComputeClient<S, C>,
    binding: Binding<S>,
) -> Vec<f32> {
    let data = client.read(binding).read();
    data.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

fn assert_buffer_is_finite<S: ComputeServer, C: ComputeChannel<S>>(
    client: &ComputeClient<S, C>,
    binding: Binding<S>,
) {
    for (i, elem) in read_buffer_to_f32(client, binding).iter().enumerate() {
        if !elem.is_finite() {
            panic!("Elem {elem} at {i} is invalid!");
        }
    }
}
