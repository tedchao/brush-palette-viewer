#![allow(clippy::single_range_in_vec_init)]
use crate::camera::Camera;
use burn::backend::Autodiff;
use burn::prelude::Int;
use burn::{
    backend::wgpu::{AutoGraphicsApi, JitBackend, WgpuRuntime},
    tensor::{Shape, Tensor},
};
use burn_compute::server::Binding;
use burn_compute::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};
use burn_cube::Runtime;
use burn_jit::JitElement;
use burn_wgpu::{JitTensor, WgpuDevice};

mod dim_check;
mod kernels;
mod prefix_sum;
mod radix_sort;
pub mod render;
mod shaders;

pub type BurnBack = JitBackend<BurnRuntime, f32, i32>;

pub type BurnRuntime = WgpuRuntime<AutoGraphicsApi>;
type BurnClient =
    ComputeClient<<BurnRuntime as Runtime>::Server, <BurnRuntime as Runtime>::Channel>;

type FloatTensor<B, const D: usize> =
    <B as burn::tensor::backend::Backend>::FloatTensorPrimitive<D>;
type IntTensor<B, const D: usize> = <B as burn::tensor::backend::Backend>::IntTensorPrimitive<D>;

#[derive(Debug, Clone)]
pub(crate) struct Aux<B: Backend> {
    pub num_visible: Tensor<B, 1, Int>,
    pub num_intersects: Tensor<B, 1, Int>,
    pub tile_bins: Tensor<B, 3, Int>,
    pub radii: Tensor<B, 1>,
    pub depthsort_gid_from_isect: Tensor<B, 1, Int>,
    pub compact_from_depthsort_gid: Tensor<B, 1, Int>,
    pub depths: Tensor<B, 1>,
    pub xys: Tensor<B, 2>,
    pub cum_tiles_hit: Tensor<B, 1, Int>,
    pub conic_comps: Tensor<B, 2>,
    pub colors: Tensor<B, 2>,
    pub final_index: Option<Tensor<B, 2, Int>>,
    pub global_from_compact_gid: Tensor<B, 1, Int>,
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
        log_scales: FloatTensor<Self, 2>,
        quats: FloatTensor<Self, 2>,
        colors: FloatTensor<Self, 2>,
        raw_opacity: FloatTensor<Self, 1>,
        background: glam::Vec3,
    ) -> (FloatTensor<Self, 3>, Aux<Self>);
}

// TODO: In rust 1.80 having a trait bound here on the inner backend would be great.
// For now all code using it will need to specify this bound itself.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}
impl AutodiffBackend for Autodiff<BurnBack> {}

fn create_tensor<E: JitElement, const D: usize>(
    client: &BurnClient,
    device: &WgpuDevice,
    shape: [usize; D],
) -> JitTensor<BurnRuntime, E, D> {
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
