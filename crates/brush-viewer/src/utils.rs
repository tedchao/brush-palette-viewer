// Utilities to go from ndarray -> image and the other way around.
use burn::tensor::{backend::Backend, Float, Shape, Tensor};
use ndarray::{ArrayView, Dim, Dimension};

pub(crate) fn ndarray_to_burn<B: Backend, const D: usize>(
    arr: ArrayView<f32, Dim<[usize; D]>>,
    device: &B::Device,
) -> Tensor<B, D, Float>
where
    Dim<[usize; D]>: Dimension,
{
    let shape = Shape::new(arr.shape().try_into().unwrap());
    Tensor::<_, 1>::from_floats(arr.as_slice().unwrap(), device).reshape(shape)
}
