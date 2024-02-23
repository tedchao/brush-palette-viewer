// Utilities to go from ndarray -> image and the other way around.

use burn::tensor::{backend::Backend, Data, Device, Element, Shape, Tensor};
use ndarray::Array;

fn to_tensor<B: Backend, T: Element>(
    data: Vec<T>,
    shape: [usize; 3],
    device: &Device<B>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(Data::new(data, Shape::new(shape)).convert(), device) / 255.0
        // permute(2, 0, 1)
        .swap_dims(2, 1) // [H, C, W]
        .swap_dims(1, 0) // [C, H, W]
        / 255 // normalize between [0, 1]
}

fn to_rerurn_tensor<B, const D: usize>(tensor: Tensor<B, D>)
where
    B: Backend,
{
    rerun::Tensor::try_from()

    rerun::te
}
