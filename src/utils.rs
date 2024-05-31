// Utilities to go from ndarray -> image and the other way around.
use burn::tensor::{backend::Backend, BasicOps, Data, Element, Shape, Tensor, TensorKind};
use ndarray::{Array, Dim, Dimension, StrideShape};

pub(crate) fn ndarray_to_burn<E: Element, B: Backend, const D: usize, K: BasicOps<B>>(
    arr: Array<E, Dim<[usize; D]>>,
    device: &B::Device,
) -> Tensor<B, D, K>
where
    K::Elem: Element,
    Dim<[usize; D]>: Dimension,
{
    let shape = Shape::new(arr.shape().try_into().unwrap());
    Tensor::from_data(Data::new(arr.into_raw_vec(), shape).convert(), device)
}

pub(crate) fn burn_to_ndarray<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
) -> Array<f32, Dim<[usize; D]>>
where
    Dim<[usize; D]>: Dimension,
    [usize; D]: Into<StrideShape<Dim<[usize; D]>>>,
{
    let dims = tensor.dims();
    let burn_data = tensor.into_data();
    Array::from_shape_vec(dims.into(), burn_data.convert::<f32>().value).unwrap()
}

pub(crate) fn burn_to_scalar<B: Backend, K: TensorKind<B>>(tensor: Tensor<B, 1, K>) -> K::Elem
where
    K: burn::tensor::BasicOps<B>,
{
    assert_eq!(tensor.dims()[0], 1);
    tensor.into_data().value[0]
}
