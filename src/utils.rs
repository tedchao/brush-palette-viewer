// Utilities to go from ndarray -> image and the other way around.
#[allow(dead_code)]
use burn::tensor::{backend::Backend, Tensor};

pub fn to_rerun_tensor<B: Backend, const D: usize>(t: Tensor<B, D>) -> rerun::TensorData {
    rerun::TensorData::new(
        t.dims()
            .map(|x| rerun::TensorDimension::unnamed(x as u64))
            .to_vec(),
        rerun::TensorBuffer::F32(t.into_data().convert().value.into()),
    )
}

// Assume 0-1, unlike rerun which always normalizes the image.
pub fn to_rerun_image<B: Backend>(t: Tensor<B, 3>) -> rerun::Image {
    let t_quant = (t * 255.0).int().clamp(0, 255);

    rerun::Image::new(rerun::TensorData::new(
        t_quant
            .dims()
            .map(|x| rerun::TensorDimension::unnamed(x as u64))
            .to_vec(),
        rerun::TensorBuffer::I8(t_quant.into_data().convert().value.into()),
    ))
}
