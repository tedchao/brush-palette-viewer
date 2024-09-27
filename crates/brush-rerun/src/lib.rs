use burn::{
    prelude::*,
    tensor::{BasicOps, Tensor},
};
use rerun::{ChannelDatatype, ColorModel};

trait BurnToRerunData {
    fn into_rerun_data(self) -> rerun::TensorData;
}

fn tensor_dims<B: Backend, const D: usize, K: BasicOps<B>>(
    tensor: &Tensor<B, D, K>,
) -> Vec<rerun::TensorDimension> {
    tensor
        .dims()
        .map(|x| rerun::TensorDimension::unnamed(x as u64))
        .to_vec()
}

impl<B: Backend, const D: usize> BurnToRerunData for Tensor<B, D> {
    fn into_rerun_data(self) -> rerun::TensorData {
        rerun::TensorData::new(
            tensor_dims(&self),
            rerun::TensorBuffer::F32(self.into_data().to_vec::<f32>().unwrap().into()),
        )
    }
}

impl<B: Backend, const D: usize> BurnToRerunData for Tensor<B, D, Int> {
    fn into_rerun_data(self) -> rerun::TensorData {
        rerun::TensorData::new(
            tensor_dims(&self),
            rerun::TensorBuffer::I32(self.into_data().to_vec::<i32>().unwrap().into()),
        )
    }
}

impl<B: Backend, const D: usize> BurnToRerunData for Tensor<B, D, Bool> {
    fn into_rerun_data(self) -> rerun::TensorData {
        rerun::TensorData::new(
            tensor_dims(&self),
            rerun::TensorBuffer::U8(self.into_data().convert::<u8>().to_vec().unwrap().into()),
        )
    }
}

pub trait BurnToRerun {
    fn into_rerun(self) -> rerun::Tensor;
}

impl<T: BurnToRerunData> BurnToRerun for T {
    fn into_rerun(self) -> rerun::Tensor {
        rerun::Tensor::new(self.into_rerun_data())
    }
}

pub trait BurnToImage {
    fn into_rerun_image(self) -> rerun::Image;
}

impl<B: Backend> BurnToImage for Tensor<B, 3> {
    fn into_rerun_image(self) -> rerun::Image {
        let [h, w, c] = self.dims();
        let color_model = if c == 3 {
            ColorModel::RGB
        } else {
            ColorModel::RGBA
        };
        rerun::Image::from_color_model_and_bytes(
            self.into_data().bytes,
            [w as u32, h as u32],
            color_model,
            ChannelDatatype::F32,
        )
    }
}

// Assume int images encode u8 data.
impl<B: Backend> BurnToImage for Tensor<B, 3, Int> {
    fn into_rerun_image(self) -> rerun::Image {
        let [h, w, c] = self.dims();
        let color_model = if c == 3 {
            ColorModel::RGB
        } else {
            ColorModel::RGBA
        };
        rerun::Image::from_color_model_and_bytes(
            self.into_data().bytes,
            [w as u32, h as u32],
            color_model,
            ChannelDatatype::U8,
        )
    }
}
