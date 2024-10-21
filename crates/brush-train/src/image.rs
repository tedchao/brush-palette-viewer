use burn::{
    prelude::Backend,
    tensor::{DType, Tensor, TensorData},
};
use image::{DynamicImage, Rgb32FImage, Rgba32FImage};

// Converts an image to a tensor. The tensor will be a floating point image with a [0, 1] image.
pub fn image_to_tensor<B: Backend>(image: &DynamicImage, device: &B::Device) -> Tensor<B, 3> {
    let (w, h) = (image.width(), image.height());
    let num_channels = image.color().channel_count();

    let data = image.to_rgb32f().into_vec();
    let tensor_data = TensorData::new(data, [h as usize, w as usize, num_channels as usize]);
    Tensor::from_data(tensor_data, device)
}

pub trait TensorDataToImage {
    fn into_image(self) -> DynamicImage;
}

pub fn tensor_into_image(data: TensorData) -> DynamicImage {
    let [h, w, c] = [data.shape[0], data.shape[1], data.shape[2]];

    let img: DynamicImage = match data.dtype {
        DType::F32 => {
            let data = data.to_vec::<f32>().unwrap();
            if c == 3 {
                Rgb32FImage::from_raw(w as u32, h as u32, data)
                    .unwrap()
                    .into()
            } else if c == 4 {
                Rgba32FImage::from_raw(w as u32, h as u32, data)
                    .unwrap()
                    .into()
            } else {
                panic!("Unsupported number of channels: {}", c);
            }
        }
        _ => panic!("unsopported dtype {:?}", data.dtype),
    };

    img
}
