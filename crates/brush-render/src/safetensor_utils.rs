// TODO: This is rather gnarly, must be an easier way to do this.
use burn::{
    prelude::Backend,
    tensor::{Float, Tensor, TensorData},
};
use safetensors::tensor::TensorView;

// Nb: this only handles float tensors, good enough :)
// TODO: Doesn't TensorData have something for this?
fn float_from_u8(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

pub(crate) fn safe_tensor_to_burn1<B: Backend>(
    t: TensorView,
    device: &B::Device,
) -> Tensor<B, 1, Float> {
    let data = TensorData::new::<f32, _>(float_from_u8(t.data()), [t.shape()[0]]);
    Tensor::from_data(data, device)
}

pub(crate) fn safe_tensor_to_burn2<B: Backend>(
    t: TensorView,
    device: &B::Device,
) -> Tensor<B, 2, Float> {
    let data = TensorData::new::<f32, _>(float_from_u8(t.data()), [t.shape()[0], t.shape()[1]]);
    Tensor::from_data(data, device)
}

pub(crate) fn safe_tensor_to_burn3<B: Backend>(
    t: TensorView,
    device: &B::Device,
) -> Tensor<B, 3, Float> {
    let data = TensorData::new::<f32, _>(
        float_from_u8(t.data()),
        [t.shape()[0], t.shape()[1], t.shape()[2]],
    );
    Tensor::from_data(data, device)
}
