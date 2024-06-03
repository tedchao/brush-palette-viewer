// Utilities to go from ndarray -> image and the other way around.
use burn::tensor::{backend::Backend, Float, Shape, Tensor, TensorKind};
use ndarray::{Array, ArrayView, Dim, Dimension, StrideShape};

pub(crate) fn ndarray_to_burn<B: Backend, const D: usize>(
    arr: ArrayView<f32, Dim<[usize; D]>>,
    device: &B::Device,
) -> Tensor<B, D, Float>
where
    Dim<[usize; D]>: Dimension,
{
    let shape = Shape::new(arr.shape().try_into().unwrap());
    Tensor::from_floats(arr.as_slice().unwrap(), device).reshape(shape)
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

pub(crate) fn burn_to_scalar<B, K>(tensor: Tensor<B, 1, K>) -> K::Elem
where
    K: burn::tensor::BasicOps<B>,
    K: TensorKind<B>,
    B: Backend,
{
    assert_eq!(tensor.dims()[0], 1);
    tensor.into_data().value[0]
}

// Quaternion multiplication function
pub(crate) fn quat_multiply<B: Backend>(q: Tensor<B, 2>, r: Tensor<B, 2>) -> Tensor<B, 2> {
    let num = q.dims()[0];

    let (q0, q1, q2, q3) = (
        q.clone().slice([0..num, 3..4]),
        q.clone().slice([0..num, 0..1]),
        q.clone().slice([0..num, 1..2]),
        q.clone().slice([0..num, 2..3]),
    );
    let (r0, r1, r2, r3) = (
        r.clone().slice([0..num, 0..1]),
        r.clone().slice([0..num, 1..2]),
        r.clone().slice([0..num, 2..3]),
        r.clone().slice([0..num, 3..4]),
    );

    Tensor::cat(
        vec![
            r0.clone() * q0.clone()
                - r1.clone() * q1.clone()
                - r2.clone() * q2.clone()
                - r3.clone() * q3.clone(),
            r0.clone() * q1.clone() + r1.clone() * q0.clone() - r2.clone() * q3.clone()
                + r3.clone() * q2.clone(),
            r0.clone() * q2.clone() + r1.clone() * q3.clone() + r2.clone() * q0.clone()
                - r3.clone() * q1.clone(),
            r0.clone() * q3.clone() - r1.clone() * q2.clone()
                + r2.clone() * q1.clone()
                + r3.clone() * q0.clone(),
        ],
        1,
    )
}

pub(crate) fn quaternion_rotation<B: Backend>(
    vectors: Tensor<B, 2>,
    quaternions: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let num = vectors.dims()[0];
    // Convert vectors to quaternions with zero real part
    let vector_quats = Tensor::cat(
        vec![
            Tensor::zeros_like(&vectors.clone().slice([0..num, 0..1])),
            vectors,
        ],
        1,
    );

    // Calculate the conjugate of quaternions
    let quaternions_conj = quaternions.clone().slice_assign(
        [0..num, 1..4],
        quaternions.clone().slice([0..num, 1..4]) * -1,
    );

    // Rotate vectors: v' = q * v * q_conjugate
    let rotated_vectors = quat_multiply(quat_multiply(quaternions, vector_quats), quaternions_conj);

    // Return only the vector part (imaginary components)
    rotated_vectors.slice([0..num, 1..4])
}
