use std::error::Error;
mod utils;

use burn::{
    backend::wgpu::{compute::WgpuRuntime, AutoGraphicsApi},
    tensor::{Distribution, Tensor},
};

mod backward;
mod forward;

use burn::backend::Autodiff;
use burn::tensor::activation;

/// We use a type alias for better readability.
pub type FloatTensor<B, const D: usize> =
    <B as burn::tensor::backend::Backend>::FloatTensorPrimitive<D>;

/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend {
    fn fused_matmul_add_relu<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
        bias: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D>;
}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}

/// We define our custom implementation using the added function on our custom backend.
pub fn matmul_add_relu_custom<B: Backend>(
    lhs: Tensor<B, 3>,
    rhs: Tensor<B, 3>,
    bias: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let output = B::fused_matmul_add_relu(
        lhs.into_primitive(),
        rhs.into_primitive(),
        bias.into_primitive(),
    );

    Tensor::from_primitive(output)
}

/// We define a reference implementation using basic tensor operations.
pub fn matmul_add_relu_reference<B: Backend>(
    lhs: Tensor<B, 3>,
    rhs: Tensor<B, 3>,
    bias: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let x = lhs.matmul(rhs) + bias;

    activation::relu(x)
}

fn autodiff<B: AutodiffBackend>(device: &B::Device) {
    let lhs = Tensor::<B, 3>::random([1, 32, 32], Distribution::Default, device).require_grad();
    let rhs = Tensor::random([32, 32, 32], Distribution::Default, device).require_grad();
    let bias = Tensor::random([32, 32, 32], Distribution::Default, device).require_grad();

    let reference = matmul_add_relu_reference(lhs.clone(), rhs.clone(), bias.clone());

    let mut gradients = reference.backward();

    let lhs_grad_ref = lhs.grad_remove(&mut gradients).unwrap();
    let rhs_grad_ref = rhs.grad_remove(&mut gradients).unwrap();
    let bias_grad_ref = bias.grad_remove(&mut gradients).unwrap();

    let lhs = lhs.detach();
    let rhs = rhs.detach();
    let bias = bias.detach();

    let custom = matmul_add_relu_custom(lhs.clone(), rhs.clone(), bias.clone());

    let mut gradients = custom.backward();

    let lhs_grad_custom = lhs.grad_remove(&mut gradients).unwrap();
    let rhs_grad_custom = rhs.grad_remove(&mut gradients).unwrap();
    let bias_grad_custom = bias.grad_remove(&mut gradients).unwrap();

    lhs_grad_ref
        .into_data()
        .convert::<f32>()
        .assert_approx_eq(&lhs_grad_custom.into_data().convert(), 3);

    println!("Both reference and the custom fused kernel have the same lhs gradient");

    rhs_grad_ref
        .into_data()
        .convert::<f32>()
        .assert_approx_eq(&rhs_grad_custom.into_data().convert(), 3);

    println!("Both reference and the custom fused kernel have the same rhs gradient");

    bias_grad_ref
        .into_data()
        .convert::<f32>()
        .assert_approx_eq(&bias_grad_custom.into_data().convert(), 3);

    println!("Both reference and the custom fused kernel have the same bias gradient");
}

type WgpuBackend = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;

fn main() -> Result<(), Box<dyn Error>> {
    let rec = rerun::RecordingStreamBuilder::new("visualize training").spawn()?;

    let device = Default::default();

    let lhs = Tensor::<WgpuBackend, 3>::random([1, 32, 32], Distribution::Default, &device);
    let rhs = Tensor::random([32, 32, 32], Distribution::Default, &device);
    let bias = Tensor::random([32, 32, 32], Distribution::Default, &device);

    let reference = matmul_add_relu_reference(lhs.clone(), rhs.clone(), bias.clone());
    let custom = matmul_add_relu_custom(lhs, rhs, bias);

    println!("Both reference and the custom fused kernel have the same output");

    // Copy the reference tensor to the ndarray backend.
    let reference_ndarray = Tensor::new(reference.clone());

    // let ref_data = reference.inference::<WgpuBackend>(&device);
    autodiff::<Autodiff<WgpuBackend>>(&device);

    Ok(())
}
