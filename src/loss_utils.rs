use burn::tensor::{backend::Backend, Tensor};

// Computes the L1 loss of the two tensors.
pub(crate) fn l1_loss<B: Backend, const D: usize>(
    prediction: Tensor<B, D>,
    gt: Tensor<B, D>,
) -> Tensor<B, 1> {
    (prediction - gt).abs().mean()
}

// Computes the L2 loss of the two tensors.
pub(crate) fn mse<B: Backend, const D: usize>(
    prediction: Tensor<B, D>,
    gt: Tensor<B, D>,
) -> Tensor<B, 1> {
    (prediction - gt).powf_scalar(2.0).mean()
}

// Computes the PSNR of the two images
fn psnr<B: Backend, const D: usize>(img1: Tensor<B, D>, img2: Tensor<B, D>) -> Tensor<B, 1> {
    -(Tensor::log(mse(img1, img2).sqrt()) / 10.0_f64.ln()) * 20.0
}

// Computes an 1D Gaussian a specific window
// fn gaussian<B: Backend>(window_size: i32, sigma: f32) -> Tensor<B, X> {
//   let gauss = Tensor<B, X>([
//       math.exp(-((x - window_size / 2) ** 2) / float(2 * sigma**2));
//       for x in range(window_size)
//   ]);
//   return gauss / gauss.sum();
// }

// // Creates a 2D Gaussian window kernel
// // fn create_window<B: Backend>(window_size: int, channel: int) -> Tensor<B, X> {
// //     let window_1d = gaussian(window_size, 1.5).unsqueeze(1)
// //     let window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
// //     let window = torch.autograd.Variable(
// //         window_2d.expand(channel, 1, window_size, window_size).contiguous()
// //     )
// //   return window;
// // }

// Computes the ssim of the two images
// fn ssim<B: Backend>(
//     img1: Tensor<B, 3>,
//     img2: Tensor<B, 3>,
//     window_size: i32,
//     size_average: bool,
// ) -> Tensor<B, 1> {
//   let channel = img1.dims()[2];
//   let window = create_window(window_size, channel);

//   mu1 = F.conv2d(img1, window, padding=window_size / 2, groups=channel)
//   mu2 = F.conv2d(img2, window, padding=window_size / 2, groups=channel)

//   mu1_sq = mu1.pow(2);
//   mu2_sq = mu2.pow(2);
//   mu1_mu2 = mu1 * mu2;

//   let sigma1_sq = (
//       F.conv2d(img1 * img1, window, padding=window_size / 2, groups=channel)
//       - mu1_sq
//   );
//   let sigma2_sq = (
//       F.conv2d(img2 * img2, window, padding=window_size / 2, groups=channel)
//       - mu2_sq
//   );
//   let sigma12 = (
//       F.conv2d(img1 * img2, window, padding=window_size / 2, groups=channel)
//       - mu1_mu2
//   );

//   let C1 = 0.01.pow(2.0);
//   let C2 = 0.03.pow(3.0);

//   let ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2);

//   if size_average {
//     ssim_map.mean()
//   } else {
//     ssim_map.mean(1).mean(1).mean(1)
//   }
// }
