use burn::tensor::{backend::Backend, module::conv2d, ops::ConvOptions, Tensor};

fn gaussian<B: Backend>(window_size: usize, sigma: f32, device: &B::Device) -> Tensor<B, 1> {
    let window_extent = (window_size / 2) as f32;
    let vals: Vec<_> = (0..window_size)
        .map(|x| f32::exp(-(x as f32 - window_extent).powf(2.0) / (2.0 * sigma.powf(2.0))))
        .collect();
    let gauss = Tensor::from_floats(vals.as_slice(), device);
    gauss.clone() / gauss.sum()
}

fn create_window<B: Backend>(
    window_size: usize,
    channel: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let window1d = gaussian(window_size, 1.5, device).reshape([window_size, 1]);
    let window2d = window1d.clone().matmul(window1d.transpose());
    window2d.unsqueeze().repeat_dim(0, channel)
}

pub fn ssim<B: Backend>(
    img1: Tensor<B, 4>,
    img2: Tensor<B, 4>,
    window_size: usize,
) -> Tensor<B, 1> {
    let device = &img1.device();

    let [b, _, _, channel] = img1.dims();
    let window = create_window::<B>(window_size, channel, device);

    let padding = window_size / 2;
    let conv_options = ConvOptions::new([1, 1], [padding, padding], [0, 0], channel);
    let mu1 = conv2d(img1.clone(), window.clone(), None, conv_options.clone());
    let mu2 = conv2d(img2.clone(), window.clone(), None, conv_options.clone());
    let mu1_sq = mu1.clone().powf_scalar(2);
    let mu2_sq = mu2.clone().powf_scalar(2);
    let mu1_mu2 = mu1 * mu2;

    let sigma1_sq = conv2d(
        img1.clone().powf_scalar(2.0),
        window.clone(),
        None,
        conv_options.clone(),
    ) - mu1_sq.clone();

    let sigma2_sq = conv2d(
        img2.clone().powf_scalar(2.0),
        window.clone(),
        None,
        conv_options.clone(),
    ) - mu2_sq.clone();

    let sigma12 = conv2d(
        img1.clone() * img2.clone(),
        window.clone(),
        None,
        conv_options.clone(),
    ) - mu1_mu2.clone();

    let c1: f32 = 0.01f32.powf(2.0);
    let c2: f32 = 0.03f32.powf(2.0);

    let ssim_map = ((mu1_mu2 * 2.0 + c1) * (sigma12 * 2.0 + c2))
        / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2));
    ssim_map.mean_dim(1).mean_dim(2).mean_dim(3).reshape([b])
}
