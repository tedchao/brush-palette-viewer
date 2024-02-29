use std::error::Error;
mod camera;
mod utils;

use burn::{
    backend::{
        wgpu::{compute::WgpuRuntime, AutoGraphicsApi},
        Autodiff,
    },
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    tensor::{
        activation::{relu, sigmoid},
        backend::{AutodiffBackend, Backend},
        Data, Tensor,
    },
};

use burn::tensor::ElementConversion;

use image::io::Reader as ImageReader;

#[derive(Module, Debug)]
struct SmallNerf<B: Backend> {
    project_up: Linear<B>,
    lins: Vec<Linear<B>>,
    project_down: Linear<B>,
}

#[derive(Config, Debug)]
pub struct SmallNerfConfig {
    hidden_width: usize,
}

impl SmallNerfConfig {
    /// Returns the initialized model.
    fn init<B: Backend>(&self, device: &B::Device) -> SmallNerf<B> {
        SmallNerf {
            project_up: LinearConfig::new(16, self.hidden_width).init(device),
            lins: std::iter::repeat_with(|| {
                LinearConfig::new(self.hidden_width, self.hidden_width).init(device)
            })
            .take(4)
            .collect(),
            project_down: LinearConfig::new(self.hidden_width, 3).init(device),
        }
    }
}

impl<B: Backend> SmallNerf<B> {
    fn forward(&self, coords: Tensor<B, 3>) -> Tensor<B, 3> {
        // coords: [H, W, c]
        let x = coords * std::f32::consts::PI;

        let x = Tensor::cat(
            vec![
                (x.clone()).sin(),
                (x.clone()).cos(),
                (x.clone() * 8).sin(),
                (x.clone() * 8).cos(),
                (x.clone() * 32).sin(),
                (x.clone() * 32).cos(),
                (x.clone() * 64).sin(),
                (x.clone() * 64).cos(),
            ],
            2,
        );

        let mut x = x;
        x = relu(self.project_up.forward(x));
        for lin in &self.lins {
            x = relu(lin.forward(x));
        }
        sigmoid(self.project_down.forward(x))
    }
}

#[derive(Config)]
pub struct ExpConfig {
    #[config(default = 4e-3)]
    pub lr: f64,

    #[config(default = 42)]
    pub seed: u64,
}

fn meshgrid2<B: Backend>(h: usize, w: usize, device: &B::Device) -> Tensor<B, 3> {
    let coords_h = Tensor::arange(0..h as i64, device).float() / (h as f64) - 0.5;
    let coords_w = Tensor::arange(0..w as i64, device).float() / (w as f64) - 0.5;

    let coords_h = Tensor::repeat(coords_h.reshape([h as i32, 1]), 1, w);
    let coords_w = Tensor::repeat(coords_w.reshape([1, w as i32]), 0, h);

    Tensor::cat(
        vec![coords_h.reshape([h, w, 1]), coords_w.reshape([h, w, 1])],
        2,
    )
}

fn run<B: AutodiffBackend>(device: &B::Device) -> Result<(), Box<dyn Error>> {
    let camera = camera::Camera::new();

    let target = ImageReader::open("./crab.webp")?.decode()?;
    let target = target.resize(300, 200, image::imageops::FilterType::Lanczos3);
    let target = Tensor::<B, 1>::from_data(Data::from(target.as_bytes()).convert(), device)
        .reshape([target.height() as usize, target.width() as usize, 3])
        / 255.0;

    let rec = rerun::RecordingStreamBuilder::new("visualize training").spawn()?;

    // Config
    let mut optimizer = AdamWConfig::new().init();
    let mut model: SmallNerf<B> = SmallNerfConfig::new(128).init(device);

    let config = ExpConfig::new();

    B::seed(config.seed);

    let [h, w, _] = target.dims();

    rec.log(
        "Target.",
        &rerun::Image::new(utils::to_rerun_tensor(target.clone())),
    )?;

    let coords = meshgrid2(h, w, device);

    for _i in 0.. {
        let output = model.forward(coords.clone());

        let loss = (output.clone() - target.clone()).powf_scalar(2.0).mean();

        if _i % 25 == 0 {
            rec.log(
                "Cur img.",
                &rerun::Image::new(utils::to_rerun_tensor(output.clone())),
            )?;
            rec.log(
                "Error.",
                &rerun::Image::new(utils::to_rerun_tensor(
                    (output.clone() - target.clone()).abs(),
                )),
            )?;
        }
        rec.log(
            "loss.",
            &rerun::Scalar::new(loss.clone().into_scalar().elem::<f64>()),
        )?;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(config.lr, model, grads);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = Default::default();
    type BackGPU = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;
    run::<Autodiff<BackGPU>>(&device)
}
