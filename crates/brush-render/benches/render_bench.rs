use std::{fs::File, io::Read};

use brush_render::{
    camera::{focal_to_fov, fov_to_focal, Camera},
    gaussian_splats::Splats,
    BurnBack,
};
use burn::{backend::Autodiff, tensor::Tensor};
use burn_wgpu::WgpuDevice;
use safetensors::SafeTensors;

fn main() {
    divan::main();
}

type DiffBack = Autodiff<brush_render::BurnBack>;

const SIZE_RANGE: [u32; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

#[divan::bench(max_time = 2, args = SIZE_RANGE)]
fn benchmark(bencher: divan::Bencher, num: u32) {
    let path = format!("bench_{num}");

    bencher
        .with_inputs(|| {
            let device = WgpuDevice::BestAvailable;
            let crab_img = image::open("./test_cases/crab.png").unwrap();

            // Convert the image to RGB format
            // Get the raw buffer
            let raw_buffer = crab_img.to_rgb8().into_raw();
            let crab_tens: Tensor<DiffBack, 3> = Tensor::<_, 1>::from_floats(
                raw_buffer
                    .iter()
                    .map(|&b| b as f32 / 255.0)
                    .collect::<Vec<_>>()
                    .as_slice(),
                &device,
            )
            .reshape([crab_img.height() as usize, crab_img.width() as usize, 3]);

            // TODO: Make function to load this?
            let mut buffer = Vec::new();
            let _ = File::open(format!("./test_cases/{path}.safetensors"))
                .unwrap()
                .read_to_end(&mut buffer)
                .unwrap();
            let tensors = SafeTensors::deserialize(&buffer).unwrap();
            let splats = Splats::<DiffBack>::from_safetensors(&tensors, &device).unwrap();

            // // Wait for GPU work.
            <BurnBack as burn::prelude::Backend>::sync(
                &device,
                burn::tensor::backend::SyncType::Wait,
            );

            (splats, crab_tens)
        })
        .bench_refs(|(splats, crab_tens)| {
            let device = WgpuDevice::BestAvailable;

            let [h, w] = [
                crab_tens.shape().dims[0] as usize,
                crab_tens.shape().dims[1] as usize,
            ];

            let fov = std::f32::consts::PI * 0.5;
            let focal = fov_to_focal(fov, w as u32);
            let fov_x = focal_to_fov(focal, w as u32);
            let fov_y = focal_to_fov(focal, h as u32);
            let camera = Camera::new(
                glam::vec3(0.0, 0.0, -8.0),
                glam::Quat::IDENTITY,
                glam::vec2(fov_x, fov_y),
                glam::vec2(0.5, 0.5),
            );

            for _ in 0..32 {
                let (out, _) = splats.render(
                    &camera,
                    glam::uvec2(w as u32, h as u32),
                    glam::vec3(0.0, 0.0, 0.0),
                    false,
                );
                let out_rgb = out.clone().slice([0..h, 0..w, 0..3]);
                let _ = (out_rgb.clone() - crab_tens.clone())
                    .powf_scalar(2.0)
                    .mean()
                    .backward();
            }

            // // Wait for GPU work.
            <BurnBack as burn::prelude::Backend>::sync(
                &device,
                burn::tensor::backend::SyncType::Wait,
            );
        })
}
