use async_fn_stream::try_fn_stream;
use async_std::stream::Stream;
use brush_render::{render::num_sh_coeffs, Backend};
use burn::{
    module::{Param, ParamId},
    tensor::{Tensor, TensorData},
};
use ply_rs::{
    parser::Parser,
    ply::{Property, PropertyAccess},
};
use std::io::BufRead;
use tracing::trace_span;

#[allow(unused)]
use web_time::{Duration, Instant};

use anyhow::{Context, Result};
use brush_render::gaussian_splats::Splats;

pub(crate) struct GaussianData {
    means: [f32; 3],
    scale: [f32; 3],
    opacity: f32,
    rotation: [f32; 4],
    sh_dc: [f32; 3],
    sh_coeffs: Vec<f32>,
}

impl PropertyAccess for GaussianData {
    fn new() -> Self {
        GaussianData {
            means: [0.0; 3],
            scale: [0.0; 3],
            opacity: 0.0,
            rotation: [0.0; 4],
            sh_dc: [0.0, 0.0, 0.0],
            sh_coeffs: Vec::new(),
        }
    }

    fn set_property(&mut self, key: &str, property: Property) {
        if let Property::Float(v) = property {
            match key {
                "x" => self.means[0] = v,
                "y" => self.means[1] = v,
                "z" => self.means[2] = v,
                "scale_0" => self.scale[0] = v,
                "scale_1" => self.scale[1] = v,
                "scale_2" => self.scale[2] = v,
                "opacity" => self.opacity = v,
                "rot_0" => self.rotation[0] = v,
                "rot_1" => self.rotation[1] = v,
                "rot_2" => self.rotation[2] = v,
                "rot_3" => self.rotation[3] = v,
                "f_dc_0" => self.sh_dc[0] = v,
                "f_dc_1" => self.sh_dc[1] = v,
                "f_dc_2" => self.sh_dc[2] = v,
                _ if key.starts_with("f_rest_") => {
                    if let Ok(idx) = key["f_rest_".len()..].parse::<u32>() {
                        if idx >= self.sh_coeffs.len() as u32 {
                            self.sh_coeffs.resize(idx as usize + 1, 0.0);
                        }
                        self.sh_coeffs[idx as usize] = v;
                    }
                }
                _ => (),
            }
        }
    }
}

fn update_splats<B: Backend>(
    splats: &mut Option<Splats<B>>,
    means: Vec<f32>,
    sh_coeffs: Vec<f32>,
    rotation: Vec<f32>,
    raw_opacities: Vec<f32>,
    log_scales: Vec<f32>,
    device: &B::Device,
) {
    let n_splats = means.len() / 3;
    let n_coeffs = sh_coeffs.len() / n_splats;

    let means = Tensor::from_data(TensorData::new(means, [n_splats, 3]), device).require_grad();
    let sh_coeffs = Tensor::from_data(
        TensorData::new(sh_coeffs, [n_splats, n_coeffs / 3, 3]),
        device,
    )
    .require_grad();
    let rotations =
        Tensor::from_data(TensorData::new(rotation, [n_splats, 4]), device).require_grad();
    let raw_opacities =
        Tensor::from_data(TensorData::new(raw_opacities, [n_splats]), device).require_grad();
    let log_scales =
        Tensor::from_data(TensorData::new(log_scales, [n_splats, 3]), device).require_grad();

    if let Some(splats) = splats.as_mut() {
        Splats::map_param(&mut splats.means, |x| {
            Tensor::cat(vec![x, means.clone()], 0)
        });
        Splats::map_param(&mut splats.rotation, |x| {
            Tensor::cat(vec![x, rotations.clone()], 0)
        });
        Splats::map_param(&mut splats.sh_coeffs, |x| {
            Tensor::cat(vec![x, sh_coeffs.clone()], 0)
        });
        Splats::map_param(&mut splats.raw_opacity, |x| {
            Tensor::cat(vec![x, raw_opacities.clone()], 0)
        });
        Splats::map_param(&mut splats.log_scales, |x| {
            Tensor::cat(vec![x, log_scales.clone()], 0)
        });
        splats.norm_rotations();
    } else {
        let mut init = Splats {
            means: Param::initialized(ParamId::new(), means),
            sh_coeffs: Param::initialized(ParamId::new(), sh_coeffs),
            rotation: Param::initialized(ParamId::new(), rotations),
            raw_opacity: Param::initialized(ParamId::new(), raw_opacities),
            log_scales: Param::initialized(ParamId::new(), log_scales),
            xys_dummy: Tensor::zeros([n_splats, 2], device).require_grad(),
        };
        init.norm_rotations();
        // Create a new splat instance if it hasn't been initialzized yet.
        *splats = Some(init);
    }
}

pub fn ply_count(ply_data: &[u8]) -> Result<usize> {
    let mut reader = std::io::Cursor::new(ply_data);
    let gaussian_parser = Parser::<GaussianData>::new();
    let header = gaussian_parser.read_header(&mut reader)?;
    header
        .elements
        .iter()
        .find(|e| e.name == "vertex")
        .map(|e| e.count)
        .context("Invalid ply file")
}

fn interleave_coeffs(sh_dc: [f32; 3], sh_rest: &[f32]) -> Vec<f32> {
    let channels = 3;
    let coeffs_per_channel = sh_rest.len() / channels;
    let mut result = Vec::with_capacity(sh_rest.len() + 3);
    result.extend(sh_dc);

    for i in 0..coeffs_per_channel {
        for j in 0..channels {
            let index = j * coeffs_per_channel + i;
            result.push(sh_rest[index]);
        }
    }
    result
}

pub fn load_splat_from_ply<B: Backend>(
    ply_data: Vec<u8>,
    device: B::Device,
) -> impl Stream<Item = Result<Splats<B>>> + 'static {
    // set up a reader, in this case a file.
    let mut reader = std::io::Cursor::new(ply_data);
    let mut splats: Option<Splats<B>> = None;

    let update_every = 50000;
    let _span = trace_span!("Read splats").entered();
    let gaussian_parser = Parser::<GaussianData>::new();

    try_fn_stream(|emitter| async move {
        let header = gaussian_parser.read_header(&mut reader)?;

        for element in &header.elements {
            if element.name == "vertex" {
                let min_props = [
                    "x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1",
                    "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2",
                ];

                if !min_props
                    .iter()
                    .all(|p| element.properties.iter().any(|x| &x.name == p))
                {
                    Err(anyhow::anyhow!("Invalid splat ply. Missing properties!"))?
                }

                let sh_coeffs_per_channel = (1 + element
                    .properties
                    .iter()
                    .filter_map(|x| {
                        x.name
                            .strip_prefix("f_rest_")
                            .and_then(|x| x.parse::<u32>().ok())
                    })
                    .max()
                    .unwrap_or(0)
                    / 3) as usize;

                let mut means = Vec::with_capacity(update_every * 3);
                let mut sh_coeffs = Vec::with_capacity(update_every * sh_coeffs_per_channel * 3);
                let mut rotation = Vec::with_capacity(update_every * 4);
                let mut opacity = Vec::with_capacity(update_every);
                let mut scales = Vec::with_capacity(update_every * 3);

                #[cfg(target_arch = "wasm32")]
                let mut last_yield_time = Instant::now();

                for i in 0..element.count {
                    let splat = match header.encoding {
                        ply_rs::ply::Encoding::Ascii => {
                            let mut line = String::new();
                            reader.read_line(&mut line)?;
                            gaussian_parser.read_ascii_element(&line, element)?
                        }
                        ply_rs::ply::Encoding::BinaryBigEndian => {
                            gaussian_parser.read_big_endian_element(&mut reader, element)?
                        }
                        ply_rs::ply::Encoding::BinaryLittleEndian => {
                            gaussian_parser.read_little_endian_element(&mut reader, element)?
                        }
                    };

                    let mut sh_coeffs_interleaved =
                        interleave_coeffs(splat.sh_dc, &splat.sh_coeffs);

                    // Limit the number of imported SH channels for now.
                    let max_sh_len = num_sh_coeffs(3) as usize * 3;

                    if sh_coeffs_interleaved.len() > max_sh_len {
                        sh_coeffs_interleaved.truncate(max_sh_len);
                    }

                    means.extend(splat.means);
                    sh_coeffs.extend(sh_coeffs_interleaved);
                    rotation.extend(splat.rotation);
                    opacity.push(splat.opacity);
                    scales.extend(splat.scale);

                    // Occasionally send some updated splats.
                    if i % update_every == update_every - 1 {
                        update_splats(
                            &mut splats,
                            means.clone(),
                            sh_coeffs.clone(),
                            rotation.clone(),
                            opacity.clone(),
                            scales.clone(),
                            &device,
                        );
                        means.clear();
                        sh_coeffs.clear();
                        rotation.clear();
                        opacity.clear();
                        scales.clear();

                        emitter
                            .emit(splats.clone().context("Failed to update splats")?)
                            .await;
                    }

                    // On wasm, yield to the browser occsionally
                    #[cfg(target_arch = "wasm32")]
                    if i % 100 == 0 && Instant::now() - last_yield_time > Duration::from_millis(5) {
                        // You'd think yield_now() should work but that doesn't actually hand back control to the browser.
                        // async_std::task::yield_now();
                        last_yield_time = Instant::now();

                        // Timeout future actually yield back to browser.
                        let never = async_std::future::pending::<()>();
                        let dur = Duration::from_millis(0);
                        let _ = async_std::future::timeout(dur, never).await;
                    }
                }

                update_splats(
                    &mut splats,
                    means,
                    sh_coeffs,
                    rotation,
                    opacity,
                    scales,
                    &device,
                );

                if let Some(splats) = splats.as_ref() {
                    if splats.num_splats() == 0 {
                        Err(anyhow::anyhow!("No splats found"))?;
                    }
                }

                emitter
                    .emit(splats.clone().context("Invalid ply file.")?)
                    .await;
            }
        }

        Ok(())
    })
}
