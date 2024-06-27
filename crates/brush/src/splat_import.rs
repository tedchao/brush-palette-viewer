use brush_render::{render::num_sh_coeffs, AutodiffBackend};
use burn::{
    module::{Param, ParamId},
    tensor::{Tensor, TensorData},
};
use ply_rs::{
    parser::Parser,
    ply::{Property, PropertyAccess},
};
use single_value_channel::Updater;
use std::io::BufRead;
use tracing::info_span;

use crate::gaussian_splats::Splats;
use anyhow::{Context, Result};

const SH_COEFFS_PER_CHANNEL: usize = num_sh_coeffs(3);
const SH_COEFFS_PER_SPLAT: usize = SH_COEFFS_PER_CHANNEL * 3;

pub(crate) struct GaussianData {
    means: glam::Vec3,
    scale: glam::Vec3,
    opacity: f32,
    rotation: glam::Vec4,
    sh_coeffs: [f32; SH_COEFFS_PER_SPLAT],
}

fn inv_sigmoid(v: f32) -> f32 {
    (v / (1.0 - v)).ln()
}

const SH_C0: f32 = 0.28209479;

fn to_interleaved_idx(val: usize) -> usize {
    let channel = val / SH_COEFFS_PER_CHANNEL;
    let coeff = (val % (SH_COEFFS_PER_CHANNEL - 1)) + 1;
    coeff * 3 + channel
}

impl PropertyAccess for GaussianData {
    fn new() -> Self {
        GaussianData {
            means: glam::Vec3::ZERO,
            scale: glam::Vec3::ONE,
            opacity: 1.0,
            rotation: glam::Vec4::ZERO,
            sh_coeffs: [0.0; SH_COEFFS_PER_SPLAT],
        }
    }

    fn set_property(&mut self, key: &str, property: Property) {
        if let Property::Float(v) = property {
            match key {
                "x" => self.means.x = v,
                "y" => self.means.y = v,
                "z" => self.means.z = v,
                "scale_0" => self.scale.x = v,
                "scale_1" => self.scale.y = v,
                "scale_2" => self.scale.z = v,
                "opacity" => self.opacity = v,
                "rot_0" => self.rotation.x = v,
                "rot_1" => self.rotation.y = v,
                "rot_2" => self.rotation.z = v,
                "rot_3" => self.rotation.w = v,
                "f_dc_0" => self.sh_coeffs[0] = v,
                "f_dc_1" => self.sh_coeffs[1] = v,
                "f_dc_2" => self.sh_coeffs[2] = v,
                _ if key.starts_with("f_rest_") => {
                    if let Ok(idx) = key["f_rest_".len()..].parse::<u32>() {
                        self.sh_coeffs[to_interleaved_idx(idx as usize)] = v;
                    }
                }
                _ => (),
            }
        }
    }
}

fn update_splats<B: AutodiffBackend>(
    splats: &mut Option<Splats<B>>,
    means: Vec<f32>,
    sh_coeffs: Vec<f32>,
    rotation: Vec<f32>,
    opacity: Vec<f32>,
    scales: Vec<f32>,
    device: &B::Device,
) {
    let num_splats = means.len() / 3;
    let num_coeffs = sh_coeffs.len() / num_splats;

    let new_means = Tensor::from_data(TensorData::new(means, [num_splats, 3]), device);
    let new_coeffs =
        Tensor::from_data(TensorData::new(sh_coeffs, [num_splats, num_coeffs]), device);
    let new_rots = Tensor::from_data(TensorData::new(rotation, [num_splats, 4]), device);
    let new_opac = Tensor::from_data(TensorData::new(opacity, [num_splats]), device);
    let new_scales = Tensor::from_data(TensorData::new(scales, [num_splats, 3]), device);

    if let Some(splats) = splats.as_mut() {
        splats.concat_splats(new_means, new_rots, new_coeffs, new_opac, new_scales);
    } else {
        // Create a new splat instance if it hasn't been initialzized yet.
        *splats = Some(Splats {
            means: Param::initialized(ParamId::new(), new_means),
            sh_coeffs: Param::initialized(ParamId::new(), new_coeffs),
            rotation: Param::initialized(ParamId::new(), new_rots),
            raw_opacity: Param::initialized(ParamId::new(), new_opac),
            log_scales: Param::initialized(ParamId::new(), new_scales),
            xys_dummy: Tensor::zeros([num_splats, 2], device),
        });
    }
}

// TODO: This is better modelled by an async stream I think.
pub fn load_splat_from_ply<B: AutodiffBackend>(
    ply_data: &[u8],
    device: &B::Device,
    updater: &Updater<Option<Splats<B>>>,
) -> Result<Splats<B>> {
    // set up a reader, in this case a file.
    let mut reader = std::io::Cursor::new(ply_data);
    let gaussian_parser = Parser::<GaussianData>::new();
    let header = gaussian_parser.read_header(&mut reader)?;

    let mut splats: Option<Splats<B>> = None;

    let update_every = 200000;

    let mut means = Vec::with_capacity(update_every);
    let mut sh_coeffs = Vec::with_capacity(update_every);
    let mut rotation = Vec::with_capacity(update_every);
    let mut opacity = Vec::with_capacity(update_every);
    let mut scales = Vec::with_capacity(update_every);

    let _span = info_span!("Read splats").entered();

    for element in &header.elements {
        if element.name == "vertex" {
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

                means.extend(<[f32; 3]>::from(splat.means));
                sh_coeffs.extend(splat.sh_coeffs);
                rotation.extend(<[f32; 4]>::from(splat.rotation.normalize()));
                opacity.push(splat.opacity);
                scales.extend(<[f32; 3]>::from(splat.scale));

                // Occasionally send some updated splats.
                if i % update_every == 0 {
                    update_splats(
                        &mut splats,
                        means,
                        sh_coeffs,
                        rotation,
                        opacity,
                        scales,
                        device,
                    );
                    means = Vec::with_capacity(update_every);
                    sh_coeffs = Vec::with_capacity(update_every);
                    rotation = Vec::with_capacity(update_every);
                    opacity = Vec::with_capacity(update_every);
                    scales = Vec::with_capacity(update_every);
                    let _ = updater.update(splats.clone());
                }
            }
        }
    }

    update_splats(
        &mut splats,
        means,
        sh_coeffs,
        rotation,
        opacity,
        scales,
        device,
    );
    let _ = updater.update(splats.clone());
    splats.context("Empty ply file.")
}
