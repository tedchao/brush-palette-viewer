use brush_render::{render::num_sh_coeffs, AutodiffBackend};
use burn::{
    module::{Param, ParamId},
    tensor::{Data, Shape, Tensor},
};
use ply_rs::{
    parser::Parser,
    ply::{Property, PropertyAccess},
};
use single_value_channel::Updater;
use std::io::{BufRead, BufReader};

use crate::gaussian_splats::Splats;
use anyhow::{Context, Result};

#[derive(Default)]
pub(crate) struct GaussianData {
    means: glam::Vec3,
    scale: glam::Vec3,
    opacity: f32,
    rotation: glam::Vec4,
    sh_coeffs: Vec<f32>,
}

fn inv_sigmoid(v: f32) -> f32 {
    (v / (1.0 - v)).ln()
}

const SH_C0: f32 = 0.28209479;

impl PropertyAccess for GaussianData {
    fn new() -> Self {
        GaussianData::default()
    }

    fn set_property(&mut self, key: String, property: Property) {
        let sh_coeff_per_channel: usize = num_sh_coeffs(3);

        match (key.as_ref(), property) {
            ("x", Property::Float(v)) => self.means.x = v,
            ("y", Property::Float(v)) => self.means.y = v,
            ("z", Property::Float(v)) => self.means.z = v,

            ("scale_0", Property::Float(v)) => self.scale.x = v,
            ("scale_1", Property::Float(v)) => self.scale.y = v,
            ("scale_2", Property::Float(v)) => self.scale.z = v,

            ("opacity", Property::Float(v)) => self.opacity = v,

            ("rot_0", Property::Float(v)) => self.rotation.x = v,
            ("rot_1", Property::Float(v)) => self.rotation.y = v,
            ("rot_2", Property::Float(v)) => self.rotation.z = v,
            ("rot_3", Property::Float(v)) => self.rotation.w = v,

            (_, Property::Float(v)) if key.starts_with("f_rest_") || key.starts_with("f_dc_") => {
                let (coeff, channel) = if key.starts_with("f_dc_") {
                    let coeff = 0;
                    let channel = key.strip_prefix("f_dc_").unwrap().parse::<usize>().unwrap();
                    (coeff, channel)
                } else {
                    let i = key
                        .strip_prefix("f_rest_")
                        .unwrap()
                        .parse::<usize>()
                        .unwrap();

                    let channel = i / sh_coeff_per_channel;

                    let coeff = if sh_coeff_per_channel == 1 {
                        1
                    } else {
                        (i % (sh_coeff_per_channel - 1)) + 1
                    };
                    (coeff, channel)
                };

                // planar
                let interleaved_idx = coeff * 3 + channel;

                if self.sh_coeffs.len() < interleaved_idx + 1 {
                    self.sh_coeffs.resize(interleaved_idx + 1, 0.0);
                }
                self.sh_coeffs[interleaved_idx] = v;
            }
            (_, _) => {}
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

    let new_means = Tensor::from_data(
        Data::new(means, Shape::new([num_splats, 3])).convert(),
        device,
    );
    let new_coeffs = Tensor::from_data(
        Data::new(sh_coeffs, Shape::new([num_splats, num_coeffs])).convert(),
        device,
    );
    let new_rots = Tensor::from_data(
        Data::new(rotation, Shape::new([num_splats, 4])).convert(),
        device,
    );
    let new_opac = Tensor::from_data(
        Data::new(opacity, Shape::new([num_splats])).convert(),
        device,
    );
    let new_scales = Tensor::from_data(
        Data::new(scales, Shape::new([num_splats, 3])).convert(),
        device,
    );

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

    let mut means = Vec::new();
    let mut sh_coeffs = Vec::new();
    let mut rotation = Vec::new();
    let mut opacity = Vec::new();
    let mut scales = Vec::new();

    for (_ignore_key, element) in &header.elements {
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
                if i % 100000 == 0 {
                    update_splats(
                        &mut splats,
                        means,
                        sh_coeffs,
                        rotation,
                        opacity,
                        scales,
                        device,
                    );
                    means = Vec::new();
                    sh_coeffs = Vec::new();
                    rotation = Vec::new();
                    opacity = Vec::new();
                    scales = Vec::new();
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
