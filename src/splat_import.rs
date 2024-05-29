use burn::{
    module::{Param, ParamId},
    tensor::{Data, Shape, Tensor},
};
use ply_rs::{
    parser::Parser,
    ply::{Property, PropertyAccess},
};
use std::io::BufReader;

use crate::{gaussian_splats::num_sh_coeffs, gaussian_splats::Splats, splat_render::Backend};
use anyhow::Result;

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
            // TODO: SH.
            ("x", Property::Float(v)) => self.means[0] = v,
            ("y", Property::Float(v)) => self.means[1] = v,
            ("z", Property::Float(v)) => self.means[2] = v,

            ("scale_0", Property::Float(v)) => self.scale[0] = v,
            ("scale_1", Property::Float(v)) => self.scale[1] = v,
            ("scale_2", Property::Float(v)) => self.scale[2] = v,

            ("opacity", Property::Float(v)) => self.opacity = v,

            ("rot_0", Property::Float(v)) => self.rotation[0] = v,
            ("rot_1", Property::Float(v)) => self.rotation[1] = v,
            ("rot_2", Property::Float(v)) => self.rotation[2] = v,
            ("rot_3", Property::Float(v)) => self.rotation[3] = v,

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

pub fn load_splat_from_ply<B: Backend>(file: &str, device: &B::Device) -> Result<Splats<B>> {
    // set up a reader, in this case a file.
    let f = std::fs::File::open(file).unwrap();
    let mut reader = BufReader::new(f);
    let gaussian_parser = Parser::<GaussianData>::new();
    let header = gaussian_parser.read_header(&mut reader)?;

    let mut cloud = Vec::new();

    for (_ignore_key, element) in &header.elements {
        if element.name == "vertex" {
            cloud = gaussian_parser.read_payload_for_element(&mut reader, element, &header)?;
        }
    }

    // Return normalized rotations.
    for gaussian in &mut cloud {
        // TODO: Clamp maximum variance? Is that needed?
        // TODO: Is scale in log(scale) or scale format?
        //
        // for i in 0..3 {
        //     gaussian.scale_opacity.scale[i] = gaussian.scale_opacity.scale[i]
        //         .max(mean_scale - MAX_SIZE_VARIANCE)
        //         .min(mean_scale + MAX_SIZE_VARIANCE)
        //         .exp();
        // }
        gaussian.rotation = gaussian.rotation.normalize();
    }

    // TODO: This is all terribly space inefficient.
    let means = cloud
        .iter()
        .flat_map(|d| [d.means.x, d.means.y, d.means.z])
        .collect::<Vec<_>>();
    let sh_coeffs = cloud
        .iter()
        .flat_map(|d| d.sh_coeffs.iter().copied())
        .collect::<Vec<_>>();
    let rotation = cloud
        .iter()
        .flat_map(|d| [d.rotation.x, d.rotation.y, d.rotation.z, d.rotation.w])
        .collect::<Vec<_>>();
    let opacity = cloud.iter().map(|d| d.opacity).collect::<Vec<_>>();
    let scales = cloud
        .iter()
        .flat_map(|d| [d.scale.x, d.scale.y, d.scale.z])
        .collect::<Vec<_>>();

    let num_points = cloud.len();

    let num_coeffs = cloud.first().unwrap().sh_coeffs.len();

    let splats = Splats {
        means: Param::initialized(
            ParamId::new(),
            Tensor::from_data(
                Data::new(means, Shape::new([num_points, 3])).convert(),
                device,
            )
            .require_grad(),
        ),
        sh_coeffs: Param::initialized(
            ParamId::new(),
            Tensor::from_data(
                Data::new(sh_coeffs, Shape::new([num_points, num_coeffs])).convert(),
                device,
            )
            .require_grad(),
        ),
        rotation: Param::initialized(
            ParamId::new(),
            Tensor::from_data(
                Data::new(rotation, Shape::new([num_points, 4])).convert(),
                device,
            )
            .require_grad(),
        ),
        raw_opacity: Param::initialized(
            ParamId::new(),
            Tensor::from_data(
                Data::new(opacity, Shape::new([num_points])).convert(),
                device,
            )
            .require_grad(),
        ),
        log_scales: Param::initialized(
            ParamId::new(),
            Tensor::from_data(
                Data::new(scales, Shape::new([num_points, 3])).convert(),
                device,
            )
            .require_grad(),
        ),
        xys_dummy: Tensor::zeros([num_points, 2], device).require_grad(),
    };

    Ok(splats)
}
