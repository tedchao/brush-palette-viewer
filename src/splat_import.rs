use std::io::BufReader;

use burn::{
    module::{Param, ParamId},
    tensor::{Data, Shape, Tensor},
};
use ply_rs::{
    parser::Parser,
    ply::{Property, PropertyAccess},
};

use crate::{gaussian_splats::Splats, splat_render::Backend};
use anyhow::Result;

#[derive(Default)]
pub(crate) struct GaussianData {
    means: glam::Vec3,
    colors: glam::Vec3,
    scale: glam::Vec3,
    opacity: f32,
    rotation: glam::Vec4,
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
        match (key.as_ref(), property) {
            // TODO: SH.
            ("x", Property::Float(v)) => self.means[0] = v,
            ("y", Property::Float(v)) => self.means[1] = v,
            ("z", Property::Float(v)) => self.means[2] = v,

            // TODO: This 0.5 shouldn't be needed anymore once we do full SH?
            ("f_dc_0", Property::Float(v)) => self.colors[0] = v * SH_C0 + 0.5,
            ("f_dc_1", Property::Float(v)) => self.colors[1] = v * SH_C0 + 0.5,
            ("f_dc_2", Property::Float(v)) => self.colors[2] = v * SH_C0 + 0.5,

            ("scale_0", Property::Float(v)) => self.scale[0] = v,
            ("scale_1", Property::Float(v)) => self.scale[1] = v,
            ("scale_2", Property::Float(v)) => self.scale[2] = v,

            ("opacity", Property::Float(v)) => self.opacity = v,

            ("rot_0", Property::Float(v)) => self.rotation[0] = v,
            ("rot_1", Property::Float(v)) => self.rotation[1] = v,
            ("rot_2", Property::Float(v)) => self.rotation[2] = v,
            ("rot_3", Property::Float(v)) => self.rotation[3] = v,

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
        .flat_map(|d| [d.means.x, d.means.y, d.means.z, 0.0])
        .collect::<Vec<_>>();
    let colors = cloud
        .iter()
        .flat_map(|d| [d.colors.x, d.colors.y, d.colors.z, 0.0])
        .collect::<Vec<_>>();
    let rotation = cloud
        .iter()
        .flat_map(|d| [d.rotation.x, d.rotation.y, d.rotation.z, d.rotation.w])
        .collect::<Vec<_>>();
    let opacity = cloud.iter().map(|d| d.opacity).collect::<Vec<_>>();
    let scales = cloud
        .iter()
        .flat_map(|d| [d.scale.x, d.scale.y, d.scale.z, 0.0])
        .collect::<Vec<_>>();

    let num_points = cloud.len();

    let splats = Splats {
        means: Param::initialized(
            ParamId::new(),
            Tensor::from_data(
                Data::new(means, Shape::new([num_points, 4])).convert(),
                device,
            )
            .require_grad(),
        ),
        colors: Param::initialized(
            ParamId::new(),
            Tensor::from_data(
                Data::new(colors, Shape::new([num_points, 4])).convert(),
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
        opacity: Param::initialized(
            ParamId::new(),
            Tensor::from_data(
                Data::new(opacity, Shape::new([num_points])).convert(),
                device,
            )
            .require_grad(),
        ),
        scales: Param::initialized(
            ParamId::new(),
            Tensor::from_data(
                Data::new(scales, Shape::new([num_points, 4])).convert(),
                device,
            )
            .require_grad(),
        ),
    };

    Ok(splats)
}
