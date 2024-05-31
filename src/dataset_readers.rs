use std::path::PathBuf;

use crate::camera::Camera;
use crate::{camera, scene};
use anyhow::Context;
use anyhow::Result;
use ndarray::Array3;

#[derive(Debug, Default)]
pub(crate) struct InputView {
    pub(crate) image: Array3<f32>, // RGBA image.
}

#[derive(Debug, Default)]
pub(crate) struct InputData {
    pub(crate) camera: Camera,
    pub(crate) view: InputView,
}

fn read_synthetic_nerf_data(
    base_path: &str,
    transformsfile: &str,
    extension: &str,
    max_frames: Option<usize>,
) -> Result<Vec<InputData>> {
    let mut cameras = vec![];

    let path = PathBuf::from(base_path).join(transformsfile);
    let file = std::fs::read_to_string(path).expect("Couldn't find transforms file.");

    let contents: serde_json::Value = serde_json::from_str(&file).unwrap();
    let fovx = contents
        .get("camera_angle_x")
        .context("Camera angle x")?
        .as_f64()
        .context("Parsing camera angle")? as f32;

    let frames_array = contents
        .get("frames")
        .context("Frames arary")?
        .as_array()
        .context("Parsing frames array")?;

    for (i, frame) in frames_array.iter().enumerate() {
        // NeRF 'transform_matrix' is a camera-to-world transform
        let transform_matrix = frame
            .get("transform_matrix")
            .context("Get transform matrix")?
            .as_array()
            .context("Transform as array")?;

        let transform_matrix: Vec<f32> = transform_matrix
            .iter()
            .flat_map(|x| {
                x.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
            })
            .collect();
        let mut transform = glam::Mat4::from_cols_slice(&transform_matrix).transpose();
        // TODO: This is all so strange? How did this even happen?
        // The original comment here makes no sense, we're transforming between some axis spaces
        // but without changing the transform, which can't be right.
        transform.y_axis *= -1.0;
        transform.z_axis *= -1.0;

        let (_, rotation, translation) = transform.to_scale_rotation_translation();

        let image_file_path = frame
            .get("file_path")
            .context("Get file path")?
            .as_str()
            .context("File path as str")?;

        let image_path = PathBuf::from(base_path).join(image_file_path.to_string() + extension);
        let image = image::io::Reader::open(image_path)?.decode()?;
        // let image = image.resize(200, 200, image::imageops::FilterType::Lanczos3);

        let im_data = image.to_rgba8().into_vec();
        let tensor = Array3::from_shape_vec(
            [image.width() as usize, image.height() as usize, 4],
            im_data,
        )?;

        let fovy = camera::focal_to_fov(camera::fov_to_focal(fovx, image.width()), image.height());

        cameras.push(InputData {
            camera: Camera::new(translation, rotation, fovx, fovy),
            view: InputView {
                image: tensor.to_owned().map(|&x| (x as f32) / 255.0),
            },
        });

        if let Some(max) = max_frames {
            if i == max - 1 {
                break;
            }
        }
    }

    Ok(cameras)
}

pub(crate) fn read_viewpoint_data(file: &str) -> Result<Vec<Camera>> {
    let mut cameras = vec![];

    let file = std::fs::read_to_string(file).expect("Couldn't find viewpoints file.");
    let contents: serde_json::Value = serde_json::from_str(&file).unwrap();

    let frames_array = contents.as_array().context("Parsing cameras as list")?;

    for cam in frames_array.iter() {
        // NeRF 'transform_matrix' is a camera-to-world transform
        let translation = cam
            .get("position")
            .context("Get transform matrix")?
            .as_array()
            .context("Transform as array")?;
        let translation: Vec<f32> = translation
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let translation = glam::vec3(translation[0], translation[1], translation[2]);

        let rot_matrix = cam
            .get("rotation")
            .context("Get rotation")?
            .as_array()
            .context("rotation as array")?;

        let rot_matrix: Vec<f32> = rot_matrix
            .iter()
            .flat_map(|x| {
                x.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
            })
            .collect();
        let rot_matrix = glam::Mat3::from_cols_slice(&rot_matrix).transpose();

        let width = cam
            .get("width")
            .context("Get width")?
            .as_i64()
            .context("parse width")? as u32;

        let height = cam
            .get("height")
            .context("Get height")?
            .as_i64()
            .context("parse height")? as u32;

        let fx = cam
            .get("fx")
            .context("Get fx")?
            .as_f64()
            .context("parse fx")?;

        let fy = cam
            .get("fy")
            .context("Get fy")?
            .as_f64()
            .context("parse fy")?;

        let rotation = glam::Quat::from_mat3(&rot_matrix);
        let fovx = camera::focal_to_fov(fx as f32, width);
        let fovy = camera::focal_to_fov(fy as f32, height);
        cameras.push(Camera::new(translation, rotation, fovx, fovy));
    }

    Ok(cameras)
}

pub(crate) fn read_scene(
    scene_path: &str,
    max_images: Option<usize>,
    with_test_data: bool,
) -> scene::Scene {
    let train_cams =
        read_synthetic_nerf_data(scene_path, "transforms_train.json", ".png", max_images)
            .expect("Failed to load train cameras.");

    let mut test_cams = vec![];

    if with_test_data {
        test_cams =
            read_synthetic_nerf_data(scene_path, "transforms_test.json", ".png", max_images)
                .expect("Failed to load test cameras.");
    }

    scene::Scene::new(train_cams, test_cams)
}
