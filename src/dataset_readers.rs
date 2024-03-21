use crate::camera::InputData;
use crate::camera::{Camera, InputView};
use crate::{camera, scene};
use anyhow::Result;
use ndarray::Array3;
use rerun::external::anyhow::Context;
use rerun::external::glam;
use std::path::PathBuf;

fn read_synthetic_nerf_data(
    base_path: &str,
    transformsfile: &str,
    extension: &str,
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

    println!("Loaded {transformsfile}");

    let frames_array = contents
        .get("frames")
        .context("Frames arary")?
        .as_array()
        .context("Parsing frames array")?;

    for frame in frames_array {
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

        //*&mut c2w.slice_mut(s![0..3, 1..3]) *= -1.0;

        let (_, rotation, translation) = transform.to_scale_rotation_translation();

        let image_file_path = frame
            .get("file_path")
            .context("Get file path")?
            .as_str()
            .context("File path as str")?;

        let image_path = PathBuf::from(base_path).join(image_file_path.to_string() + extension);
        let image = image::io::Reader::open(image_path)?.decode()?;
        let image = image.resize(200, 200, image::imageops::FilterType::Lanczos3);

        let im_data = image.to_rgba8().into_vec();
        let tensor = Array3::from_shape_vec(
            [image.width() as usize, image.height() as usize, 4],
            im_data,
        )?;

        let fovy = camera::focal_to_fov(camera::fov_to_focal(fovx, image.width()), image.height());

        cameras.push(InputData {
            camera: Camera::new(
                translation,
                rotation,
                fovx,
                fovy,
                image.width(),
                image.height(),
                0.01,
                1.0,
            ),
            view: InputView {
                image: tensor.to_owned().map(|&x| (x as f32) / 255.0),
            },
        })
    }

    Ok(cameras)
}

pub(crate) fn read_scene(scene_path: &str) -> scene::Scene {
    println!("Reading Training Transforms");
    let train_cams = read_synthetic_nerf_data(scene_path, "transforms_train.json", ".png")
        .expect("Failed to load train cameras.");

    // println!("Reading Test Transforms");
    // let test_cams = read_synthetic_nerf_data(scene_path, "transforms_test.json", ".png")
    //     .expect("Failed to load test cameras.");

    scene::Scene::new(train_cams, vec![])
}
