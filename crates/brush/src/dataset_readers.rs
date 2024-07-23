use std::io::Cursor;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use brush_render::camera;
use brush_render::camera::Camera;
use ndarray::Array3;

#[derive(Debug, Default, Clone)]
pub(crate) struct InputView {
    pub(crate) camera: Camera,
    pub(crate) image: Array3<f32>,
}

fn normalized_path_string(path: &Path) -> String {
    Path::new(path)
        .components()
        .collect::<PathBuf>()
        .to_str()
        .unwrap()
        .replace(std::path::MAIN_SEPARATOR, "/")
}

pub fn read_synthetic_nerf_data(
    zip_data: &[u8],
    max_frames: Option<usize>,
) -> Result<Vec<InputView>> {
    let mut cameras = vec![];

    let mut archive = zip::ZipArchive::new(Cursor::new(zip_data))?;

    let transform_fname = archive
        .file_names()
        .find(|x| x.contains("transforms_train.json"))
        .unwrap()
        .to_owned();

    let base_path = Path::new(&transform_fname)
        .parent()
        .unwrap_or(Path::new("./"));

    let contents: serde_json::Value = {
        let mut transforms_file = archive.by_name(&transform_fname)?;
        let mut transform_buf = String::new();
        transforms_file.read_to_string(&mut transform_buf)?;
        serde_json::from_str(&transform_buf)?
    };

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

        transform = glam::Mat4::from_rotation_x(std::f32::consts::PI / 2.0) * transform;

        let (_, rotation, translation) = transform.to_scale_rotation_translation();

        let image_file_path = frame
            .get("file_path")
            .context("Get file path")?
            .as_str()
            .context("File path as str")?;

        let image_path =
            normalized_path_string(&base_path.join(image_file_path.to_owned() + ".png"));

        let mut img_buffer = Vec::new();
        archive.by_name(&image_path)?.read_to_end(&mut img_buffer)?;
        // Create a cursor from the buffer
        let image = image::ImageReader::new(Cursor::new(img_buffer))
            .with_guessed_format()?
            .decode()?;

        // let image = image::io::Reader::open(image_path)?.decode()?;
        let image = image.resize(400, 400, image::imageops::FilterType::Lanczos3);

        let im_data = image.to_rgba8().into_vec();
        let tensor = Array3::from_shape_vec(
            [image.width() as usize, image.height() as usize, 4],
            im_data,
        )?;

        let fovy = camera::focal_to_fov(camera::fov_to_focal(fovx, image.width()), image.height());

        cameras.push(InputView {
            camera: Camera::new(translation, rotation, fovx, fovy),
            image: tensor.to_owned().map(|&x| (x as f32) / 255.0),
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
