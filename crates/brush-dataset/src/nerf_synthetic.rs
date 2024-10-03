use std::io::Cursor;
use std::io::Read;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use brush_render::camera;
use brush_render::camera::Camera;
use brush_train::scene::Scene;
use brush_train::scene::SceneView;
use zip::ZipArchive;

use crate::clamp_img_to_max_size;
use crate::normalized_path_string;
use crate::Dataset;

#[derive(serde::Deserialize)]
struct SyntheticScene {
    camera_angle_x: f32,
    frames: Vec<FrameData>,
}

#[derive(serde::Deserialize)]
struct FrameData {
    transform_matrix: Vec<Vec<f32>>,
    file_path: String,
}

fn read_transforms_file(
    archive: &mut ZipArchive<Cursor<&[u8]>>,
    name: &str,
    max_frames: Option<usize>,
    max_resolution: Option<u32>,
) -> anyhow::Result<Scene> {
    let mut views = vec![];

    let transform_fname = archive
        .file_names()
        .find(|x| x.ends_with(name))
        .context("No transforms file")?
        .to_owned();
    let base_path = Path::new(&transform_fname)
        .parent()
        .unwrap_or(Path::new("./"));
    let transform_buf = {
        let mut transforms_file = archive.by_name(&transform_fname)?;
        let mut transform_buf = String::new();
        transforms_file.read_to_string(&mut transform_buf)?;
        transform_buf
    };
    let scene_train: SyntheticScene = serde_json::from_str(&transform_buf)?;

    let fovx = scene_train.camera_angle_x;
    for (i, frame) in scene_train.frames.iter().enumerate() {
        // NeRF 'transform_matrix' is a camera-to-world transform
        let transform_matrix: Vec<f32> = frame.transform_matrix.iter().flatten().copied().collect();
        let mut transform = glam::Mat4::from_cols_slice(&transform_matrix).transpose();

        // Swap basis to go from z-up, left handed (a la OpenCV) to our kernel format
        // (right-handed, y-down).
        transform.y_axis *= -1.0;
        transform.z_axis *= -1.0;

        transform = glam::Mat4::from_rotation_x(std::f32::consts::PI / 2.0) * transform;

        let (_, rotation, translation) = transform.to_scale_rotation_translation();

        let image_path =
            normalized_path_string(&base_path.join(frame.file_path.to_owned() + ".png"));

        let img_buffer = archive
            .by_name(&image_path)?
            .bytes()
            .collect::<Result<Vec<_>, _>>()?;

        // Create a cursor from the buffer
        let mut image = image::load_from_memory(&img_buffer)?;

        if let Some(max_resolution) = max_resolution {
            image = clamp_img_to_max_size(image, max_resolution);
        }

        // Blend in white background to image
        let mut rgba_image = image.to_rgba8();
        for pixel in rgba_image.pixels_mut() {
            let alpha = pixel[3] as f32 / 255.0;
            pixel[0] = (pixel[0] as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
            pixel[1] = (pixel[1] as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
            pixel[2] = (pixel[2] as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
            pixel[3] = 255;
        }
        image = image::DynamicImage::ImageRgba8(rgba_image).to_rgb8().into();

        let fovy = camera::focal_to_fov(camera::fov_to_focal(fovx, image.width()), image.height());

        views.push(SceneView {
            name: image_path,
            camera: Camera::new(
                translation,
                rotation,
                glam::vec2(fovx, fovy),
                glam::vec2(0.5, 0.5),
            ),
            image,
        });

        if let Some(max) = max_frames {
            if i == max - 1 {
                break;
            }
        }
    }
    // Assume nerf synthetic has a white background. Maybe add a custom json field to customize this
    // or something.
    let background = glam::Vec3::ONE;
    Ok(Scene::new(views, background))
}

// TODO: This could be simplified with some serde-fu by creating a struct
// we deserialize into.
pub fn read_dataset(
    zip_data: &[u8],
    max_frames: Option<usize>,
    max_resolution: Option<u32>,
) -> Result<Dataset> {
    let mut archive = zip::ZipArchive::new(Cursor::new(zip_data))?;
    let train_scene = read_transforms_file(
        &mut archive,
        "transforms_train.json",
        max_frames,
        max_resolution,
    )?;
    // Not entirely sure yet if we want to report stats on both test
    // and eval.
    // let test_scene = read_transforms_file(
    //     &mut archive,
    //     "transforms_test.json",
    //     max_frames,
    //     max_resolution,
    // ).ok();
    let eval_scene = read_transforms_file(
        &mut archive,
        "transforms_val.json",
        max_frames,
        max_resolution,
    )
    .ok();

    Ok(Dataset {
        train: train_scene,
        test: None,
        eval: eval_scene,
    })
}
