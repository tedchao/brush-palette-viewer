use anyhow::Context;
use anyhow::Result;
use async_fn_stream::try_fn_stream;
use brush_render::camera;
use brush_render::camera::Camera;
use brush_train::scene::Scene;
use brush_train::scene::SceneView;
use futures_lite::Stream;
use futures_lite::StreamExt;

use std::io::Cursor;
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;

use crate::clamp_img_to_max_size;
use crate::normalized_path_string;
use crate::Dataset;
use crate::ZipData;

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
    mut archive: ZipArchive<Cursor<ZipData>>,
    name: &'static str,
    max_frames: Option<usize>,
    max_resolution: Option<u32>,
) -> Result<impl Stream<Item = anyhow::Result<SceneView>>> {
    let transform_fname = archive
        .file_names()
        .find(|x| x.ends_with(name))
        .context("No transforms file")?
        .to_owned();
    let base_path = Path::new(&transform_fname)
        .parent()
        .unwrap_or(Path::new("./"))
        .to_owned();
    let transform_buf = {
        let mut transforms_file = archive.by_name(&transform_fname)?;
        let mut transform_buf = String::new();
        transforms_file.read_to_string(&mut transform_buf)?;
        transform_buf
    };
    let scene_train: SyntheticScene = serde_json::from_str(&transform_buf)?;

    Ok(try_fn_stream(|emitter| async move {
        let fovx = scene_train.camera_angle_x;

        for (i, frame) in scene_train.frames.iter().enumerate() {
            let _span = tracing::trace_span!("Dataset image").entered();

            // NeRF 'transform_matrix' is a camera-to-world transform
            let transform_matrix: Vec<f32> =
                frame.transform_matrix.iter().flatten().copied().collect();
            let mut transform = glam::Mat4::from_cols_slice(&transform_matrix).transpose();

            // Swap basis to go from z-up, left handed (a la OpenCV) to our kernel format
            // (right-handed, y-down).
            transform.y_axis *= -1.0;
            transform.z_axis *= -1.0;

            transform = glam::Mat4::from_rotation_x(std::f32::consts::PI / 2.0) * transform;

            let (_, rotation, translation) = transform.to_scale_rotation_translation();

            let image_path =
                normalized_path_string(&base_path.join(frame.file_path.to_owned() + ".png"));

            let comp_span = tracing::trace_span!("Decompress image").entered();
            let img_buffer = archive
                .by_name(&image_path)?
                .bytes()
                .collect::<Result<Vec<_>, _>>()?;
            drop(comp_span);

            // Create a cursor from the buffer
            let mut image = tracing::trace_span!("Decode image")
                .in_scope(|| image::load_from_memory(&img_buffer))?;

            if let Some(max_resolution) = max_resolution {
                image = clamp_img_to_max_size(image, max_resolution);
            }

            // Blend in white background to image
            if image.color().has_alpha() {
                let _span = tracing::trace_span!("Blend image").entered();
                let rgba_image = image.as_rgba8().context("Unsupported image")?;
                let mut rgb_image = image::RgbImage::new(image.width(), image.height());
                for (rgb, rgba) in rgb_image.pixels_mut().zip(rgba_image.pixels()) {
                    let alpha = rgba.0[3] as u32;
                    let r = ((255 - alpha) * 255 + alpha * rgba.0[0] as u32) / 255;
                    let g = ((255 - alpha) * 255 + alpha * rgba.0[1] as u32) / 255;
                    let b = ((255 - alpha) * 255 + alpha * rgba.0[2] as u32) / 255;
                    *rgb = image::Rgb([r as u8, g as u8, b as u8]);
                }
                image = rgb_image.into();
            }

            let fovy =
                camera::focal_to_fov(camera::fov_to_focal(fovx, image.width()), image.height());

            emitter
                .emit(SceneView {
                    name: image_path,
                    camera: Camera::new(
                        translation,
                        rotation,
                        glam::vec2(fovx, fovy),
                        glam::vec2(0.5, 0.5),
                    ),
                    image,
                })
                .await;

            if let Some(max) = max_frames {
                if i == max - 1 {
                    break;
                }
            }
        }
        Ok(())
    }))
}

pub fn read_dataset(
    archive: ZipArchive<Cursor<ZipData>>,
    max_frames: Option<usize>,
    max_resolution: Option<u32>,
) -> Result<impl Stream<Item = Result<Dataset>>> {
    // Assume nerf synthetic has a white background. Maybe add a custom json field to customize this
    // or something.
    let background = glam::Vec3::ONE;

    let mut dataset = Dataset::empty();

    let train_stream = read_transforms_file(
        archive.clone(),
        "transforms_train.json",
        max_frames,
        max_resolution,
    )?;

    let stream = try_fn_stream(|emitter| async move {
        let mut train_scene = Scene::new(vec![], background);
        let mut train_stream = std::pin::pin!(train_stream);
        while let Some(view) = train_stream.next().await {
            train_scene.add_view(view?);
            dataset.train = train_scene.clone();
            emitter.emit(dataset.clone()).await;
        }

        // Not entirely sure yet if we want to report stats on both test
        // and eval. If so, just read  "transforms_test.json"
        let val_stream =
            read_transforms_file(archive, "transforms_val.json", max_frames, max_resolution);

        if let Ok(val_stream) = val_stream {
            let mut val_stream = std::pin::pin!(val_stream);
            let mut val_scene = Scene::new(vec![], background);

            while let Some(view) = val_stream.next().await {
                val_scene.add_view(view?);
                dataset.eval = Some(val_scene.clone());
                emitter.emit(dataset.clone()).await;
            }
        }

        Ok(())
    });

    Ok(stream)
}
