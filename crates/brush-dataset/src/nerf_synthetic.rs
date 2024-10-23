use anyhow::Context;
use anyhow::Result;
use async_fn_stream::try_fn_stream;
use brush_render::camera;
use brush_render::camera::Camera;
use brush_train::scene::SceneView;
use std::future::Future;
use std::io::Cursor;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use zip::ZipArchive;

use crate::{
    clamp_img_to_max_size, normalized_path_string, DataStream, Dataset, LoadDatasetArgs, ZipData,
};

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
    load_args: &LoadDatasetArgs,
) -> Result<Vec<impl Future<Output = anyhow::Result<SceneView>>>> {
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
    let fovx = scene_train.camera_angle_x;

    let iter = scene_train
        .frames
        .into_iter()
        .take(load_args.max_frames.unwrap_or(usize::MAX))
        .map(move |frame| {
            let base_path = base_path.clone();
            let mut archive = archive.clone();
            let load_args = load_args.clone();

            async move {
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

                if let Some(max_resolution) = load_args.max_resolution {
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

                let view = SceneView {
                    name: image_path,
                    camera: Camera::new(
                        translation,
                        rotation,
                        glam::vec2(fovx, fovy),
                        glam::vec2(0.5, 0.5),
                    ),
                    image: Arc::new(image),
                };
                anyhow::Result::<SceneView>::Ok(view)
            }
        });

    Ok(iter.collect())
}

pub fn read_dataset_views(
    archive: ZipArchive<Cursor<ZipData>>,
    load_args: &LoadDatasetArgs,
) -> Result<DataStream<Dataset>> {
    // Assume nerf synthetic has a white background. Maybe add a custom json field to customize this
    // or something.
    let background = glam::Vec3::ONE;

    let load_args = load_args.clone();
    let train_handles = read_transforms_file(archive.clone(), "transforms_train.json", &load_args)?;

    let stream = try_fn_stream(|emitter| async move {
        let mut train_views = vec![];
        let mut eval_views = vec![];

        for (i, handle) in train_handles.into_iter().enumerate() {
            // I cannot wait for let chains.
            if let Some(eval_period) = load_args.eval_split_every {
                if i % eval_period == 0 {
                    eval_views.push(handle.await?);
                } else {
                    train_views.push(handle.await?);
                }
            } else {
                train_views.push(handle.await?);
            }

            emitter
                .emit(Dataset::from_views(
                    train_views.clone(),
                    eval_views.clone(),
                    background,
                ))
                .await;
        }

        // Not entirely sure yet if we want to report stats on both test
        // and eval. If so, just read  "transforms_test.json"
        let val_stream = read_transforms_file(archive, "transforms_val.json", &load_args);

        if let Ok(val_stream) = val_stream {
            for handle in val_stream {
                eval_views.push(handle.await?);
                emitter
                    .emit(Dataset::from_views(
                        train_views.clone(),
                        eval_views.clone(),
                        background,
                    ))
                    .await;
            }
        }

        Ok(())
    });

    Ok(Box::pin(stream))
}
