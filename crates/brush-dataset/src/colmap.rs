use std::{
    io::{Cursor, Read},
    sync::Arc,
};

use crate::{DataStream, Dataset, ZipData};
use anyhow::Result;
use async_fn_stream::try_fn_stream;
use async_std::task::{self, JoinHandle};
use brush_render::camera::{self, Camera};
use brush_train::scene::{Scene, SceneView};
use glam::Vec3;
use zip::ZipArchive;

use crate::colmap_read_model;

fn read_views(
    mut archive: ZipArchive<Cursor<ZipData>>,
    max_frames: Option<usize>,
    max_resolution: Option<u32>,
) -> Result<Vec<JoinHandle<Result<SceneView>>>> {
    let (bin, cam_path, img_path) = if archive.by_name("sparse/0/cameras.bin").is_ok() {
        (true, "sparse/0/cameras.bin", "sparse/0/images.bin")
    } else if archive.by_name("sparse/0/cameras.txt").is_ok() {
        (false, "sparse/0/cameras.txt", "sparse/0/images.txt")
    } else {
        anyhow::bail!("No COLMAP data found (either text or binary.");
    };

    let cam_model_data = {
        let mut cam_file = archive.by_name(cam_path)?;
        colmap_read_model::read_cameras(&mut cam_file, bin)?
    };

    let img_infos = {
        let img_file = archive.by_name(img_path)?;
        let mut buf_reader = std::io::BufReader::new(img_file);
        colmap_read_model::read_images(&mut buf_reader, bin)?
    };

    let img_info_list = img_infos
        .into_values()
        .take(max_frames.unwrap_or(usize::MAX))
        .collect::<Vec<_>>();

    let handles = img_info_list
        .iter()
        .map(move |img_info| {
            let mut archive = archive.clone();
            let cam = cam_model_data[&img_info.camera_id].clone();
            let translation = img_info.tvec;
            let quat = img_info.quat;
            let img_path = img_info.name.clone();

            task::spawn(async move {
                let focal = cam.focal();

                let fovx = camera::focal_to_fov(focal.x, cam.width as u32);
                let fovy = camera::focal_to_fov(focal.y, cam.height as u32);

                let center = cam.principal_point();
                let center_uv = center / glam::vec2(cam.width as f32, cam.height as f32);

                let image_data = archive.by_name(&format!("images/{img_path}"))?;
                let img_bytes = image_data.bytes().collect::<std::io::Result<Vec<u8>>>()?;
                let mut img = image::load_from_memory(&img_bytes)?;
                if let Some(max) = max_resolution {
                    img = crate::clamp_img_to_max_size(img, max);
                }
                // Convert w2c to c2w.
                let world_to_cam = glam::Affine3A::from_rotation_translation(quat, translation);

                let cam_to_world = world_to_cam.inverse();

                let (_, quat, translation) = cam_to_world.to_scale_rotation_translation();

                let view = SceneView {
                    name: img_path.to_string(),
                    camera: Camera::new(translation, quat, glam::vec2(fovx, fovy), center_uv),
                    image: Arc::new(img),
                };
                anyhow::Result::<SceneView>::Ok(view)
            })
        })
        .collect();

    Ok(handles)
}

pub(crate) fn read_dataset(
    archive: ZipArchive<Cursor<ZipData>>,
    max_frames: Option<usize>,
    max_resolution: Option<u32>,
) -> Result<DataStream> {
    let handles = read_views(archive, max_frames, max_resolution)?;

    let stream = try_fn_stream(|emitter| async move {
        let mut dataset = Dataset::empty();
        let mut views = vec![];

        for handle in handles {
            views.push(handle.await?);
            dataset.train = Scene::new(views.clone(), Vec3::ZERO);
            emitter.emit(dataset.clone()).await;
        }

        Ok(())
    });

    Ok(Box::pin(stream))
}
