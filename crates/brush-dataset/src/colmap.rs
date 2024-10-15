use std::io::{Cursor, Read};

use crate::{Dataset, ZipData};
use anyhow::Result;
use async_fn_stream::try_fn_stream;
use brush_render::camera::{self, Camera};
use brush_train::scene::{Scene, SceneView};
use futures_lite::Stream;
use zip::ZipArchive;

use crate::colmap_read_model;

#[allow(unused_variables)]
pub fn read_dataset(
    mut archive: ZipArchive<Cursor<ZipData>>,
    max_frames: Option<usize>,
    max_resolution: Option<u32>,
) -> Result<impl Stream<Item = Result<Dataset>>> {
    let files: Vec<_> = archive.file_names().collect();

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

    // Colmap data should be real images, with no transparency, which means background should't
    // matter. Most viewers assume that means a black background.
    let background = glam::vec3(0.0, 0.0, 0.0);

    let stream = try_fn_stream(|emitter| async move {
        let mut train_scene = Scene::new(vec![], background);

        let img_infos = {
            let img_file = archive.by_name(img_path)?;
            let mut buf_reader = std::io::BufReader::new(img_file);
            colmap_read_model::read_images(&mut buf_reader, bin)?
        };

        let mut img_info_list = img_infos
            .iter()
            .map(|(id, info)| (*id, info))
            .collect::<Vec<_>>();

        // The ids aren't guaranteed to be meaningful but seem to usually correspond
        // to capture order.
        img_info_list.sort_by_key(|(id, info)| *id);

        for chunk in img_info_list.chunks(16) {
            let mut receivers = Vec::new();

            for (_, img_info) in chunk.iter() {
                let (sender, receiver) = async_channel::bounded(1);
                receivers.push(receiver);

                let mut archive = archive.clone();
                let cam = cam_model_data[&img_info.camera_id].clone();
                let translation = img_info.tvec;
                let quat = img_info.quat;
                let img_path = img_info.name.clone();

                let load_fut = move || async move {
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
                    let world_to_cam = glam::Mat4::from_rotation_translation(quat, translation);
                    let cam_to_world = world_to_cam.inverse();

                    let (_, quat, translation) = cam_to_world.to_scale_rotation_translation();

                    let view = SceneView {
                        name: img_path.to_string(),
                        camera: Camera::new(translation, quat, glam::vec2(fovx, fovy), center_uv),
                        image: img,
                    };
                    anyhow::Result::<SceneView>::Ok(view)
                };

                let send_fut = || async move {
                    let res = load_fut().await;
                    sender.send(res).await.expect("Failed to send signal");
                };

                #[cfg(not(target_family = "wasm"))]
                std::thread::spawn(|| futures_lite::future::block_on(send_fut()));

                #[cfg(target_family = "wasm")]
                wasm_bindgen_futures::spawn_local(send_fut)
            }

            for receiver in receivers {
                let res = receiver.recv().await.unwrap();
                train_scene.add_view(res?);

                emitter
                    .emit(Dataset::new(train_scene.clone(), None, None))
                    .await;

                if let Some(max) = max_frames {
                    if train_scene.views().len() >= max {
                        break;
                    }
                }
            }
        }

        Ok(())
    });

    Ok(stream)
}
