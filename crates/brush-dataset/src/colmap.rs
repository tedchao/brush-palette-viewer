use brush_render::camera::{self};
use brush_train::scene::SceneView;
use std::io::Cursor;

use crate::colmap_read_model;

#[allow(unused_variables)]
pub fn read_dataset(
    zip_data: &[u8],
    max_frames: Option<usize>,
    max_resolution: Option<u32>,
) -> anyhow::Result<Vec<SceneView>> {
    let mut archive = zip::ZipArchive::new(Cursor::new(zip_data))?;
    let files: Vec<_> = dbg!(archive.file_names().collect());

    let (bin, cam_path, img_path) = if archive.by_name("sparse/0/cameras.bin").is_ok() {
        (true, "sparse/0/cameras.bin", "sparse/0/images.bin")
    } else if archive.by_name("sparse/0/cameras.txt").is_ok() {
        (false, "sparse/0/cameras.txt", "sparse/0/images.txt")
    } else {
        anyhow::bail!("No COLMAP data found (either text or binary.");
    };

    let cam_model_data = {
        let mut cameras_bin = archive.by_name(cam_path)?;
        colmap_read_model::read_cameras_binary(&mut cameras_bin)?
    };

    let img_data = {
        let mut cameras_bin = archive.by_name(img_path)?;
        colmap_read_model::read_images_binary(&mut cameras_bin)?
    };

    let views = vec![];

    for (_, img) in img_data {
        let cam = &cam_model_data[&img.camera_id];

        // TODO: Handle off-center calibration.
        let focal = cam.focal();
        let fovx = camera::focal_to_fov(focal.x, cam.width as u32);
        let fovy = camera::focal_to_fov(focal.y, cam.height as u32);

        let img_path = img.name;
        println!("{img_path}");

        // let view = SceneView {
        //     camera: Camera::new(img.tvec, img.qvec, fovx, fovy),
        //     image: img_path,
        // };
        // let view = SceneView {
        //     camera: Camera::new(img.tvec, img.qvec, fovx, fovy),
        // }
        // img.camera_id
    }

    Ok(views)
}
