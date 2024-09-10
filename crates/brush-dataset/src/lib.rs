pub mod colmap;
pub mod colmap_read_model;
pub mod nerf_synthetic;
pub mod scene_batch;

use std::path::{Path, PathBuf};

use anyhow::Result;
use brush_train::scene::SceneView;
use image::DynamicImage;
use ndarray::Array3;

pub(crate) fn normalized_path_string(path: &Path) -> String {
    Path::new(path)
        .components()
        .skip_while(|c| matches!(c, std::path::Component::CurDir))
        .collect::<PathBuf>()
        .to_str()
        .unwrap()
        .replace(std::path::MAIN_SEPARATOR, "/")
}

pub(crate) fn clamp_img_to_max_size(image: DynamicImage, max_size: u32) -> DynamicImage {
    if image.width() <= max_size && image.height() <= max_size {
        return image;
    }

    let aspect_ratio = image.width() as f32 / image.height() as f32;
    let (new_width, new_height) = if image.width() > image.height() {
        (max_size, (max_size as f32 / aspect_ratio) as u32)
    } else {
        ((max_size as f32 * aspect_ratio) as u32, max_size)
    };
    image.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
}

fn img_to_tensor(image: &image::DynamicImage) -> Result<Array3<f32>> {
    let im_data = image.to_rgba8().into_vec();
    let tensor = Array3::from_shape_vec(
        [image.height() as usize, image.width() as usize, 4],
        im_data,
    )?;
    Ok(tensor.to_owned().map(|&x| (x as f32) / 255.0))
}

pub fn read_dataset(
    zip_data: &[u8],
    max_frames: Option<usize>,
    max_resolution: Option<u32>,
) -> Result<Vec<SceneView>> {
    nerf_synthetic::read_dataset(zip_data, max_frames, max_resolution)
        .or_else(move |_| colmap::read_dataset(zip_data, max_frames, max_resolution))
}
