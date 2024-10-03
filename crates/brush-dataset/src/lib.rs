pub mod colmap;
pub mod colmap_read_model;
pub mod nerf_synthetic;
pub mod scene_batch;

use anyhow::Result;
use brush_train::scene::Scene;
use image::DynamicImage;
use std::path::{Path, PathBuf};

pub struct Dataset {
    pub train: Scene,
    pub test: Option<Scene>,
    pub eval: Option<Scene>,
}

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

pub fn read_dataset(
    zip_data: &[u8],
    max_frames: Option<usize>,
    max_resolution: Option<u32>,
) -> Result<Dataset> {
    nerf_synthetic::read_dataset(zip_data, max_frames, max_resolution)
        .or_else(move |_| colmap::read_dataset(zip_data, max_frames, max_resolution))
}
