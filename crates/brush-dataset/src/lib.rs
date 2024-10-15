pub mod colmap;
pub mod colmap_read_model;
pub mod nerf_synthetic;
pub mod scene_batch;

use anyhow::Result;
use brush_train::scene::Scene;
use futures_lite::{Stream, StreamExt};
use glam::Vec3;
use image::DynamicImage;
use std::{
    io::Cursor,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};
use zip::ZipArchive;

#[derive(Clone)]
pub struct ZipData {
    data: Arc<Vec<u8>>,
}

impl AsRef<[u8]> for ZipData {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

impl ZipData {
    pub fn open_for_read(&self) -> Cursor<ZipData> {
        Cursor::new(self.clone())
    }
}

impl From<Vec<u8>> for ZipData {
    fn from(value: Vec<u8>) -> Self {
        Self {
            data: Arc::new(value),
        }
    }
}

#[derive(Clone)]
pub struct Dataset {
    pub train: Scene,
    #[allow(unused)]
    pub test: Option<Scene>,
    pub eval: Option<Scene>,
}

impl Dataset {
    pub fn empty() -> Self {
        Dataset {
            train: Scene::new(vec![], Vec3::ZERO),
            test: None,
            eval: None,
        }
    }

    fn new(train: Scene, test: Option<Scene>, eval: Option<Scene>) -> Self {
        Self { train, test, eval }
    }
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
    archive: ZipArchive<Cursor<ZipData>>,
    max_frames: Option<usize>,
    max_resolution: Option<u32>,
) -> Result<Pin<Box<dyn Stream<Item = Result<Dataset>> + 'static>>> {
    let nerf = nerf_synthetic::read_dataset(archive.clone(), max_frames, max_resolution);
    if let Ok(stream) = nerf {
        return Ok(stream.boxed_local::<'static>());
    }
    let colmap = colmap::read_dataset(archive.clone(), max_frames, max_resolution);
    if let Ok(stream) = colmap {
        return Ok(stream.boxed_local::<'static>());
    }

    anyhow::bail!("Couldn't parse dataset as any format. Only some formats are supported.")
}
