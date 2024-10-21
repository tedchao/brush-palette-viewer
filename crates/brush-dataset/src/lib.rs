pub mod colmap;
pub mod colmap_read_model;
pub mod nerf_synthetic;
pub mod scene_batch;
pub mod splat_export;
pub mod splat_import;

use anyhow::Result;
use async_std::stream::Stream;
use brush_render::{gaussian_splats::Splats, Backend};
use brush_train::scene::{Scene, SceneView};
use glam::Vec3;
use image::DynamicImage;
use splat_import::load_splat_from_ply;
use std::{
    io::{Cursor, Read},
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
    pub eval: Option<Scene>,
}

impl Dataset {
    pub fn empty() -> Self {
        Dataset {
            train: Scene::new(vec![], Vec3::ZERO),
            eval: None,
        }
    }

    pub fn from_views(
        train_views: Vec<SceneView>,
        eval_views: Vec<SceneView>,
        background: glam::Vec3,
    ) -> Self {
        Dataset {
            train: Scene::new(train_views, background),
            eval: if eval_views.is_empty() {
                None
            } else {
                Some(Scene::new(eval_views.clone(), background))
            },
        }
    }
}

#[derive(Clone)]
pub struct LoadDatasetArgs {
    pub max_frames: Option<usize>,
    pub max_resolution: Option<u32>,
    pub eval_split_every: Option<usize>,
}

#[derive(Clone)]
pub struct LoadInitArgs {
    pub sh_degree: u32,
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

type DataStream = Pin<Box<dyn Stream<Item = Result<Dataset>> + Send + 'static>>;

pub fn read_dataset_views(
    archive: ZipArchive<Cursor<ZipData>>,
    load_args: &LoadDatasetArgs,
) -> Result<DataStream> {
    let nerf = nerf_synthetic::read_dataset_views(archive.clone(), load_args);
    if let Ok(stream) = nerf {
        return Ok(stream);
    }
    let colmap = colmap::read_dataset_views(archive.clone(), load_args);
    if let Ok(stream) = colmap {
        return Ok(stream);
    }
    anyhow::bail!("Couldn't parse dataset as any format. Only some formats are supported.")
}

type SplatStream<B> = Pin<Box<dyn Stream<Item = Result<Splats<B>>> + Send + 'static>>;

fn read_init_ply<B: Backend>(
    mut archive: ZipArchive<Cursor<ZipData>>,
    device: &B::Device,
) -> Result<SplatStream<B>> {
    let data = archive
        .by_name("init.ply")
        .map(|f| f.bytes().collect::<std::io::Result<Vec<u8>>>())?;

    let Ok(data) = data else {
        anyhow::bail!("Couldn't load data")
    };

    let splat_stream = load_splat_from_ply::<B>(data, device.clone());
    Ok(Box::pin(splat_stream))
}

pub fn read_dataset_init<B: Backend>(
    archive: ZipArchive<Cursor<ZipData>>,
    device: &B::Device,
    load_args: &LoadInitArgs,
) -> Result<SplatStream<B>> {
    // If there's an init.ply definitey use that. Nb:
    // this ignores the specified number of SH channels atm.
    if let Ok(stream) = read_init_ply(archive.clone(), device) {
        return Ok(stream);
    }

    let colmap = colmap::read_init_splat(archive.clone(), device, load_args);
    if let Ok(stream) = colmap {
        return Ok(stream);
    }
    anyhow::bail!("No splat initialization possible.")
}
