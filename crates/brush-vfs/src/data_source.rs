use crate::{BrushVfs, VfsConstructError};
use rrfd::PickFileError;
use std::{path::Path, str::FromStr};
use tokio_stream::StreamExt;
use tokio_util::io::StreamReader;

#[derive(Clone, Debug)]
pub enum DataSource {
    PickFile,
    PickDirectory,
    Url(String),
    Path(String),
}

// Implement FromStr to allow Clap to parse string arguments into DataSource
impl FromStr for DataSource {
    type Err = String; // TODO: Really is a never type but meh.

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            s if s.starts_with("http://") || s.starts_with("https://") => {
                Ok(Self::Url(s.to_owned()))
            }
            // This path might not exist but that's ok, rather find that out later.
            s => Ok(Self::Path(s.to_owned())),
        }
    }
}

use thiserror::Error;
#[derive(Debug, Error)]
pub enum DataSourceError {
    #[error(transparent)]
    FilePicking(#[from] PickFileError),
    #[error(transparent)]
    VfsError(#[from] VfsConstructError),
    #[error(transparent)]
    ReqwestError(#[from] reqwest::Error),
}

impl DataSource {
    pub async fn into_vfs(self) -> Result<BrushVfs, DataSourceError> {
        match self {
            Self::PickFile => {
                let reader = rrfd::pick_file().await?;
                Ok(BrushVfs::from_reader(reader).await?)
            }
            Self::PickDirectory => {
                let picked = rrfd::pick_directory().await?;
                Ok(BrushVfs::from_path(&picked).await?)
            }
            Self::Url(url) => {
                let mut url = url.clone();
                url = url.replace("https://", "");

                if url.starts_with("https://") || url.starts_with("http://") {
                    // fine, can use as is.
                } else if url.starts_with('/') {
                    #[cfg(target_family = "wasm")]
                    {
                        // Assume that this instead points to a GET request for the server.
                        url = web_sys::window()
                            .expect("No window object available")
                            .location()
                            .origin()
                            .expect("Coultn't figure out origin")
                            + &url;
                    }
                    // On non-wasm... not much we can do here, what server would we ask?
                } else {
                    // Just try to add https:// and hope for the best. Eg. if someone specifies google.com/splat.ply.
                    url = format!("https://{url}");
                }

                let response = reqwest::get(url).await?.bytes_stream();

                let response =
                    response.map(|b| b.map_err(|_e| std::io::ErrorKind::ConnectionAborted));
                let reader = StreamReader::new(response);
                Ok(BrushVfs::from_reader(reader).await?)
            }
            Self::Path(path) => Ok(BrushVfs::from_path(Path::new(&path)).await?),
        }
    }
}
