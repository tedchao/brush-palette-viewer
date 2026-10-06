use crate::{BrushVfs, VfsConstructError};
use core::fmt;
use rrfd::PickFileError;
use serde::Deserialize;
#[cfg(not(target_family = "wasm"))]
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use tokio::io::BufReader;

#[derive(Clone, Debug, Deserialize)]
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

impl fmt::Display for DataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PickFile => write!(f, "File"),
            Self::PickDirectory => write!(f, "Directory"),
            Self::Url(_) => write!(f, "URL"),
            Self::Path(_) => write!(f, "Path"),
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
    #[cfg(not(target_family = "wasm"))]
    #[error(transparent)]
    ReqwestError(#[from] reqwest::Error),
    #[error("WASM fetch error: {0}")]
    FetchError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl DataSource {
    pub async fn into_vfs(self) -> Result<Arc<BrushVfs>, DataSourceError> {
        match self {
            Self::PickFile => {
                let picked = rrfd::pick_file().await?;
                log::info!("Got file: {}", picked.name);
                let reader = BufReader::new(picked.reader);
                Ok(Arc::new(
                    BrushVfs::from_reader(reader, Some(picked.name)).await?,
                ))
            }
            Self::PickDirectory => {
                #[cfg(not(target_family = "wasm"))]
                {
                    let picked = rrfd::pick_directory().await?;
                    Ok(Arc::new(BrushVfs::from_path(&picked).await?))
                }
                #[cfg(target_family = "wasm")]
                {
                    let dir_handle = rrfd::wasm::pick_directory_handle().await?;
                    Ok(Arc::new(BrushVfs::from_directory_handle(dir_handle).await?))
                }
            }
            Self::Url(url) => Self::fetch_url(url).await,
            #[cfg(not(target_family = "wasm"))]
            Self::Path(path) => Ok(Arc::new(BrushVfs::from_path(Path::new(&path)).await?)),
            #[cfg(target_family = "wasm")]
            Self::Path(_) => {
                panic!("Cannot load from filesystem path on WASM");
            }
        }
    }

    /// For a URL source, fetch `file_name` from the same directory as the URL
    /// (eg. a .pply's .gswp sidecar). Returns Ok(None) for non-URL sources or
    /// when the server responds 404.
    pub async fn fetch_sibling(
        &self,
        file_name: &str,
    ) -> Result<Option<Vec<u8>>, DataSourceError> {
        let Self::Url(url) = self else {
            return Ok(None);
        };
        let url = resolve_url(url);
        // Drop any query/fragment, then replace the last path segment.
        let base = url.split(['?', '#']).next().unwrap_or(&url);
        let Some((dir, _)) = base.rsplit_once('/') else {
            return Ok(None);
        };
        fetch_bytes(&format!("{dir}/{file_name}")).await
    }

    /// The last path segment of a URL source, without query or fragment.
    pub fn url_file_name(&self) -> Option<&str> {
        let Self::Url(url) = self else {
            return None;
        };
        let base = url.split(['?', '#']).next()?;
        base.rsplit('/').next().filter(|s| !s.is_empty())
    }

    async fn fetch_url(url: String) -> Result<Arc<BrushVfs>, DataSourceError> {
        let url = resolve_url(&url);

        #[cfg(not(target_family = "wasm"))]
        {
            use tokio_stream::StreamExt;
            use tokio_util::io::StreamReader;

            let response = reqwest::get(&url).await?;

            // Try to get filename from Content-Disposition header, fall back to URL
            let name = response
                .headers()
                .get(reqwest::header::CONTENT_DISPOSITION)
                .and_then(|h| h.to_str().ok())
                .and_then(|s| {
                    // Parse "attachment; filename=\"name.ply\"" or "filename=name.ply"
                    s.split(';').find_map(|part| {
                        let part = part.trim();
                        if part.starts_with("filename=") {
                            let name = part.trim_start_matches("filename=");
                            Some(name.trim_matches('"').to_owned())
                        } else {
                            None
                        }
                    })
                })
                .or_else(|| url.rsplit('/').next().map(String::from));

            let stream = response.bytes_stream();
            let stream = stream.map(|b| b.map_err(|_e| std::io::ErrorKind::ConnectionAborted));
            let reader = StreamReader::new(stream);
            Ok(Arc::new(BrushVfs::from_reader(reader, name).await?))
        }

        #[cfg(target_family = "wasm")]
        {
            use tokio_util::compat::FuturesAsyncReadCompatExt;
            use wasm_streams::ReadableStream;

            let resp = wasm_fetch(&url).await?;

            if !resp.ok() {
                return Err(DataSourceError::FetchError(format!(
                    "HTTP error: {}",
                    resp.status()
                )));
            }

            // Try to get filename from Content-Disposition header, fall back to URL
            let name = resp
                .headers()
                .get("Content-Disposition")
                .ok()
                .flatten()
                .and_then(|s| {
                    // Parse "attachment; filename=\"name.ply\"" or "filename=name.ply"
                    s.split(';').find_map(|part| {
                        let part = part.trim();
                        if part.starts_with("filename=") {
                            let name = part.trim_start_matches("filename=");
                            Some(name.trim_matches('"').to_owned())
                        } else {
                            None
                        }
                    })
                })
                .or_else(|| url.rsplit('/').next().map(String::from));

            let body = resp
                .body()
                .ok_or_else(|| DataSourceError::FetchError("Response has no body".to_string()))?;

            let readable_stream = ReadableStream::from_raw(body);
            let async_read = readable_stream.into_async_read().compat();
            let async_read = BufReader::new(async_read);
            Ok(Arc::new(BrushVfs::from_reader(async_read, name).await?))
        }
    }
}

/// Normalize a user-provided URL: absolute paths are resolved against the page
/// origin on WASM, and a missing scheme defaults to https.
fn resolve_url(url: &str) -> String {
    if url.starts_with("https://") || url.starts_with("http://") {
        // fine, can use as is.
        url.to_owned()
    } else if url.starts_with('/') {
        #[cfg(target_family = "wasm")]
        {
            // Assume that this instead points to a GET request for the server.
            web_sys::window()
                .expect("No window object available")
                .location()
                .origin()
                .expect("Coultn't figure out origin")
                + url
        }
        // On non-wasm... not much we can do here, what server would we ask?
        #[cfg(not(target_family = "wasm"))]
        url.to_owned()
    } else {
        // Just try to add https:// and hope for the best. Eg. if someone specifies google.com/splat.ply.
        format!("https://{url}")
    }
}

/// Fetch a URL fully into memory. Returns Ok(None) on HTTP 404.
async fn fetch_bytes(url: &str) -> Result<Option<Vec<u8>>, DataSourceError> {
    #[cfg(not(target_family = "wasm"))]
    {
        let response = reqwest::get(url).await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let bytes = response.error_for_status()?.bytes().await?;
        Ok(Some(bytes.to_vec()))
    }

    #[cfg(target_family = "wasm")]
    {
        let resp = wasm_fetch(url).await?;
        if resp.status() == 404 {
            return Ok(None);
        }
        if !resp.ok() {
            return Err(DataSourceError::FetchError(format!(
                "HTTP error: {}",
                resp.status()
            )));
        }
        let read_err = |e| DataSourceError::FetchError(format!("Failed to read body: {e:?}"));
        let buffer = resp.array_buffer().map_err(read_err)?;
        let buffer = wasm_bindgen_futures::JsFuture::from(buffer)
            .await
            .map_err(read_err)?;
        Ok(Some(js_sys::Uint8Array::new(&buffer).to_vec()))
    }
}

#[cfg(target_family = "wasm")]
async fn wasm_fetch(url: &str) -> Result<web_sys::Response, DataSourceError> {
    use web_sys::wasm_bindgen::JsCast;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts).map_err(|e| {
        DataSourceError::FetchError(format!("Failed to create request: {:?}", e))
    })?;

    let window = web_sys::window()
        .ok_or_else(|| DataSourceError::FetchError("No window object available".to_string()))?;

    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| DataSourceError::FetchError(format!("Fetch failed: {:?}", e)))?;

    resp_value
        .dyn_into::<Response>()
        .map_err(|e| DataSourceError::FetchError(format!("Failed to cast to Response: {:?}", e)))
}
