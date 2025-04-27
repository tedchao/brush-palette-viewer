#[cfg(target_os = "android")]
pub mod android;

use std::path::PathBuf;
use tokio::io::AsyncRead;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PickFileError {
    #[error("No file was selected")]
    NoFileSelected,
    #[error("No directory was selected")]
    NoDirectorySelected,
    #[error("IO error while saving file.")]
    IoError(#[from] std::io::Error),
}

/// Pick a file and return the name & bytes of the file.
pub async fn pick_file() -> Result<impl AsyncRead + Unpin, PickFileError> {
    #[cfg(not(target_os = "android"))]
    {
        let file = rfd::AsyncFileDialog::new()
            .pick_file()
            .await
            .ok_or(PickFileError::NoFileSelected)?;

        #[cfg(target_family = "wasm")]
        {
            Ok(std::io::Cursor::new(file.read().await))
        }

        #[cfg(not(target_family = "wasm"))]
        {
            let file = tokio::fs::File::open(file.path()).await?;
            Ok(tokio::io::BufReader::new(file))
        }
    }

    #[cfg(target_os = "android")]
    {
        let file = android::pick_file().await?;
        tokio::io::BufReader::new(file)
    }
}

pub async fn pick_directory() -> Result<PathBuf, PickFileError> {
    #[cfg(all(not(target_os = "android"), not(target_family = "wasm")))]
    {
        let dir = rfd::AsyncFileDialog::new()
            .pick_folder()
            .await
            .ok_or(PickFileError::NoDirectorySelected)?;

        Ok(dir.path().to_path_buf())
    }

    #[cfg(any(target_os = "android", target_family = "wasm"))]
    {
        panic!("No folder picking on Android or wasm yet.")
    }
}

/// Saves data to a file and returns the filename the data was saved too.
///
/// Nb: Does not work on Android currently.
pub async fn save_file(default_name: &str, data: Vec<u8>) -> Result<(), PickFileError> {
    #[cfg(not(target_os = "android"))]
    {
        let file = rfd::AsyncFileDialog::new()
            .set_file_name(default_name)
            .save_file()
            .await
            .ok_or(PickFileError::NoFileSelected)?;

        #[cfg(not(target_family = "wasm"))]
        tokio::fs::write(file.path(), data).await?;

        #[cfg(target_family = "wasm")]
        file.write(&data).await?;

        Ok(())
    }

    #[cfg(target_os = "android")]
    {
        let _ = default_name;
        panic!("No saving on Android yet.")
    }
}
