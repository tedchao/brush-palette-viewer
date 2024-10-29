pub mod android;

#[allow(unused)]
use anyhow::Context;
use anyhow::Result;

pub enum FileHandle {
    #[cfg(not(target_os = "android"))]
    Rfd(rfd::FileHandle),
    Android(android::PickedFile),
}

impl FileHandle {
    pub fn file_name(&self) -> String {
        match self {
            #[cfg(not(target_os = "android"))]
            FileHandle::Rfd(file_handle) => file_handle.file_name(),
            FileHandle::Android(picked_file) => picked_file.file_name.clone(),
        }
    }

    pub async fn write(&self, data: &[u8]) -> std::io::Result<()> {
        match self {
            #[cfg(not(target_os = "android"))]
            FileHandle::Rfd(file_handle) => file_handle.write(data).await,
            FileHandle::Android(_) => {
                let _ = data;
                unimplemented!("No saving on Android yet.")
            }
        }
    }

    pub async fn read(&self) -> Vec<u8> {
        match self {
            #[cfg(not(target_os = "android"))]
            FileHandle::Rfd(file_handle) => file_handle.read().await,
            FileHandle::Android(picked_file) => picked_file.data.clone(),
        }
    }
}

/// Pick a file and return the name & bytes of the file.
pub async fn pick_file() -> Result<FileHandle> {
    #[cfg(not(target_os = "android"))]
    {
        let file = rfd::AsyncFileDialog::new()
            .pick_file()
            .await
            .context("No file selected")?;

        Ok(FileHandle::Rfd(file))
    }
    #[cfg(target_os = "android")]
    {
        android::pick_file().await.map(FileHandle::Android)
    }
}

/// Saves data to a file and returns the filename the data was saved too.
///
/// Nb: Does not work on Android currently.
pub async fn save_file(default_name: &str) -> Result<FileHandle> {
    #[cfg(not(target_os = "android"))]
    {
        let file = rfd::AsyncFileDialog::new()
            .set_file_name(default_name)
            .save_file()
            .await
            .context("No file selected")?;
        Ok(FileHandle::Rfd(file))
    }
    #[cfg(target_os = "android")]
    {
        let _ = default_name;
        unimplemented!("No saving on Android yet.")
    }
}
