#[cfg(target_os = "android")]
pub mod android;

use anyhow::Result;

#[derive(Clone, Debug)]
pub struct PickedFile {
    pub data: Vec<u8>,
    pub file_name: String,
}

/// Pick a file and return the name & bytes of the file.
pub async fn pick_file() -> Result<PickedFile> {
    #[cfg(not(target_os = "android"))]
    {
        async move {
            use anyhow::Context;
            let file = rfd::AsyncFileDialog::new()
                .pick_file()
                .await
                .context("No file selected")?;
            let file_data = file.read().await;
            Ok(PickedFile {
                data: file_data,
                file_name: file.file_name(),
            })
        }
        .await
    }

    #[cfg(target_os = "android")]
    {
        android::pick_file().await
    }
}

/// Saves data to a file and returns the filename the data was saved too.
///
/// Nb: Does not work on Android currently.
pub async fn save_file(default_name: &str, data: Vec<u8>) -> Result<String> {
    #[cfg(not(target_os = "android"))]
    {
        async move {
            use anyhow::Context;
            let file = rfd::AsyncFileDialog::new()
                .set_file_name(default_name)
                .save_file()
                .await
                .context("No file selected")?;
            file.write(&data).await?;
            Ok(file.file_name())
        }
        .await
    }

    #[cfg(target_os = "android")]
    {
        unimplemented!("No saving on Android yet.")
    }
}
