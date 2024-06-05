use burn::tensor::backend::Backend;
use tracing::{info_span, span::EnteredSpan};

// Disaable a warning when tracy/sync_tracy are not enabled, it's ok.
#[allow(dead_code)]
pub struct SyncSpan<'a, B: Backend> {
    _span: EnteredSpan,
    device: &'a B::Device,
}

impl<'a, B: Backend> SyncSpan<'a, B> {
    pub fn new(name: &'static str, device: &'a B::Device) -> Self {
        let span = info_span!("sync", name).entered();
        Self {
            _span: span,
            device,
        }
    }
}

impl<'a, B: Backend> Drop for SyncSpan<'a, B> {
    fn drop(&mut self) {
        #[cfg(feature = "sync_tracy")]
        B::sync(self.device);
    }
}
