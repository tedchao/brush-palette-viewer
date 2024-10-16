pub mod eval;
pub mod ssim;
pub mod train;

pub mod image;
pub mod scene;

use std::future::Future;

use async_std::task::JoinHandle;

/// Spawn a future (on the async executor on native, as a JS promise on web).
pub fn spawn_future<T: Send + 'static>(
    future: impl Future<Output = T> + Send + 'static,
) -> JoinHandle<T> {
    #[cfg(not(target_family = "wasm"))]
    {
        // multithreading on native is nice but is in fact required - egui blocks the main thread so all spawn_local
        // futures don't actually seem to run.
        async_std::task::spawn(future)
    }
    #[cfg(target_family = "wasm")]
    {
        async_std::task::spawn_local(future)
    }
}
