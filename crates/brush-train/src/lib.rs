use burn_wgpu::{AutoGraphicsApi, RuntimeOptions};

pub mod eval;
pub mod ssim;
pub mod train;

pub mod image;
pub mod scene;

pub async fn create_wgpu_setup() -> burn_wgpu::WgpuSetup {
    burn_wgpu::init_setup_async::<AutoGraphicsApi>(
        &burn_wgpu::WgpuDevice::DefaultDevice,
        RuntimeOptions {
            tasks_max: 64,
            memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
        },
    )
    .await
}
