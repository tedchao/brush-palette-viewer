use std::sync::Arc;

use eframe::egui_wgpu::WgpuConfiguration;

pub(crate) fn get_config() -> WgpuConfiguration {
    WgpuConfiguration {
        device_descriptor: Arc::new(|adapter| wgpu::DeviceDescriptor {
            label: Some("egui+burn wgpu device"),
            required_features: wgpu::Features::default(),
            required_limits: adapter.limits(),
            // cube already batches allocations.
            memory_hints: wgpu::MemoryHints::MemoryUsage,
        }),
        supported_backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    }
}
