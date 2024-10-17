use crate::{viewer::ViewerContext, ViewerPanel};
use burn_jit::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};

pub(crate) struct StatsPanel {
    device: WgpuDevice,
}

impl StatsPanel {
    pub(crate) fn new(device: WgpuDevice) -> Self {
        Self { device }
    }
}

impl ViewerPanel for StatsPanel {
    fn title(&self) -> String {
        "Stats".to_owned()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &mut ViewerContext) {
        let client = WgpuRuntime::client(&self.device);
        let memory = client.memory_usage();
        ui.label(format!("Memory usage: {}", memory));
    }
}
