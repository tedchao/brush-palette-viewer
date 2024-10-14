#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::single_range_in_vec_init)]

use egui_tiles::SimplificationOptions;
use viewer::ViewerContext;

mod burn_texture;
mod orbit_controls;
mod splat_import;
mod visualize;

mod panels;
mod train_loop;

pub mod viewer;
pub mod wgpu_config;

trait ViewerPane {
    fn title(&self) -> String;
    fn ui(&mut self, ui: &mut egui::Ui, controls: &mut ViewerContext) -> egui_tiles::UiResponse;
}

struct ViewerTree {
    context: ViewerContext,
}

type PaneType = Box<dyn ViewerPane>;

impl egui_tiles::Behavior<PaneType> for ViewerTree {
    fn tab_title_for_pane(&mut self, pane: &PaneType) -> egui::WidgetText {
        pane.title().into()
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut PaneType,
    ) -> egui_tiles::UiResponse {
        pane.ui(ui, &mut self.context)
    }

    /// What are the rules for simplifying the tree?
    fn simplification_options(&self) -> SimplificationOptions {
        SimplificationOptions {
            all_panes_must_have_tabs: true,
            ..Default::default()
        }
    }
}
