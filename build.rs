use wgsl_bindgen::{GlamWgslTypeMap, WgslBindgenOptionBuilder, WgslTypeSerializeStrategy};

fn main() {
    let bindgen = WgslBindgenOptionBuilder::default()
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .emit_rerun_if_change(true)
        .short_constructor(0)
        .type_map(GlamWgslTypeMap)
        .workspace_root("shaders")
        .add_entry_point("shaders/project_forward.wgsl")
        .output("src/splat_render/gen/bindings.rs")
        .build()
        .unwrap();
    bindgen.generate().unwrap();
}
