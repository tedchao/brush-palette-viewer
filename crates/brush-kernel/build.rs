fn main() -> Result<(), brush_wgsl::GenError> {
    brush_wgsl::build_modules(
        &["src/shaders/wg.wgsl"],
        &[],
        "src/shaders",
        "src/shaders/mod.rs",
    )
}
