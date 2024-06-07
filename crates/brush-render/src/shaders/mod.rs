pub(crate) mod project_forward {
    pub(crate) const WORKGROUP_SIZE: [u32; 3] = [512, 1, 1];
   pub(crate) const SPLATS_PER_GROUP: u32 = 512;
   pub(crate) const TILE_WIDTH: u32 = 16;
   pub(crate) const COV_BLUR: f32 = 0.3;
#[repr(C, align(16))]
#[derive(bytemuck::NoUninit, Debug, PartialEq, Clone, Copy)]
pub(crate) struct Uniforms {
    clip_thresh: f32,
    sh_degree: u32,
}
pub(crate) fn create_shader_source() -> String {
include_str!("src/shaders/project_forward.wgsl").to_owned()
}
}
pub(crate) mod map_gaussian_to_intersects {
    pub(crate) const WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
   pub(crate) const TILE_WIDTH: u32 = 16;
#[repr(C, align(16))]
#[derive(bytemuck::NoUninit, Debug, PartialEq, Clone, Copy)]
pub(crate) struct Uniforms {
}
pub(crate) fn create_shader_source() -> String {
include_str!("src/shaders/map_gaussian_to_intersects.wgsl").to_owned()
}
}
pub(crate) mod get_tile_bin_edges {
    pub(crate) const WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
   pub(crate) const VERTICAL_GROUPS: u32 = 8;
   pub(crate) const THREAD_COUNT: u32 = 256;
pub(crate) fn create_shader_source() -> String {
include_str!("src/shaders/get_tile_bin_edges.wgsl").to_owned()
}
}
pub(crate) mod rasterize {
    pub(crate) const WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];
   pub(crate) const TILE_WIDTH: u32 = 16;
   pub(crate) const TILE_SIZE: u32 = 256;
#[repr(C, align(16))]
#[derive(bytemuck::NoUninit, Debug, PartialEq, Clone, Copy)]
pub(crate) struct Uniforms {
}
pub(crate) fn create_shader_source() -> String {
include_str!("src/shaders/rasterize.wgsl").to_owned()
}
}
pub(crate) mod rasterize_backwards {
    pub(crate) const WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];
   pub(crate) const TILE_WIDTH: u32 = 16;
   pub(crate) const TILE_SIZE: u32 = 256;
#[repr(C, align(16))]
#[derive(bytemuck::NoUninit, Debug, PartialEq, Clone, Copy)]
pub(crate) struct Uniforms {
}
pub(crate) fn create_shader_source() -> String {
include_str!("src/shaders/rasterize_backwards.wgsl").to_owned()
}
}
pub(crate) mod project_backwards {
    pub(crate) const WORKGROUP_SIZE: [u32; 3] = [512, 1, 1];
   pub(crate) const SPLATS_PER_GROUP: u32 = 512;
#[repr(C, align(16))]
#[derive(bytemuck::NoUninit, Debug, PartialEq, Clone, Copy)]
pub(crate) struct Uniforms {
    sh_degree: u32,
}
pub(crate) fn create_shader_source() -> String {
include_str!("src/shaders/project_backwards.wgsl").to_owned()
}
}
