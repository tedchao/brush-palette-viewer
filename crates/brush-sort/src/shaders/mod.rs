pub(crate) mod sort_count {
    pub(crate) const WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
   pub(crate) const WG: u32 = 256;
   pub(crate) const BIN_COUNT: u32 = 16;
   pub(crate) const ELEMENTS_PER_THREAD: u32 = 4;
   pub(crate) const BLOCK_SIZE: u32 = 1024;
   pub(crate) const VERTICAL_GROUPS: u32 = 8;
#[repr(C, align(16))]
#[derive(bytemuck::NoUninit, Debug, PartialEq, Clone, Copy)]
pub(crate) struct Uniforms {
    shift: u32,
}
pub(crate) fn create_shader_source() -> String {
include_str!("src/shaders/sort_count.wgsl").to_owned()
}
}
pub(crate) mod sort_reduce {
    pub(crate) const WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
   pub(crate) const WG: u32 = 256;
   pub(crate) const BIN_COUNT: u32 = 16;
   pub(crate) const ELEMENTS_PER_THREAD: u32 = 4;
   pub(crate) const BLOCK_SIZE: u32 = 1024;
pub(crate) fn create_shader_source() -> String {
include_str!("src/shaders/sort_reduce.wgsl").to_owned()
}
}
pub(crate) mod sort_scan_add {
    pub(crate) const WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
   pub(crate) const WG: u32 = 256;
   pub(crate) const BIN_COUNT: u32 = 16;
   pub(crate) const ELEMENTS_PER_THREAD: u32 = 4;
   pub(crate) const BLOCK_SIZE: u32 = 1024;
pub(crate) fn create_shader_source() -> String {
include_str!("src/shaders/sort_scan_add.wgsl").to_owned()
}
}
pub(crate) mod sort_scan {
    pub(crate) const WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
   pub(crate) const WG: u32 = 256;
   pub(crate) const BIN_COUNT: u32 = 16;
   pub(crate) const ELEMENTS_PER_THREAD: u32 = 4;
   pub(crate) const BLOCK_SIZE: u32 = 1024;
pub(crate) fn create_shader_source() -> String {
include_str!("src/shaders/sort_scan.wgsl").to_owned()
}
}
pub(crate) mod sort_scatter {
    pub(crate) const WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
   pub(crate) const WG: u32 = 256;
   pub(crate) const BITS_PER_PASS: u32 = 4;
   pub(crate) const BIN_COUNT: u32 = 16;
   pub(crate) const ELEMENTS_PER_THREAD: u32 = 4;
   pub(crate) const BLOCK_SIZE: u32 = 1024;
   pub(crate) const VERTICAL_GROUPS: u32 = 8;
#[repr(C, align(16))]
#[derive(bytemuck::NoUninit, Debug, PartialEq, Clone, Copy)]
pub(crate) struct Uniforms {
    shift: u32,
}
pub(crate) fn create_shader_source() -> String {
include_str!("src/shaders/sort_scatter.wgsl").to_owned()
}
}
