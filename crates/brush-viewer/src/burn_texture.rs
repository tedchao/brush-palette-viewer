use std::sync::Arc;

use brush_render::Backend;
use burn::tensor::Tensor;
use burn_wgpu::{JitTensor, WgpuRuntime};
use eframe::egui_wgpu::Renderer;
use egui::epaint::mutex::RwLock as EguiRwLock;
use egui::TextureId;
use wgpu::ImageDataLayout;

fn copy_buffer_to_texture(
    img: JitTensor<WgpuRuntime, f32>,
    texture: &wgpu::Texture,
    encoder: &mut wgpu::CommandEncoder,
) {
    let [height, width, _] = img.shape.dims();

    img.client.sync(burn::tensor::backend::SyncType::Flush);

    let img_res = img.client.get_resource(img.handle.clone().binding());

    // Put compute passes in encoder before copying the buffer.
    let bytes_per_row = Some(4 * width as u32);

    // Now copy the buffer to the texture.
    encoder.copy_buffer_to_texture(
        wgpu::ImageCopyBuffer {
            buffer: img_res.resource().buffer.as_ref(),
            layout: ImageDataLayout {
                offset: img_res.resource().offset(),
                bytes_per_row,
                rows_per_image: None,
            },
        },
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        },
    );
}

pub struct BurnTexture {
    pub texture: wgpu::Texture,
    pub id: TextureId,
}

fn create_texture(size: glam::UVec2, device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Splat backbuffer"),
        size: wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
    })
}

impl BurnTexture {
    pub fn new<B: Backend>(
        tensor: Tensor<B, 3>,
        device: &wgpu::Device,
        renderer: Arc<EguiRwLock<Renderer>>,
    ) -> Self {
        let [h, w, _] = tensor.shape().dims();
        let texture = create_texture(glam::uvec2(w as u32, h as u32), device);
        let view = texture.create_view(&Default::default());
        let id = renderer
            .write()
            .register_native_texture(device, &view, wgpu::FilterMode::Linear);
        Self { texture, id }
    }

    pub fn update_texture<B: Backend<FloatTensorPrimitive = JitTensor<WgpuRuntime, f32>>>(
        &mut self,
        tensor: Tensor<B, 3>,
        device: &wgpu::Device,
        renderer: Arc<EguiRwLock<Renderer>>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let [h, w, _] = tensor.shape().dims();
        let size = glam::uvec2(w as u32, h as u32);

        let dirty = self.texture.width() != size.x || self.texture.height() != size.y;

        if dirty {
            self.texture = create_texture(glam::uvec2(w as u32, h as u32), device);

            renderer.write().update_egui_texture_from_wgpu_texture(
                device,
                &self.texture.create_view(&Default::default()),
                wgpu::FilterMode::Linear,
                self.id,
            )
        }

        copy_buffer_to_texture(tensor.into_primitive().tensor(), &self.texture, encoder);
    }
}
