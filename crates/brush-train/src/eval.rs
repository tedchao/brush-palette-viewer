use brush_render::RenderAux;
use brush_render::{gaussian_splats::Splats, Backend};
use burn::tensor::ElementConversion;
use image::DynamicImage;
use rand::seq::IteratorRandom;

use crate::image::{image_to_tensor, tensor_into_image};
use crate::scene::Scene;

// TODO: Add ssim, maybe lpips.
pub struct EvalView {
    pub psnr: f32,
    pub rendered: DynamicImage,
    pub ground_truth: DynamicImage,
    pub aux: RenderAux,
}

pub struct EvalStats {
    pub samples: Vec<EvalView>,
}

pub async fn eval_stats<B: Backend>(
    splats: &Splats<B>,
    eval_scene: &Scene,
    num_frames: Option<usize>,
    device: &B::Device,
) -> EvalStats {
    let indices = if let Some(num) = num_frames {
        let mut rng = rand::thread_rng();
        (0..eval_scene.view_count()).choose_multiple(&mut rng, num)
    } else {
        (0..eval_scene.view_count()).collect()
    };

    let eval_views: Vec<_> = indices
        .into_iter()
        .map(|i| eval_scene.get_view(i).unwrap().clone())
        .collect();

    let mut ret = vec![];

    for view in eval_views {
        let ground_truth = view.image;
        let res = glam::uvec2(ground_truth.width(), ground_truth.height());

        let gt_tensor = image_to_tensor::<B>(&ground_truth, device);
        let (rendered, aux) = splats.render(&view.camera, res, eval_scene.background, false);

        let render_rgb = rendered.slice([0..res.y as usize, 0..res.x as usize, 0..3]);
        let mse = (render_rgb.clone() - gt_tensor).powf_scalar(2.0).mean();

        let psnr = mse.recip().log() * 10.0 / std::f32::consts::LN_10;
        let psnr = psnr.into_scalar_async().await.elem::<f32>();

        let eval_render = tensor_into_image(render_rgb.into_data_async().await);

        ret.push(EvalView {
            psnr,
            rendered: eval_render,
            ground_truth,
            aux,
        });
    }

    EvalStats { samples: ret }
}
