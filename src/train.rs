use std::time;

use anyhow::Result;
use burn::lr_scheduler::LrScheduler;
use burn::tensor::ElementConversion;
use burn::{
    config::Config,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::Tensor,
};
use ndarray::{Array, Array3};
use rand::{rngs::StdRng, SeedableRng};

use crate::gaussian_splats::{self, SplatsTrainState};
use crate::splat_render::{self, Backend};
use crate::{
    dataset_readers,
    gaussian_splats::{Splats, SplatsConfig},
    utils,
};
use rand::seq::SliceRandom;

#[derive(Config)]
pub(crate) struct TrainConfig {
    #[config(default = 42)]
    pub(crate) seed: u64,
    #[config(default = 50000)]
    pub(crate) train_steps: u32,
    #[config(default = false)]
    pub(crate) random_bck_color: bool,
    #[config(default = 1e-2)]
    pub lr: f64,
    #[config(default = 1e-3)]
    pub min_lr: f64,
    #[config(default = 25)]
    pub visualize_every: u32,
    #[config(default = 3000)]
    pub init_points: usize,
    #[config(default = 2.0)]
    pub init_aabb: f32,
    pub scene_path: String,
}

struct TrainStats<B: Backend> {
    pred_image: Tensor<B, 3>,
    loss: Tensor<B, 1>,
    psnr: Tensor<B, 1>,
    aux: crate::splat_render::Aux<B>,
    gt_image: Array3<f32>,
}

// Training loop.
pub(crate) fn train<B: splat_render::AutodiffBackend>(
    config: &TrainConfig,
    device: &B::Device,
) -> Result<()>
where
    B::InnerBackend: Backend,
{
    #[cfg(feature = "rerun")]
    let rec = rerun::RecordingStreamBuilder::new("visualize training").spawn()?;

    println!("Loading dataset.");
    let scene = dataset_readers::read_scene(&config.scene_path, None, false);
    let config_optimizer = AdamConfig::new();
    let mut optim = config_optimizer.init::<B, Splats<B>>();

    B::seed(config.seed);
    let mut rng = StdRng::from_seed([10; 32]);

    #[cfg(feature = "rerun")]
    scene.visualize(&rec)?;

    let mut splats: Splats<B> =
        SplatsConfig::new(config.init_points, config.init_aabb, 0, 1.0).build(device);

    let mut state: SplatsTrainState<B> = SplatsTrainState::new(config.init_points, device);

    // TODO: Original implementation has differennt learning rates for almost all params.
    let mut scheduler = burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig::new(
        config.lr,
        config.train_steps as usize,
    )
    .with_min_lr(config.min_lr)
    .init();

    println!("Start training.");

    // By default use 8 second window with 16 accumulators
    for iter in 0..config.train_steps {
        let lr = LrScheduler::<B>::step(&mut scheduler);

        let start_time = time::Instant::now();

        // Get a random camera
        // TODO: Reproduciable RNG thingies.
        // TODO: If there is no camera, maybe just bail?
        let viewpoint = scene
            .train_data
            .choose(&mut rng)
            .expect("Dataset should have at least 1 camera.");

        let camera = &viewpoint.camera;

        let background_color = if config.random_bck_color {
            glam::vec3(rand::random(), rand::random(), rand::random())
        } else {
            scene.default_bg_color
        };

        let view_image = &viewpoint.view.image;
        let img_size = glam::uvec2(view_image.shape()[1] as u32, view_image.shape()[0] as u32);

        let (pred_img, aux) = splats.render(camera, img_size, background_color);
        let dims = pred_img.dims();

        let rgb_img = pred_img.clone().slice([0..dims[0], 0..dims[1], 0..3]);
        let gt_image = utils::ndarray_to_burn(view_image.clone(), device);
        // TODO: Burn should be able to slice open ranges.
        let rgb = gt_image.clone().slice([0..dims[0], 0..dims[1], 0..3]);
        let alpha = gt_image.clone().slice([0..dims[0], 0..dims[1], 3..4]);
        let gt_image = rgb * alpha.clone()
            + (-alpha + 1.0)
                * Tensor::from_floats(
                    [
                        background_color[0],
                        background_color[1],
                        background_color[2],
                    ],
                    device,
                )
                .unsqueeze::<3>();

        let mse = (rgb_img - gt_image).powf_scalar(2.0).mean();
        let psnr = mse.clone().recip().log() * 10.0 / std::f32::consts::LN_10;

        let mut grads = mse.backward();
        // get viewspace grads & remove it from the struct.
        let xys_grad = Tensor::from_inner(splats.xys_dummy.grad_remove(&mut grads).unwrap());

        // Gradients linked to each parameter of the model.
        let grads = GradientsParams::from_grads(grads, &splats);
        splats = optim.step(lr, splats, grads);

        state.update_stats(&aux, xys_grad);

        let stats = TrainStats {
            aux,
            gt_image: viewpoint.view.image.clone(),
            loss: mse,
            psnr,
            pred_image: pred_img,
        };

        if iter % 100 == 0 {
            splats = gaussian_splats::prune_invisible_point(splats, &mut state, 0.05);
            // Need to reset the optimizer - otherwise it tries to add the gradients from the old points.
            // Is there a way to keep the momentum instead?
            optim = config_optimizer.init::<B, Splats<B>>();
        }

        #[cfg(not(feature = "rerun"))]
        drop(stats);

        #[cfg(feature = "rerun")]
        {
            rec.set_time_sequence("iterations", iter);
            rec.log(
                "losses/main",
                &rerun::Scalar::new(utils::burn_to_scalar(stats.loss).elem::<f64>()),
            )?;

            rec.log(
                "stats/PSNR",
                &rerun::Scalar::new(utils::burn_to_scalar(stats.psnr).elem::<f64>()),
            )?;

            rec.log("lr/current", &rerun::Scalar::new(lr))?;

            rec.log(
                "performance/step_ms",
                &rerun::Scalar::new((time::Instant::now() - start_time).as_secs_f64() * 1000.0)
                    .clone(),
            )?;

            rec.log(
                "tiling/num intersects",
                &rerun::Scalar::new(utils::burn_to_scalar(stats.aux.num_intersects).elem::<f64>())
                    .clone(),
            )?;

            rec.log(
                "tiling/num visible",
                &rerun::Scalar::new(utils::burn_to_scalar(stats.aux.num_visible).elem::<f64>())
                    .clone(),
            )?;

            if iter % config.visualize_every == 0 {
                rec.log(
                    "images/ground truth",
                    &rerun::Image::try_from(stats.gt_image).unwrap(),
                )?;

                let tile_bins = stats.aux.tile_bins;
                let tile_size = tile_bins.dims();
                let tile_depth = tile_bins
                    .clone()
                    .slice([0..tile_size[0], 0..tile_size[1], 1..2])
                    - tile_bins
                        .clone()
                        .slice([0..tile_size[0], 0..tile_size[1], 0..1]);

                let tile_depth = Array::from_shape_vec(
                    tile_depth.dims(),
                    tile_depth.to_data().convert::<u32>().value,
                )
                .unwrap();
                rec.log(
                    "images/tile depth",
                    &rerun::Tensor::try_from(tile_depth).unwrap().clone(),
                )?;

                let pred_image = Array::from_shape_vec(
                    stats.pred_image.dims(),
                    stats.pred_image.to_data().convert::<f32>().value,
                )
                .unwrap();
                let pred_image = pred_image.map(|x| (*x * 255.0).clamp(0.0, 255.0) as u8);

                rec.log(
                    "images/predicted",
                    &rerun::Image::try_from(pred_image).unwrap(),
                )?;

                let first_cam = &scene.train_data[0].camera;

                let (img, _) =
                    splats.render(first_cam, glam::uvec2(512, 512), glam::vec3(0.0, 0.0, 0.0));
                let img = Array::from_shape_vec(img.dims(), img.to_data().convert::<f32>().value)
                    .unwrap();
                let img = img.map(|x| (*x * 255.0).clamp(0.0, 255.0) as u8);
                rec.log(
                    "images/fixed camera render",
                    &rerun::Image::try_from(img).unwrap(),
                )?;

                splats.visualize(&rec)?;
            }
        }
    }
    Ok(())
}
