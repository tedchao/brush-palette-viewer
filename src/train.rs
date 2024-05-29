use std::time;

use anyhow::Result;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::lr_scheduler::LrScheduler;
use burn::tensor::ElementConversion;
use burn::{
    config::Config,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::Tensor,
};
use ndarray::{Array, Array3};
use rand::{rngs::StdRng, SeedableRng};

use crate::gaussian_splats::{
    densify_and_prune, prune_invisible_points, reset_opacity, SplatsTrainState,
};
use crate::splat_render::{self, Backend};
use crate::{
    dataset_readers,
    gaussian_splats::{Splats, SplatsConfig},
    utils,
};
use rand::seq::SliceRandom;

#[derive(Config)]
pub(crate) struct LrConfig {
    #[config(default = 3e-3)]
    min_lr: f64,
    #[config(default = 3e-3)]
    max_lr: f64,
}

#[derive(Config)]
pub(crate) struct TrainConfig {
    pub lr_mean: LrConfig,
    pub lr_opac: LrConfig,
    pub lr_rest: LrConfig,

    #[config(default = 42)]
    pub(crate) seed: u64,
    #[config(default = 400)]
    pub(crate) warmup_steps: u32,
    #[config(default = 100)]
    pub(crate) refine_every: u32,
    // threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    #[config(default = 0.01)]
    pub(crate) cull_alpha_thresh: f32,
    #[config(default = 0.0015)]
    pub(crate) clone_split_grad_threshold: f32,

    #[config(default = 0.05)]
    pub(crate) split_clone_size_threshold: f32,

    // threshold of scale for culling huge gaussians.
    #[config(default = 0.5)]
    pub(crate) cull_scale_thresh: f32,
    #[config(default = 0.15)]
    pub(crate) cull_screen_size: f32,
    #[config(default = 20)]
    pub(crate) reset_alpha_every: u32,
    #[config(default = 10000)]
    pub(crate) train_steps: u32,
    #[config(default = false)]
    pub(crate) random_bck_color: bool,

    #[config(default = 25)]
    pub visualize_every: u32,
    #[config(default = 5000)]
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
    let opt_config = AdamConfig::new().with_epsilon(1e-15);
    let mut optim = opt_config.init::<B, Splats<B>>();

    B::seed(config.seed);
    let mut rng = StdRng::from_seed([10; 32]);

    #[cfg(feature = "rerun")]
    scene.visualize(&rec)?;

    let mut splats: Splats<B> =
        SplatsConfig::new(config.init_points, config.init_aabb, 0, 1.0).build(device);

    let mut state: SplatsTrainState<B> = SplatsTrainState::new(config.init_points, device);

    let mut sched_mean =
        CosineAnnealingLrSchedulerConfig::new(config.lr_mean.max_lr, config.train_steps as usize)
            .with_min_lr(config.lr_mean.min_lr)
            .init();

    let mut sched_opac =
        CosineAnnealingLrSchedulerConfig::new(config.lr_opac.max_lr, config.train_steps as usize)
            .with_min_lr(config.lr_opac.min_lr)
            .init();

    let mut sched_rest =
        CosineAnnealingLrSchedulerConfig::new(config.lr_rest.max_lr, config.train_steps as usize)
            .with_min_lr(config.lr_rest.min_lr)
            .init();

    println!("Start training.");

    // By default use 8 second window with 16 accumulators
    for iter in 0..config.train_steps {
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

        let (pred_img, aux) = splats.render(camera, img_size, background_color, false);
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

        let mut grad_means = GradientsParams::new();
        let mut grad_opac = GradientsParams::new();
        let mut grad_rest = GradientsParams::new();

        grad_means.register(
            splats.means.clone().consume().0,
            splats.means.grad_remove(&mut grads).unwrap(),
        );
        grad_rest.register(
            splats.sh_coeffs.clone().consume().0,
            splats.sh_coeffs.grad_remove(&mut grads).unwrap(),
        );
        grad_rest.register(
            splats.rotation.clone().consume().0,
            splats.rotation.grad_remove(&mut grads).unwrap(),
        );
        grad_rest.register(
            splats.log_scales.clone().consume().0,
            splats.log_scales.grad_remove(&mut grads).unwrap(),
        );
        grad_opac.register(
            splats.raw_opacity.clone().consume().0,
            splats.raw_opacity.grad_remove(&mut grads).unwrap(),
        );

        let lr_mean = LrScheduler::<B>::step(&mut sched_mean);
        let lr_opac = LrScheduler::<B>::step(&mut sched_opac);
        let lr_rest = LrScheduler::<B>::step(&mut sched_rest);

        splats = optim.step(lr_mean, splats, grad_means);
        splats = optim.step(lr_opac, splats, grad_opac);
        splats = optim.step(lr_rest, splats, grad_rest);

        state.update_stats(&aux, xys_grad);

        let stats = TrainStats {
            aux,
            gt_image: viewpoint.view.image.clone(),
            loss: mse,
            psnr,
            pred_image: pred_img,
        };

        if iter % config.refine_every == 0 {
            // Remove barely visible gaussians.
            prune_invisible_points(&mut splats, &mut state, config.cull_alpha_thresh);

            if iter > config.warmup_steps {
                densify_and_prune(
                    &mut splats,
                    &mut state,
                    config.clone_split_grad_threshold / (img_size.x.max(img_size.y) as f32),
                    Some(config.cull_screen_size * (dims[0] as f32)),
                    Some(config.cull_scale_thresh),
                    config.split_clone_size_threshold,
                    device,
                );

                if iter % (config.refine_every * config.reset_alpha_every) == 0 {
                    reset_opacity(&mut splats);
                }
            }
            state = SplatsTrainState::new(splats.num_splats(), device);

            optim = opt_config.init::<B, Splats<B>>();
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

            rec.log("lr/mean", &rerun::Scalar::new(lr_mean))?;
            rec.log("lr/opac", &rerun::Scalar::new(lr_opac))?;
            rec.log("lr/rest", &rerun::Scalar::new(lr_rest))?;

            rec.log(
                "performance/step_ms",
                &rerun::Scalar::new((time::Instant::now() - start_time).as_secs_f64() * 1000.0)
                    .clone(),
            )?;

            rec.log(
                "splats/num_intersects",
                &rerun::Scalar::new(utils::burn_to_scalar(stats.aux.num_intersects).elem::<f64>())
                    .clone(),
            )?;

            rec.log(
                "splats/num_visible",
                &rerun::Scalar::new(utils::burn_to_scalar(stats.aux.num_visible).elem::<f64>())
                    .clone(),
            )?;

            rec.log(
                "splats/num",
                &rerun::Scalar::new(splats.num_splats() as f64).clone(),
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

                // TODO: Could render a u32 texture here and decode?
                let (img, _) = splats.render(
                    first_cam,
                    glam::uvec2(512, 512),
                    glam::vec3(0.0, 0.0, 0.0),
                    true,
                );

                let bytes = img
                    .to_data()
                    .value
                    .iter()
                    .flat_map(|x| x.elem::<u32>().to_le_bytes())
                    .collect::<Vec<_>>();

                let img = Array::from_shape_vec([512, 512, 4], bytes).unwrap();
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
