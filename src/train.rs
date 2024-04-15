use std::time::Instant;

use anyhow::Result;
use burn::lr_scheduler::LrScheduler;
use burn::nn::loss::MseLoss;
use burn::{
    config::Config,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::Tensor,
};
use ndarray::{Array, Array1, Array3};
use rand::{rngs::StdRng, SeedableRng};
use tracing::info_span;

use crate::splat_render::{self, AutodiffBackend, Backend, BurnBack};
use crate::{
    dataset_readers,
    gaussian_splats::{Splats, SplatsConfig},
    scene::Scene,
    utils,
};

use rand::seq::SliceRandom;

#[derive(Config)]
pub(crate) struct TrainConfig {
    #[config(default = 42)]
    pub(crate) seed: u64,
    #[config(default = 15000)]
    pub(crate) train_steps: u32,
    #[config(default = false)]
    pub(crate) random_bck_color: bool,
    #[config(default = 2e-2)]
    pub lr: f64,
    #[config(default = 1e-2)]
    pub min_lr: f64,
    #[config(default = 10)]
    pub visualize_every: u32,
    #[config(default = 16000)]
    pub init_points: usize,
    #[config(default = 2.0)]
    pub init_aabb: f32,
    pub scene_path: String,
}

struct TrainStats<B: Backend> {
    total_points: usize,
    pred_image: Tensor<B, 3>,
    gt_image: Array3<f32>,
    loss: Array1<f32>,
    aux: crate::splat_render::Aux<BurnBack>,
}

// Consists of the following steps.
//   1. Updating the learning rates
//   2. Enabling the SH degrees if appropriate
//   3. Run a forward/backward
//   4. Run adapative density control
//   5. Update the parameters

// Args:
//   scene: The scene that we will do the forward backward.
//   cfg: A Namespace with all the configuration for training.
fn train_step<B: AutodiffBackend>(
    scene: &Scene,
    mut splats: Splats<B>,
    _iteration: u32,
    cur_lr: f64,
    config: &TrainConfig,
    loss: &MseLoss<B>,
    optim: &mut impl Optimizer<Splats<B>, B>,
    rng: &mut StdRng,
    device: &B::Device,
) -> (Splats<B>, TrainStats<B>)
where
    B::InnerBackend: Backend,
{
    let _span = info_span!("Train step").entered();

    // Get a random camera
    // TODO: Reproduciable RNG thingies.
    // TODO: If there is no camera, maybe just bail?
    let viewpoint = scene
        .train_data
        .choose(rng)
        .expect("Dataset should have at least 1 camera.");

    let camera = &viewpoint.camera;

    let background_color = if config.random_bck_color {
        glam::vec3(rand::random(), rand::random(), rand::random())
    } else {
        scene.default_bg_color
    };

    let (pred_img, aux) = splats.render(camera, background_color);
    let dims = pred_img.dims();

    let rgb_img = pred_img.clone().slice([0..dims[0], 0..dims[1], 0..3]);
    let gt_image = utils::ndarray_to_burn(viewpoint.view.image.clone(), device);
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

    let loss_scalar = loss.forward(
        rgb_img.clone(),
        gt_image.clone(),
        burn::nn::loss::Reduction::Auto,
    );
    let grads = loss_scalar.backward();

    // Gradients linked to each parameter of the model.
    let grads = GradientsParams::from_grads(grads, &splats);

    // Update the model using the optimizer.
    splats = optim.step(cur_lr, splats, grads);

    let num_points = splats.cur_num_points();
    (
        splats,
        TrainStats {
            total_points: num_points,
            aux,
            gt_image: viewpoint.view.image.clone(),
            loss: Array::from_shape_vec(loss_scalar.dims(), loss_scalar.to_data().convert().value)
                .unwrap(),
            pred_image: pred_img,
        },
    )
}

// Training loop.
pub(crate) fn train<B: splat_render::AutodiffBackend>(
    config: &TrainConfig,
    device: &B::Device,
) -> Result<()>
where
    B::InnerBackend: Backend,
{
    let rec = rerun::RecordingStreamBuilder::new("visualize training").spawn()?;

    println!("Loading dataset.");
    let scene = dataset_readers::read_scene(&config.scene_path, None, false);
    let config_optimizer = AdamConfig::new();
    let mut optim = config_optimizer.init::<B, Splats<B>>();

    B::seed(config.seed);
    let mut rng = StdRng::from_seed([10; 32]);
    scene.visualize(&rec)?;

    let mut splats: Splats<B> =
        SplatsConfig::new(config.init_points, config.init_aabb, 0, 1.0).build(device);

    // TODO: Original implementation has differennt learning rates for almost all params.
    let mut scheduler = burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig::new(
        config.lr,
        config.train_steps as usize,
    )
    .with_min_lr(config.min_lr)
    .init();

    println!("Start training.");

    let loss = MseLoss::new();

    // By default use 8 second window with 16 accumulators
    for iter in 0..config.train_steps + 1 {
        let start_time = Instant::now();

        let lr = LrScheduler::<B>::step(&mut scheduler);

        let (new_splats, stats) = train_step(
            &scene, splats, iter, lr, config, &loss, &mut optim, &mut rng, device,
        );

        // Replace current model.
        splats = new_splats;

        rec.set_time_sequence("iterations", iter);
        rec.log("losses/main", &rerun::Scalar::new(stats.loss[[0]] as f64))?;
        rec.log("lr/current", &rerun::Scalar::new(lr))?;
        if iter > 5 {
            rec.log(
                "performance/train step ms",
                &rerun::Scalar::new((Instant::now() - start_time).as_secs_f32() as f64 * 1000.0),
            )?;
        }
        rec.log(
            "points/total",
            &rerun::Scalar::new(stats.total_points as f64),
        )?;
        rec.log(
            "points/num intersects",
            &rerun::Scalar::new(stats.aux.num_intersects as f64),
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

            let first_cam = &scene.train_data[1].camera;

            let start_time = Instant::now();
            let (img, _) = splats.render(first_cam, glam::vec3(0.0, 0.0, 0.0));
            let img =
                Array::from_shape_vec(img.dims(), img.to_data().convert::<f32>().value).unwrap();
            rec.log(
                "performance/render ms",
                &rerun::Scalar::new((Instant::now() - start_time).as_secs_f32() as f64 * 1000.0),
            )?;

            let img = img.map(|x| (*x * 255.0).clamp(0.0, 255.0) as u8);
            rec.log(
                "images/fixed camera render",
                &rerun::Image::try_from(img).unwrap(),
            )?;

            splats.visualize(&rec)?;
        }
    }
    Ok(())
}
