use anyhow::Result;
use burn::lr_scheduler::LrScheduler;
use burn::{
    config::Config,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::Tensor,
};
use ndarray::{Array, Array1, Array3};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use crate::splat_render::{self, AutodiffBackend, Backend};
use crate::{
    dataset_readers,
    gaussian_splats::{Splats, SplatsConfig},
    loss_utils,
    scene::Scene,
    utils,
};

#[derive(Config)]
pub(crate) struct TrainConfig {
    #[config(default = 42)]
    pub(crate) seed: u64,
    #[config(default = 250)]
    pub(crate) train_steps: i32,
    #[config(default = false)]
    pub(crate) random_bck_color: bool,
    #[config(default = 3e-2)]
    pub lr: f64,
    #[config(default = 3e-4)]
    pub min_lr: f64,
    pub(crate) scene_path: String,
}

struct TrainStats {
    total_points: usize,
    pred_image: Array3<f32>,
    gt_image: Array3<f32>,
    loss: Array1<f32>,
}

fn compute_loss<B: Backend>(this_image: Tensor<B, 3>, other_image: Tensor<B, 3>) -> Tensor<B, 1> {
    // TODO: Restore ssim loss.
    loss_utils::l1_loss(this_image, other_image)
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
    iteration: i32,
    cur_lr: f64,
    config: &TrainConfig,
    optim: &mut impl Optimizer<Splats<B>, B>,
    rng: &mut StdRng,
    device: &B::Device,
) -> (Splats<B>, TrainStats)
where
    B::InnerBackend: Backend,
{
    println!("Train step {iteration}");

    if iteration % 1000 == 0 {
        splats.oneup_sh_degree();
    }

    // Get a random camera
    // TODO: Reproduciable RNG thingies.
    // TODO: If there is no camera, maybe just bail?
    let viewpoint = scene
        .train_data
        .choose(rng)
        .expect("Dataset should have at least 1 camera.");

    let background_color = if config.random_bck_color {
        glam::vec3(rand::random(), rand::random(), rand::random())
    } else {
        scene.default_bg_color
    };

    let render = splats.render(&viewpoint.camera, background_color);
    let render_img = render.clone();
    let gt_image = utils::ndarray_to_burn(viewpoint.view.image.clone(), device);
    // TODO: Burn should be able to slice open ranges.
    let dims = gt_image.dims();
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

    let loss = compute_loss(render_img.clone(), gt_image.clone());
    let grads = loss.backward();

    // Gradients linked to each parameter of the model.
    let grads = GradientsParams::from_grads(grads, &splats);

    // Update the model using the optimizer.
    splats = optim.step(cur_lr, splats, grads);

    // splats.update_rolling_statistics(render);
    // adaptive_density_control(scene, cfg);

    let num_points = splats.cur_num_points();
    (
        splats,
        TrainStats {
            total_points: num_points,
            gt_image: viewpoint.view.image.clone(),
            loss: Array::from_shape_vec(loss.dims(), loss.to_data().convert().value).unwrap(),
            pred_image: Array::from_shape_vec(
                render_img.dims(),
                render_img.to_data().convert().value,
            )
            .unwrap(),
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

    println!("Reading dataset.");

    let scene = dataset_readers::read_scene(&config.scene_path);
    let config_optimizer = AdamConfig::new();
    let mut optim = config_optimizer.init::<B, Splats<B>>();

    B::seed(config.seed);
    let mut rng = StdRng::from_seed([10; 32]);

    println!("Visualize scene.");

    scene.visualize(&rec)?;

    println!("Create splats.");

    let mut splats: Splats<B> = SplatsConfig::new(200 * 200, 1.0, 0, 1.0).build(device);

    // TODO: Original implementation has learning rates different for almost all params.
    let mut scheduler = burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig::new(
        config.lr,
        config.train_steps as usize,
    )
    .with_min_lr(config.min_lr)
    .init();

    println!("Start of training.");

    for iter in 0..config.train_steps + 1 {
        let lr = LrScheduler::<B>::step(&mut scheduler);

        let (new_splats, stats) = train_step(
            &scene, splats, iter, lr, config, &mut optim, &mut rng, device,
        );

        // Replace current model.
        splats = new_splats;

        rec.log("stats/loss", &rerun::Scalar::new(stats.loss[[0]] as f64))?;
        rec.log("stats/lr", &rerun::Scalar::new(lr))?;
        rec.log(
            "stats/total points",
            &rerun::Scalar::new(stats.total_points as f64),
        )?;
        rec.log(
            "images/ground truth",
            &utils::ndarray_to_rerun_image(&stats.gt_image),
        )?;
        rec.log(
            "images/predicted",
            &utils::ndarray_to_rerun_image(&stats.pred_image),
        )?;

        splats.visualize(&rec)?;
    }
    Ok(())
}
