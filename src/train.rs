use std::error::Error;

use burn::tensor::backend::Backend;

use crate::{dataset_readers, gaussian_splats::GaussianSplats, scene::Scene};

struct TrainConfig {
    scene_path: String,
    model_path: String,
    llffhold: String,
    resolution: (i32, i32),
    load_iter: i32,
    initialize_colmap_random: bool,
    max_iterations: i32,
}

struct GsScene;

// A single training step.

// Consists of the following steps.
//   1. Updating the learning rates
//   2. Enabling the SH degrees if appropriate
//   3. Run a forward/backward
//   4. Run adapative density control
//   5. Update the parameters

// Args:
//   scene: The scene that we will do the forward backward.
//   cfg: A Namespace with all the configuration for training.

// Returns:
//   A dictionary with any interesting information that could be used for
//   logging. The return can be ignored if the caller is not interested in
//   logging.
fn train_step<B: Backend>(
    scene: &mut Scene,
    splats: &mut GaussianSplats<B>,
    iteration: i32,
    cfg: TrainConfig,
) {
    splats.update_learning_rate(iteration);

    if iteration % 1000 == 0 {
        splats.oneup_sh_degree();
    }

    // let (total_loss, l1, dssim, pred_image, gt_image) = forward_backward(scene, cfg);
    // adaptive_density_control(scene, cfg);
    // scene.optimizer_step();
    // TODO: Decide on some metrics.
    //   return {
    //       "iteration": iteration,
    //       "total_points": scene.gaussian_splats.xyz.shape[0],
    //       "total_loss": total_loss,
    //       "l1": l1,
    //       "dssim": dssim,
    //       "pred_image": pred_image,
    //       "gt_image": gt_image,
    //       "iter_time": iter_start.elapsed_time(iter_end),
    //   }
}

// Training loop.
fn train(cfg: TrainConfig) -> Result<GsScene, Box<dyn Error>> {
    let rec = rerun::RecordingStreamBuilder::new("visualize training").spawn()?;
    let (scene, splats) = dataset_readers::read_scene(
        cfg.scene_path,
        cfg.model_path,
        cfg.llffhold,
        cfg.resolution,
        cfg.load_iter,
        cfg.initialize_colmap_random,
    );

    for iter in scene.iteration..cfg.max_iterations + 1 {
        let train_pkg = train_step(scene, splats, iter, cfg);
        // TODO: Log some metrics.
        // gs_train_utils.log_train_step(tb_writer, train_pkg)
    }

    return scene;
}
