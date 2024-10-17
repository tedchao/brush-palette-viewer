use anyhow::Result;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_render::{AutodiffBackend, Backend, RenderAux};
use burn::lr_scheduler::exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig};
use burn::lr_scheduler::LrScheduler;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::Adam;
use burn::tensor::{Bool, Distribution, Int};
use burn::{
    config::Config,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::Tensor,
};
use tracing::trace_span;

#[derive(Config)]
pub struct TrainConfig {
    #[config(default = 30000)]
    total_steps: usize,

    // period of steps where refinement is turned off
    #[config(default = 500)]
    warmup_steps: u32,

    // period of steps where gaussians are culled and densified
    #[config(default = 100)]
    refine_every: u32,

    #[config(default = 0.5)]
    stop_refine_percent: f32,

    // threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality
    #[config(default = 0.1)]
    cull_alpha_thresh: f32,

    // threshold of scale for culling huge gaussians
    #[config(default = 0.5)]
    cull_scale_thresh: f32,

    // Every this many refinement steps, reset the alpha
    #[config(default = 30)]
    reset_alpha_every: u32,

    // threshold of positional gradient norm for densifying gaussians
    // TODO: Abs grad.
    #[config(default = 0.00005)]
    densify_grad_thresh: f32,

    // below this size, gaussians are *duplicated*, otherwise split.
    #[config(default = 0.01)]
    densify_size_thresh: f32,

    #[config(default = 0.0)]
    ssim_weight: f32,

    // Learning rates.
    lr_mean: ExponentialLrSchedulerConfig,
    #[config(default = 0.0025)]
    lr_coeffs: f64,

    #[config(default = 0.05)]
    lr_opac: f64,

    #[config(default = 0.005)]
    lr_scale: f64,

    #[config(default = 0.001)]
    lr_rotation: f64,

    #[config(default = 5000)]
    schedule_steps: u32,

    #[config(default = 42)]
    seed: u64,

    #[config(default = 1000)]
    pub eval_every: u32,

    #[config(default = 400)]
    pub visualize_splats_every: u32,
}

#[derive(Clone, Debug)]
pub struct SceneBatch<B: Backend> {
    pub gt_images: Tensor<B, 4>,
    pub cameras: Vec<Camera>,
}

pub struct TrainStepStats<B: AutodiffBackend> {
    pub pred_images: Tensor<B, 4>,
    pub auxes: Vec<RenderAux>,
    pub loss: Tensor<B, 1>,
    pub lr_mean: f64,
    pub iter: u32,
}

pub struct SplatTrainer<B: AutodiffBackend>
where
    B::InnerBackend: Backend,
{
    pub iter: u32,

    config: TrainConfig,

    sched_mean: ExponentialLrScheduler,
    optim: OptimizerAdaptor<Adam<B::InnerBackend>, Splats<B>, B>,
    opt_config: AdamConfig,

    // Helper tensors for accumulating the viewspace_xy gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    grad_2d_accum: Tensor<B, 1>,
    xy_grad_counts: Tensor<B, 1, Int>,
}

fn quaternion_vec_multiply<B: Backend>(
    quaternions: Tensor<B, 2>,
    vectors: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let num_points = quaternions.dims()[0];

    // Extract components of quaternions
    let qw = quaternions.clone().slice([0..num_points, 0..1]);
    let qx = quaternions.clone().slice([0..num_points, 1..2]);
    let qy = quaternions.clone().slice([0..num_points, 2..3]);
    let qz = quaternions.clone().slice([0..num_points, 3..4]);

    // Extract components of vectors
    let vx = vectors.clone().slice([0..num_points, 0..1]);
    let vy = vectors.clone().slice([0..num_points, 1..2]);
    let vz = vectors.clone().slice([0..num_points, 2..3]);

    // Compute intermediate terms
    let term1 = qw.clone() * vx.clone() + qy.clone() * vz.clone() - qz.clone() * vy.clone();
    let term2 = qw.clone() * vy.clone() - qx.clone() * vz.clone() + qz.clone() * vx.clone();
    let term3 = qw.clone() * vz.clone() + qx.clone() * vy.clone() - qy.clone() * vx.clone();
    let term4 = qx.clone() * vx.clone() + qy.clone() * vy.clone() + qz.clone() * vz.clone();

    // Compute final result
    let rx = vx
        + (qw.clone() * term1.clone() + qx.clone() * term4.clone() - qy.clone() * term3.clone()
            + qz.clone() * term2.clone())
            * 2.0;
    let ry = vy
        + (qw.clone() * term2.clone() - qx.clone() * term3.clone()
            + qy.clone() * term4.clone()
            + qz.clone() * term1.clone())
            * 2.0;
    let rz = vz
        + (qw * term3.clone() + qx * term2.clone() - qy * term1.clone() + qz * term4.clone()) * 2.0;

    Tensor::cat(vec![rx, ry, rz], 1)
}

impl<B: AutodiffBackend> SplatTrainer<B>
where
    B::InnerBackend: Backend,
{
    pub fn new(num_points: usize, config: &TrainConfig, splats: &Splats<B>) -> Self {
        let opt_config = AdamConfig::new().with_epsilon(1e-15);
        let optim = opt_config.init::<B, Splats<B>>();

        let device = &splats.means.device();

        Self {
            config: config.clone(),
            iter: 0,
            sched_mean: config.lr_mean.init(),
            optim,
            opt_config,
            grad_2d_accum: Tensor::zeros([num_points], device),
            xy_grad_counts: Tensor::zeros([num_points], device),
        }
    }

    fn reset_stats(&mut self, num_points: usize, device: &B::Device) {
        self.grad_2d_accum = Tensor::zeros([num_points], device);
        self.xy_grad_counts = Tensor::zeros([num_points], device);
    }

    pub(crate) fn reset_opacity(&self, splats: &mut Splats<B>) {
        let inv_sigmoid = |x: f32| (x / (1.0 - x)).ln();
        let base = inv_sigmoid(self.config.cull_alpha_thresh * 0.5);
        Splats::map_param(&mut splats.raw_opacity, |op| Tensor::zeros_like(&op) + base);
    }

    pub async fn step(
        &mut self,
        batch: SceneBatch<B>,
        background_color: glam::Vec3,
        splats: Splats<B>,
    ) -> Result<(Splats<B>, TrainStepStats<B>), anyhow::Error> {
        let mut splats = splats;

        let device = &splats.means.device();
        // let _span = trace_span!("Train step").entered();

        let [batch_size, img_h, img_w, _] = batch.gt_images.dims();

        let (pred_images, auxes, loss) = {
            let mut renders = vec![];
            let mut auxes = vec![];

            for i in 0..batch.cameras.len() {
                let camera = &batch.cameras[i];

                let (pred_image, aux) = splats.render(
                    camera,
                    glam::uvec2(img_w as u32, img_h as u32),
                    background_color,
                    false,
                );

                renders.push(pred_image);
                auxes.push(aux);
            }

            let pred_images = Tensor::stack(renders, 0);

            let _span = trace_span!("Calculate losses", sync_burn = true).entered();

            let pred_rgb = pred_images
                .clone()
                .slice([0..batch_size, 0..img_h, 0..img_w, 0..3]);
            let gt_rgb = batch
                .gt_images
                .clone()
                .slice([0..batch_size, 0..img_h, 0..img_w, 0..3]);

            let loss = (pred_rgb.clone() - gt_rgb.clone()).abs().mean();
            let loss = if self.config.ssim_weight > 0.0 {
                let ssim_loss = crate::ssim::ssim(
                    pred_rgb.clone().permute([0, 3, 1, 2]),
                    gt_rgb.clone().permute([0, 3, 1, 2]),
                    11,
                );
                loss * (1.0 - self.config.ssim_weight)
                    + (-ssim_loss + 1.0) * self.config.ssim_weight
            } else {
                loss
            };

            (pred_images, auxes, loss)
        };

        let max_refine_step =
            (self.config.stop_refine_percent * self.config.total_steps as f32) as u32;

        let mut grads = trace_span!("Backward pass", sync_burn = true).in_scope(|| loss.backward());

        if self.iter < max_refine_step {
            trace_span!("Housekeeping", sync_burn = true).in_scope(|| {
                let xys_grad = Tensor::from_inner(
                    splats
                        .xys_dummy
                        .grad_remove(&mut grads)
                        .expect("XY gradients need to be calculated."),
                );

                // From normalized to pixels.
                let xys_grad = xys_grad
                    * Tensor::<_, 1>::from_floats([img_w as f32 / 2.0, img_h as f32 / 2.0], device)
                        .reshape([1, 2]);

                let grad_mag = xys_grad.powf_scalar(2.0).sum_dim(1).squeeze(1).sqrt();

                // TODO: Add += to Burn.
                if self.iter > self.config.warmup_steps {
                    self.grad_2d_accum = self.grad_2d_accum.clone() + grad_mag.clone();
                    self.xy_grad_counts =
                        self.xy_grad_counts.clone() + grad_mag.greater_elem(0.0).int();
                }
            });

            splats = trace_span!("Optimizer step", sync_burn = true).in_scope(|| {
                let mut splats = splats;
                let grad_means =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.means.id]);
                splats = self.optim.step(self.sched_mean.step(), splats, grad_means);

                let grad_opac =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.raw_opacity.id]);
                splats = self.optim.step(self.config.lr_opac, splats, grad_opac);

                let grad_coeff =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.sh_coeffs.id]);
                splats = self.optim.step(self.config.lr_coeffs, splats, grad_coeff);

                let grad_rot =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.rotation.id]);
                splats = self.optim.step(self.config.lr_rotation, splats, grad_rot);

                let grad_scale =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.log_scales.id]);
                splats = self.optim.step(self.config.lr_scale, splats, grad_scale);

                // Make sure rotations are still valid after optimization step.
                splats.norm_rotations();
                splats
            });

            if self.iter >= self.config.warmup_steps && self.iter % self.config.refine_every == 0 {
                let grads = self.grad_2d_accum.clone()
                    / self.xy_grad_counts.clone().clamp(1, i32::MAX).float();

                let big_grad_mask = grads.greater_equal_elem(self.config.densify_grad_thresh);

                // TODO: Carry adam state.
                // let optim_record = self.optim.to_record();
                // let means = splats.means.val();
                // let means_state = optim_record[&splats.means.id].into_state();

                let split_clone_size_mask = splats
                    .log_scales
                    .val()
                    .exp()
                    .max_dim(1)
                    .squeeze(1)
                    .lower_elem(self.config.densify_size_thresh);

                let mut append_means = vec![];
                let mut append_rots = vec![];
                let mut append_coeffs = vec![];
                let mut append_opac = vec![];
                let mut append_scales = vec![];

                let clone_inds = Tensor::stack::<2>(
                    vec![split_clone_size_mask.clone(), big_grad_mask.clone()],
                    1,
                )
                .all_dim(1)
                .squeeze::<1>(1)
                .argwhere_async()
                .await;

                let clone_count = clone_inds.dims()[0];
                if clone_count > 0 {
                    let clone_inds = clone_inds.squeeze(1);
                    let cur_scale = splats.log_scales.val().select(0, clone_inds.clone());
                    let cur_rots = splats.rotation.val().select(0, clone_inds.clone());

                    // Slightly offset cloned gaussians so they don't just follow the original one.
                    let samples = quaternion_vec_multiply(
                        cur_rots.clone(),
                        Tensor::random([clone_count, 3], Distribution::Normal(0.0, 0.5), device)
                            * cur_scale.clone().exp(),
                    );

                    append_means.push(splats.means.val().select(0, clone_inds.clone()) + samples);
                    append_rots.push(splats.rotation.val().select(0, clone_inds.clone()));
                    append_coeffs.push(splats.sh_coeffs.val().select(0, clone_inds.clone()));
                    append_opac.push(splats.raw_opacity.val().select(0, clone_inds.clone()));
                    append_scales.push(splats.log_scales.val().select(0, clone_inds.clone()));
                }

                let split_inds =
                    Tensor::stack::<2>(vec![split_clone_size_mask.bool_not(), big_grad_mask], 1)
                        .all_dim(1)
                        .squeeze::<1>(1)
                        .argwhere_async()
                        .await;

                let split_count = split_inds.dims()[0];
                if split_count > 0 {
                    let split_inds = split_inds.squeeze(1);
                    let cur_means = splats.means.val().select(0, split_inds.clone());
                    let cur_coeff = splats.sh_coeffs.val().select(0, split_inds.clone());
                    let cur_raw_opac = splats.raw_opacity.val().select(0, split_inds.clone());
                    let cur_log_scale = splats.log_scales.val().select(0, split_inds.clone());
                    let cur_rots = splats.rotation.val().select(0, split_inds.clone());

                    let samples = quaternion_vec_multiply(
                        cur_rots.clone(),
                        Tensor::random([split_count, 3], Distribution::Normal(0.0, 0.5), device)
                            * cur_log_scale.clone().exp(),
                    );

                    // TODO: Should also modify the old splat to be smaller & shifted.

                    append_means.push(cur_means.clone() + samples);
                    append_rots.push(cur_rots.clone());
                    append_coeffs.push(cur_coeff.clone());
                    append_opac.push(cur_raw_opac);
                    append_scales.push((cur_log_scale.exp() / 1.6).log());
                }

                if !append_means.is_empty() {
                    let append_means = Tensor::cat(append_means, 0);
                    let append_rots = Tensor::cat(append_rots, 0);
                    let append_coeffs = Tensor::cat(append_coeffs, 0);
                    let append_opac = Tensor::cat(append_opac, 0);
                    let append_scales = Tensor::cat(append_scales, 0);

                    concat_splats(
                        &mut splats,
                        append_means,
                        append_rots,
                        append_coeffs,
                        append_opac,
                        append_scales,
                    );
                }

                // Do some more processing. Important to do this last as otherwise you might mess up the correspondence
                // of gradient <-> splat.

                // Remove barely visible gaussians.
                let alpha_mask = burn::tensor::activation::sigmoid(splats.raw_opacity.val())
                    .lower_elem(self.config.cull_alpha_thresh);
                prune_points(&mut splats, alpha_mask).await;

                // Delete Gaussians with too large of a radius in world-units.
                let scale_mask = splats
                    .log_scales
                    .val()
                    .exp()
                    .max_dim(1)
                    .squeeze(1)
                    .greater_elem(self.config.cull_scale_thresh);
                prune_points(&mut splats, scale_mask).await;

                // Stats don't line up anymore so have to reset them.
                self.reset_stats(splats.num_splats(), device);

                // TODO: Want to do state surgery and keep momenta for splats.
                self.optim = self.opt_config.init();

                // Nb: th
                if self.iter % (self.config.refine_every * self.config.reset_alpha_every) == 0 {
                    self.reset_opacity(&mut splats);
                }
            }
        }

        self.iter += 1;

        let stats = TrainStepStats {
            pred_images,
            auxes,
            loss,
            lr_mean: self.sched_mean.step(),
            iter: self.iter,
        };

        Ok((splats, stats))
    }
}

// Prunes points based on the given mask.
//
// Args:
//   mask: bool[n]. If True, prune this Gaussian.
pub async fn prune_points<B: AutodiffBackend>(splats: &mut Splats<B>, prune: Tensor<B, 1, Bool>) {
    // bool[n]. If True, delete these Gaussians.
    let prune_count = prune.dims()[0];

    if prune_count == 0 {
        return;
    }

    let valid_inds = prune.bool_not().argwhere_async().await.squeeze(1);
    let start_splats = splats.num_splats();
    let new_points = valid_inds.dims()[0];

    if new_points < start_splats {
        splats.means = splats
            .means
            .clone()
            .map(|x| Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad());
        splats.sh_coeffs = splats
            .sh_coeffs
            .clone()
            .map(|x| Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad());
        splats.rotation = splats
            .rotation
            .clone()
            .map(|x| Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad());
        splats.raw_opacity = splats
            .raw_opacity
            .clone()
            .map(|x| Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad());
        splats.log_scales = splats
            .log_scales
            .clone()
            .map(|x| Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad());
    }
}

pub fn concat_splats<B: AutodiffBackend>(
    splats: &mut Splats<B>,
    means: Tensor<B, 2>,
    rotations: Tensor<B, 2>,
    sh_coeffs: Tensor<B, 2>,
    raw_opacities: Tensor<B, 1>,
    log_scales: Tensor<B, 2>,
) {
    Splats::map_param(&mut splats.means, |x| {
        Tensor::cat(vec![x, means.clone()], 0)
    });
    Splats::map_param(&mut splats.rotation, |x| {
        Tensor::cat(vec![x, rotations.clone()], 0)
    });
    Splats::map_param(&mut splats.sh_coeffs, |x| {
        Tensor::cat(vec![x, sh_coeffs.clone()], 0)
    });
    Splats::map_param(&mut splats.raw_opacity, |x| {
        Tensor::cat(vec![x, raw_opacities.clone()], 0)
    });
    Splats::map_param(&mut splats.log_scales, |x| {
        Tensor::cat(vec![x, log_scales.clone()], 0)
    });
}
