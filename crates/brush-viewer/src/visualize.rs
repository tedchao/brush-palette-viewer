#![allow(unused_imports, unused_variables)]

use std::borrow::Borrow;

use anyhow::Result;
use brush_render::{gaussian_splats::Splats, AutodiffBackend, Backend};
use brush_train::scene::Scene;
use brush_train::train::TrainStepStats;
use burn::tensor::{activation::sigmoid, ElementConversion, Tensor};
use image::GenericImageView;

#[cfg(not(target_family = "wasm"))]
use brush_rerun::BurnToRerun;

#[cfg(not(target_family = "wasm"))]
use rerun::{Color, FillMode, RecordingStream};

pub struct VisualizeTools {
    #[cfg(not(target_family = "wasm"))]
    rec: Option<RecordingStream>,
}

impl VisualizeTools {
    pub fn new() -> Self {
        #[cfg(target_family = "wasm")]
        {
            Self {}
        }

        #[cfg(not(target_family = "wasm"))]
        {
            // Spawn rerun - creating this is already explicatly done by a user.
            let rec = rerun::RecordingStreamBuilder::new("Brush").spawn().ok();
            Self { rec }
        }
    }

    pub(crate) async fn log_splats<B: Backend>(&self, splats: Splats<B>) -> Result<()> {
        #[cfg(not(target_family = "wasm"))]
        {
            let Some(rec) = self.rec.as_ref() else {
                return Ok(());
            };

            if !rec.is_enabled() {
                return Ok(());
            }

            let means = splats
                .means
                .val()
                .into_data_async()
                .await
                .to_vec::<f32>()
                .unwrap();
            let means = means.chunks(3).map(|c| glam::vec3(c[0], c[1], c[2]));

            let sh_c0 = 0.2820947917738781;
            let base_rgb =
                splats.sh_coeffs.val().slice([0..splats.num_splats(), 0..3]) * sh_c0 + 0.5;

            let transparency = sigmoid(splats.raw_opacity.val());

            let colors = base_rgb.into_data_async().await.to_vec::<f32>().unwrap();
            let colors = colors.chunks(3).map(|c| {
                Color::from_rgb(
                    (c[0] * 255.0) as u8,
                    (c[1] * 255.0) as u8,
                    (c[2] * 255.0) as u8,
                )
            });

            // Visualize 2 sigma, and simulate some of the small covariance blurring.
            let radii = (splats.log_scales.val().exp() * transparency.unsqueeze_dim(1) * 2.0
                + 0.004)
                .into_data_async()
                .await
                .to_vec()
                .unwrap();

            let rotations = splats
                .rotation
                .val()
                .into_data_async()
                .await
                .to_vec::<f32>()
                .unwrap();
            let rotations = rotations
                .chunks(4)
                .map(|q| glam::Quat::from_array([q[1], q[2], q[3], q[0]]));

            let radii = radii.chunks(3).map(|r| glam::vec3(r[0], r[1], r[2]));

            rec.log(
                "world/splat/points",
                &rerun::Ellipsoids3D::from_centers_and_half_sizes(means, radii)
                    .with_quaternions(rotations)
                    .with_colors(colors)
                    .with_fill_mode(FillMode::Solid),
            )?;
        }

        Ok(())
    }

    pub(crate) fn log_scene(&self, scene: &Scene) -> Result<()> {
        #[cfg(not(target_family = "wasm"))]
        {
            let Some(rec) = self.rec.as_ref() else {
                return Ok(());
            };

            if !rec.is_enabled() {
                return Ok(());
            }

            rec.log_static("world", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN)?;

            for (i, view) in scene.views.iter().enumerate() {
                let path = format!("world/dataset/camera/{i}");
                let (width, height) = (view.image.width(), view.image.height());

                let vis_size = glam::uvec2(width, height);
                let rerun_camera = rerun::Pinhole::from_focal_length_and_resolution(
                    view.camera.focal(vis_size),
                    glam::vec2(vis_size.x as f32, vis_size.y as f32),
                );
                rec.log_static(path.clone(), &rerun_camera)?;
                rec.log_static(
                    path.clone(),
                    &rerun::Transform3D::from_translation_rotation(
                        view.camera.position,
                        view.camera.rotation,
                    ),
                )?;
                rec.log_static(
                    path + "/image",
                    &rerun::Image::from_dynamic_image(view.image.as_ref().clone())?,
                )?;
            }
        }

        Ok(())
    }

    pub fn log_eval_stats(&self, iter: u32, stats: brush_train::eval::EvalStats) -> Result<()> {
        #[cfg(not(target_family = "wasm"))]
        {
            let Some(rec) = self.rec.as_ref() else {
                return Ok(());
            };

            if !rec.is_enabled() {
                return Ok(());
            }

            rec.set_time_sequence("iterations", iter);

            let avg_psnr =
                stats.samples.iter().map(|s| s.psnr).sum::<f32>() / (stats.samples.len() as f32);
            rec.log("stats/eval_psnr", &rerun::Scalar::new(avg_psnr as f64))?;

            for (i, samp) in stats.samples.iter().enumerate() {
                let render = samp.rendered.clone();
                let [w, h] = [render.width(), render.height()];
                rec.log(
                    format!("eval/render {i}"),
                    &rerun::Image::from_rgb24(render.into_bytes(), [w, h]),
                )?;
                rec.log(
                    format!("eval/render {i}"),
                    &rerun::Image::from_rgb24(samp.ground_truth.as_bytes().to_vec(), [w, h]),
                )?;
            }
        }
        Ok(())
    }

    pub async fn log_train_stats<B: AutodiffBackend>(
        &self,
        iter: u32,
        splats: Splats<B>,
        stats: TrainStepStats<B>,
    ) -> Result<()> {
        #[cfg(not(target_family = "wasm"))]
        {
            let Some(rec) = self.rec.as_ref() else {
                return Ok(());
            };

            if !rec.is_enabled() {
                return Ok(());
            }

            rec.set_time_sequence("iterations", iter);
            rec.log("lr/mean", &rerun::Scalar::new(stats.lr_mean))?;
            rec.log("lr/rotation", &rerun::Scalar::new(stats.lr_rotation))?;
            rec.log("lr/scale", &rerun::Scalar::new(stats.lr_scale))?;
            rec.log("lr/coeffs", &rerun::Scalar::new(stats.lr_coeffs))?;
            rec.log("lr/opac", &rerun::Scalar::new(stats.lr_opac))?;

            rec.log(
                "splats/num_splats",
                &rerun::Scalar::new(splats.num_splats() as f64).clone(),
            )?;
            let [batch_size, img_h, img_w, _] = stats.pred_images.dims();
            let pred_rgb =
                stats
                    .pred_images
                    .clone()
                    .slice([0..batch_size, 0..img_h, 0..img_w, 0..3]);
            let gt_rgb = stats
                .gt_images
                .clone()
                .slice([0..batch_size, 0..img_h, 0..img_w, 0..3]);
            let mse = (pred_rgb.clone() - gt_rgb.clone()).powf_scalar(2.0).mean();
            let psnr = mse.recip().log() * 10.0 / std::f32::consts::LN_10;
            rec.log(
                "losses/main",
                &rerun::Scalar::new(stats.loss.clone().into_scalar_async().await.elem::<f64>()),
            )?;
            rec.log(
                "stats/train_psnr",
                &rerun::Scalar::new(psnr.into_scalar_async().await.elem::<f64>()),
            )?;
            // Not sure what's best here, atm let's just log the first batch render only.
            // Maybe could do an average instead?
            let main_aux = &stats.auxes[0];

            rec.log(
                "splats/num_intersects",
                &rerun::Scalar::new(main_aux.read_num_intersections() as f64),
            )?;
            rec.log(
                "splats/splats_visible",
                &rerun::Scalar::new(main_aux.read_num_visible() as f64),
            )?;
            // rec.log(
            //     "images/tile_depth",
            //     &main_aux.read_tile_depth().into_rerun(),
            // )?;
        }

        Ok(())
    }
}
