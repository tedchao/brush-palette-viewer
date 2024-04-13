use crate::{
    dataset_readers,
    gaussian_splats::{self, Splats},
    splat_render::Backend,
};
use anyhow::Result;
use ndarray::Array;
use tracing_subscriber::{filter, layer::SubscriberExt, util::SubscriberInitExt, Layer};

pub(crate) fn view<B: Backend>(path: &str, viewpoints: &str, device: &B::Device) -> Result<()> {
    let rec = rerun::RecordingStreamBuilder::new("visualize training").spawn()?;
    let splats: Splats<B> = gaussian_splats::create_from_ply(path, device)?;

    let cameras = dataset_readers::read_viewpoint_data(viewpoints)?;
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default()),
    )
    .expect("setup tracy layer");

    for (i, camera) in cameras.iter().enumerate() {
        let (img, _) = splats.render(camera, glam::vec3(0.0, 0.0, 0.0));

        let img = Array::from_shape_vec(img.dims(), img.to_data().convert::<f32>().value).unwrap();
        let img = img.map(|x| (*x * 255.0).clamp(0.0, 255.0) as u8);

        rec.log_timeless(
            format!("images/fixed_camera_render_{i}"),
            &rerun::Image::try_from(img).unwrap(),
        )?;

        let rerun_camera = rerun::Pinhole::from_focal_length_and_resolution(
            camera.focal(),
            glam::vec2(camera.width as f32, camera.height as f32),
        );
        // TODO: make a function.
        let cam_path = format!("world/camera_{i}");
        rec.log_timeless(cam_path.clone(), &rerun_camera)?;
        rec.log_timeless(
            cam_path.clone(),
            &rerun::Transform3D::from_translation_rotation(camera.position(), camera.rotation()),
        )?;

        splats.visualize(&rec)?;
    }

    Ok(())
}
