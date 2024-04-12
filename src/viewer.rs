use crate::{
    camera::Camera,
    dataset_readers,
    gaussian_splats::{self, Splats},
    splat_render::Backend,
};
use anyhow::Result;
use ndarray::Array;

pub(crate) fn view<B: Backend>(path: &str, viewpoints: &str, device: &B::Device) -> Result<()> {
    let rec = rerun::RecordingStreamBuilder::new("visualize training").spawn()?;
    let splats: Splats<B> = gaussian_splats::create_from_ply(path, device)?;

    let cameras = dataset_readers::read_viewpoint_data(viewpoints)?;

    for (i, cam) in cameras[0..10].iter().enumerate() {
        let camera = Camera {
            width: 600,
            height: 600,
            fovx: cam.fovx,
            fovy: cam.fovy,
            ..cam.clone()
        };

        let (img, _) = splats.render(&camera, glam::vec3(0.0, 0.0, 0.0));
        let img = Array::from_shape_vec(img.dims(), img.to_data().convert::<f32>().value).unwrap();
        rec.log_timeless(
            format!("images/fixed camera render {i}"),
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
