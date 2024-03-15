use std::path::PathBuf;

use crate::camera::{Camera, InputData};
use crate::gaussian_splats::GaussianSplats;
use crate::scene;
use burn::tensor::Tensor;
use serde_json;

// Reads cameras from a NeRF json file.
// Args:
//   path: Base directory of the transforms.json scene.
//   transformsfile: The name of the .json file to be processed.
//   extension: Often transforms file are missing the extension.
// Returns:
//   A list of cameras corresponding to the files parsed.
fn read_data_from_file(base_path: &str, transformsfile: &str, extension: &str) -> Vec<InputData> {
    let cameras = vec![];

    let path = PathBuf::from(base_path).join(transformsfile);
    let file = std::fs::read_to_string(path).expect("Couldn't find transforms file.");
    let contents: serde_json::Value = serde_json::from_str(&file).unwrap();
    let fovx = contents.get("camera_angle_x");

    // for frame in contents["frames"] {
    //     cam_name = os.path.join(path, frame["file_path"] + extension)

    //     // NeRF 'transform_matrix' is a camera-to-world transform
    //     c2w = torch.tensor(frame["transform_matrix"])
    //     // Change from OGL/Blender (Y up, Z back) to COLMAP (Y down, Z forward)
    //     c2w[:3, 1:3] *= -1

    //     // get the world-to-camera transform and set R, T
    //     w2c = torch.inverse(c2w)
    //     rotation = w2c[:3, :3]
    //     translation = w2c[:3, 3]

    //     image_path = os.path.join(path, cam_name)
    //     image_file = gfile.GFile(image_path, "rb")
    //     image = Image.open(image_file)
    //     im_data = torch.tensor(np.array(image.convert("RGBA"))) / 255.0

    //     image = im_data[..., :3].permute(2, 0, 1)
    //     bg_mask = im_data[..., 3:4].permute(2, 0, 1)

    //     fovy = gs_utils.focal_to_fov(
    //         gs_utils.fov_to_focal(fovx, image.shape[2]), image.shape[1]
    //     )

    //     cameras.append(
    //         gs_camera.InputCamera(
    //             rotation=rotation,
    //             translation=translation,
    //             fovy=fovy,
    //             fovx=fovx,
    //             image=image,
    //             image_path=image_path,
    //             bg_mask=bg_mask,
    //         )
    //     )
    // }

    cameras
}

// Reads synthetic scenes from NeRF.
fn read_nerf_synthetic_scene(scene_path: &str, model_path: &str, resolution: i32) -> scene::Scene {
    if resolution != -1 {
        println!("In NeRF Synthetic Datasets we ignore user specified resolution");
    }

    println!("Reading Training Transforms");
    let train_cams = read_data_from_file(scene_path, "transforms_train.json", ".png");
    println!("Reading Test Transforms");
    let test_cams = read_data_from_file(scene_path, "transforms_test.json", ".png");

    // Since this data set has no colmap data, we start with random points
    let num_pts = 100_000;
    let aabb_scale = 2.6;
    // We create random points inside the bounds of the synthetic Blender scenes
    println!("Generating {num_pts} random points...");

    // let initial_xyz = Tensor::rand((num_pts, 3)) * aabb_scale - aabb_scale / 2.0;
    // let initial_rgb = Tensor::tensor([[0.5, 0.5, 0.5]]).repeat(num_pts, 1);

    let mut scene = scene::Scene::new(model_path);
    scene.initialize(train_cams, test_cams);
    scene
}

// // Reads a colmap scene.
// fn read_colmap_scene(
//     scene_path: &str,
//     model_path: &str,
//     resolution: i32,
//     llffhold: i32,
//     initialize_random: bool,
// ) -> scene::Scene {
//   images_root_path = os.path.join(scene_path, "images")
//   colmap_root_path = os.path.join(scene_path, "sparse/0")

//   colmap_scene = pycolmap.SceneManager(colmap_root_path)
//   colmap_scene.load()

//   cameras = []
//   logging.info("Reading colmap images and cameras:")

//   for _, image in tqdm.tqdm(colmap_scene.images.items()):
//     width = colmap_scene.cameras[image.camera_id].width
//     height = colmap_scene.cameras[image.camera_id].height
//     fovx = gs_utils.focal_to_fov(
//         colmap_scene.cameras[image.camera_id].fx, width
//     )
//     fovy = gs_utils.focal_to_fov(
//         colmap_scene.cameras[image.camera_id].fy, height
//     )
//     rotation = gs_utils.qvec2rotmat(
//         torch.tensor(image.q.q[None, ...], dtype=torch.float32)
//     ).squeeze()

//     translation = torch.tensor(image.tvec, dtype=torch.float32)
//     image_name = image.name
//     image_path = os.path.join(images_root_path, image_name)

//     image_file = gfile.GFile(image_path, "rb")
//     pil_image = Image.open(image_file)
//     if pil_image.size[0] != width or pil_image.size[1] != height:
//       raise ValueError(
//           f"Colmap width/height {(width,height)} does not match the actual"
//           f" image file {pil_image.size}!"
//       )

//     resized_pil_image = gs_utils.downscale_pil_image(resolution, pil_image)

//     im_data = torch.tensor(np.array(resized_pil_image.convert("RGB"))) / 255.0
//     im_data = im_data.permute(2, 0, 1)
//     bg_mask = torch.ones_like(im_data)

//     cameras.append(
//         gs_camera.InputCamera(
//             rotation=rotation,
//             translation=translation,
//             fovy=fovy,
//             fovx=fovx,
//             image=im_data,
//             image_path=image_path,
//             bg_mask=bg_mask,
//         )
//     )

//   cameras.sort(key=lambda x: x.image_name)

//   if llffhold > 0:
//     train_cams = [c for idx, c in enumerate(cameras) if idx % llffhold != 0]
//     test_cams = [c for idx, c in enumerate(cameras) if idx % llffhold == 0]
//   else:
//     train_cams = cameras
//     test_cams = []

//   colmap_xyz = torch.tensor(colmap_scene.points3D, dtype=torch.float32)
//   colmap_rgb = (
//       torch.tensor(colmap_scene.point3D_colors, dtype=torch.float32) / 255.0
//   )

//   if initialize_random:
//     num_pts = 100_000
//     aabb_scale = colmap_xyz.amax(dim=0) - colmap_xyz.amin(dim=0)
//     aabb_offset = colmap_xyz.mean(dim=0)
//     initial_xyz = (
//         torch.rand((num_pts, 3)) * 2.0 - 1.0
//     ) * aabb_scale + aabb_offset
//     initial_rgb = torch.tensor([[0.5, 0.5, 0.5]]).repeat(num_pts, 1)
//   else:
//     initial_xyz = colmap_xyz
//     initial_rgb = colmap_rgb

//   scene = gs_scene.Scene(model_path)
//   scene.initialize(
//       initial_xyz=initial_xyz,
//       initial_rgb=initial_rgb,
//       train_cameras=train_cams,
//       test_cameras=test_cams,
//   )

//   return scene
// }

// fn retrieve_checkpoint(model_path: &str, load_iter: i32) -> String {
//   """Retrieves the path to a checkpoint.

//   Args:
//     model_path: The model path specified during the original training run.
//     load_iter: The iteration number wish to be loaded.

//   Returns:
//     If the user has specified one then it returns the corresponding path.
//     If the user hasn't specified one then it returns the most recent.
//     If none of them exist then returns None
//   """
//   ckpt_path = os.path.join(model_path, "checkpoint")
//   if load_iter:
//     print(f"Trying to load from args.load_iter={load_iter}")
//     load_path = os.path.join(ckpt_path, f"iteration_{load_iter}")
//     assert gfile.Exists(load_path), f"Checkpoint {load_path} does not exist"
//   else:
//     print(f"Looking for the latest checkpoint in {ckpt_path}")
//     folder_list = gfile.ListDir(ckpt_path)
//     saved_iters = [int(ckpt_name.split("_")[-1]) for ckpt_name in folder_list]
//     if saved_iters:
//       load_path = os.path.join(ckpt_path, f"iteration_{max(saved_iters)}")
//       print(f"Found checkpoint: {load_path}")
//     else:
//       load_path = ""
//       print("No checkpoint found!")

//   load_path
// }

// fn read_scene(
//     scene_path: str,
//     model_path: str,
//     llffhold: int = -1,
//     resolution: int = 1,
//     load_iter: int = -1,
//     initialize_colmap_random: bool = False,
// ) -> scene::Scene {
//   """Reads a scene.

//   The scene is either from loaded from a checkpoint or is freshly initialized
//   from a dataset.

//   Args:
//     scene_path: Path from a multiview dataset.
//     model_path: Path that the model is saved to and/or loaded from.
//     llffhold: In colmap datasets this parameter is used to split the train/test
//       cameras. When this value is positive and a colmap dataset is loaded we
//       assign every Nth camera to the test set where N=llfhold.
//     resolution: Assumes it is a downscale factor if it is [1, 2, 4, 8]. Anything
//       else is assumed to be the target resolution of the width of the image. If
//       nothing is specified we downsample to 1600 in case the image has higher
//       resolution. IMPORTANT: This applies only to colmap datasets.
//     load_iter: If load_iter>0 we try to load a checkpoint from that specified
//       iteration if we find it.
//     initialize_colmap_random: Initializes colmap scenes with random points.

//   Returns:
//     The loaded scene.
//   """
//   print(f"Input Directory: {scene_path}")
//   ckpt_path = retrieve_checkpoint(model_path, load_iter)
//   if ckpt_path:
//     # Load from checkpoint.
//     scene = gs_scene.Scene(model_path)
//     scene.load_checkpoint(ckpt_path)
//   else:
//     # Load from scene_path and initialize from colmap/random
//     if gfile.Exists(os.path.join(scene_path, "sparse")):
//       scene = read_colmap_scene(
//           scene_path, model_path, resolution, llffhold, initialize_colmap_random
//       )
//     elif gfile.Exists(os.path.join(scene_path, "transforms_train.json")):
//       scene = read_nerf_synthetic_scene(scene_path, model_path, resolution)
//     else:
//       panic!(f"Could recognize a scene in {scene_path}")

//     scene
// }
