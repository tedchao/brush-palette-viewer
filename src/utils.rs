// Utilities to go from ndarray -> image and the other way around.
use burn::tensor::{backend::Backend, Tensor};

pub fn to_rerun_tensor<B: Backend, const D: usize>(t: Tensor<B, D>) -> rerun::TensorData {
    rerun::TensorData::new(
        t.dims()
            .map(|x| rerun::TensorDimension::unnamed(x as u64))
            .to_vec(),
        rerun::TensorBuffer::F32(t.into_data().convert().value.into()),
    )
}

#[allow(dead_code)]
// Assume 0-1, unlike rerun which always normalizes the image.
pub fn to_rerun_image<B: Backend>(t: Tensor<B, 3>) -> rerun::Image {
    let t_quant = (t * 255.0).int().clamp(0, 255);

    rerun::Image::new(rerun::TensorData::new(
        t_quant
            .dims()
            .map(|x| rerun::TensorDimension::unnamed(x as u64))
            .to_vec(),
        rerun::TensorBuffer::I8(t_quant.into_data().convert().value.into()),
    ))
}

// fn lookat_to_rt(
//     eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor
// ) -> tuple[torch.Tensor, torch.Tensor]:
//   """Returns the rotation and translation from look-at system of vectors."""
//   look_dir = target - eye
//   look_dir = look_dir / look_dir.norm()
//   right = torch.cross(up, look_dir)
//   right = right / right.norm()
//   up = torch.cross(look_dir, right)
//   up = up / up.norm()

//   tx = -torch.dot(eye, right)
//   ty = -torch.dot(eye, up)
//   tz = -torch.dot(eye, look_dir)

//   rotation = torch.tensor([
//       [right[0], right[1], right[2]],
//       [up[0], up[1], up[2]],
//       [look_dir[0], look_dir[1], look_dir[2]],
//   ])
//   translation = torch.tensor([tx, ty, tz])

//   return rotation, translation

pub fn inverse_sigmoid<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    Tensor::log(x / (x.neg() + 1.0))
}

// // Converts a quaternion to a rotation matrix
// fn qvec2rotmat<B>(r: Tensor<B>) -> Tensor<B> {
//   let q = r / r.norm(dim=-1, keepdim=True);

//   rot_mat = torch.zeros(q.shape[:-1] + (3, 3), device=q.device)

//   r = q[..., 0]
//   x = q[..., 1]
//   y = q[..., 2]
//   z = q[..., 3]

//   rot_mat[..., 0, 0] = 1 - 2 * (y * y + z * z)
//   rot_mat[..., 0, 1] = 2 * (x * y - r * z)
//   rot_mat[..., 0, 2] = 2 * (x * z + r * y)
//   rot_mat[..., 1, 0] = 2 * (x * y + r * z)
//   rot_mat[..., 1, 1] = 1 - 2 * (x * x + z * z)
//   rot_mat[..., 1, 2] = 2 * (y * z - r * x)
//   rot_mat[..., 2, 0] = 2 * (x * z - r * y)
//   rot_mat[..., 2, 1] = 2 * (y * z + r * x)
//   rot_mat[..., 2, 2] = 1 - 2 * (x * x + y * y)

//   return rot_mat
// }

// def average_sqd_knn(xyz: torch.Tensor, k: int = 4) -> torch.Tensor:
//   """Returns the average square distance of k nearest neighbors."""
//   return torch.tensor(
//       (
//           neighbors.NearestNeighbors(n_neighbors=k)
//           .fit(xyz)
//           .kneighbors(xyz)[0][:, 1:]
//           ** 2
//       ).mean(axis=1)
//   )

// def downscale_pil_image(
//     user_resolution: int, pil_image: Image.Image
// ) -> Image.Image:
//   """Downscale a PIL image to a given resolution.

//   Args:
//     user_resolution: Assumes it is a downscale factor if it is [1, 2, 4, 8].
//       Anything else is assumed to be the target resolution of the width of the
//       image. If nothing is specified we downsample to 1600 in case the image has
//       higher resolution.
//     pil_image: The image to be downscaled.

//   Returns:
//     Downscaled pil.Image
//   """
//   orig_w, orig_h = pil_image.size

//   if user_resolution <= 0 or user_resolution is None:
//     if orig_w > 1600:
//       # User didn't specify a resolution and resolution >1600
//       scale_factor = 1600 / orig_w
//     else:
//       scale_factor = 1
//   else:
//     if user_resolution in [1, 2, 4, 8]:
//       # The user specified a downscale factor
//       scale_factor = 1 / user_resolution
//     else:
//       # The user specified a specific resolution
//       scale_factor = user_resolution / orig_w

//   target_resolution = (int(orig_w * scale_factor), int(orig_h * scale_factor))
//   return pil_image.resize(target_resolution)

// def world2view_from_rotation_translation(
//     rotation: torch.Tensor, translation: torch.Tensor
// ) -> torch.Tensor:
//   """Constructs a world to view matrix from rotation and translation."""
//   rt = torch.zeros((4, 4))
//   rt[:3, :3] = rotation
//   rt[:3, 3] = translation
//   rt[3, 3] = 1.0
//   return rt

// def get_camera_center(world_view_transform: torch.Tensor) -> torch.Tensor:
//   view_inv = torch.inverse(world_view_transform)
//   return view_inv[:3, 3]

// def get_projection_matrix(
//     znear: float, zfar: float, fovx: float, fovy: float
// ) -> torch.Tensor:
//   """Constructs a projection matrix from znear, zfar, fovx, fovy."""
//   top = math.tan((fovy / 2)) * znear
//   bottom = -top
//   right = math.tan((fovx / 2)) * znear
//   left = -right

//   proj_mat = torch.zeros([4, 4])

//   z_sign = 1.0

//   proj_mat[0, 0] = 2.0 * znear / (right - left)
//   proj_mat[1, 1] = 2.0 * znear / (top - bottom)
//   proj_mat[0, 2] = (right + left) / (right - left)
//   proj_mat[1, 2] = (top + bottom) / (top - bottom)
//   proj_mat[3, 2] = z_sign
//   proj_mat[2, 2] = z_sign * zfar / (zfar - znear)
//   proj_mat[2, 3] = -(zfar * znear) / (zfar - znear)

//   return proj_mat.type(torch.float32)

// def fov_to_focal(fov: float, pixels: int) -> float:
//   """Converts field of view to focal length."""
//   return pixels / (2 * math.tan(fov / 2))

// def focal_to_fov(focal: float, pixels: int) -> float:
//   """Converts focal length to field of view."""
//   return 2 * math.atan(pixels / (2 * focal))

// def project_points(
//     points: torch.Tensor, transf_matrix: torch.Tensor
// ) -> torch.Tensor:
//   """Projects points to NDC with a given P@V matrix."""

//   n_points, _ = points.shape
//   ones = torch.ones(n_points, 1, dtype=points.dtype, device=points.device)
//   points_hom = torch.cat([points, ones], dim=1)
//   points_out = (transf_matrix @ points_hom[..., None]).squeeze(-1)

//   denom = points_out[..., 3:] + 0.0000001
//   return (points_out[..., :3] / denom).squeeze(dim=0)

// class ExponentialDecayLr:
//   """Adapted from Plenoxels that adapted it from JaxNeRF.

//   The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
//   is log-linearly interpolated elsewhere (equivalent to exponential decay).
//   If lr_delay_steps>0 then the learning rate will be scaled by some smooth
//   function of lr_delay_mult, such that the initial learning rate is
//   lr_init*lr_delay_mult at the beginning of optimization but will be eased back
//   to the normal learning rate when steps>lr_delay_steps.
//   """

//   def __init__(
//       self,
//       lr_init: float,
//       lr_final: float,
//       lr_delay_steps: float = 0,
//       lr_delay_mult: float = 1.0,
//       max_steps: int = 1000000,
//   ):
//     self.lr_init = lr_init
//     self.lr_final = lr_final
//     self.lr_delay_steps = lr_delay_steps
//     self.lr_delay_mult = lr_delay_mult
//     self.max_steps = max_steps

//   def __call__(self, step: int) -> float:
//     if step < 0 or (self.lr_init == 0.0 and self.lr_final == 0.0):
//       # Disable this parameter
//       return 0.0
//     if self.lr_delay_steps > 0:
//       # A kind of reverse cosine decay.
//       delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
//           0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
//       )
//     else:
//       delay_rate = 1.0
//     t = np.clip(step / self.max_steps, 0, 1)
//     log_lerp = np.exp(
//         np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t
//     )
//     return delay_rate * log_lerp
