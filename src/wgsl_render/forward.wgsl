 // TODO: No cooperative groups in WGSL?

// TODO: How to represent SH really.
struct ShData {
	c0: vec3f,
}

const SH_C0: f64 = 0.28209479177387814;

@binding(0) @group(0) var<storage> clamped: array<bool>;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
fn compute_color_from_sh(idx: i32, deg: i32, max_coeffs: i32, means: array<vec3f>, campos: vec3f, shs: array<ShData>) -> vec3f {
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	let pos = means[idx];
	let dir = normalize(pos - campos);
	
	let sh = shs[idx];
	var result = SH_C0 * sh.c0;

	// if (deg > 0)
	// {
	// 	float x = dir.x;
	// 	float y = dir.y;
	// 	float z = dir.z;
	// 	result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

	// 	if (deg > 1)
	// 	{
	// 		float xx = x * x, yy = y * y, zz = z * z;
	// 		float xy = x * y, yz = y * z, xz = x * z;
	// 		result = result +
	// 			SH_C2[0] * xy * sh[4] +
	// 			SH_C2[1] * yz * sh[5] +
	// 			SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
	// 			SH_C2[3] * xz * sh[7] +
	// 			SH_C2[4] * (xx - yy) * sh[8];

	// 		if (deg > 2)
	// 		{
	// 			result = result +
	// 				SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
	// 				SH_C3[1] * xy * z * sh[10] +
	// 				SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
	// 				SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
	// 				SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
	// 				SH_C3[5] * z * (xx - yy) * sh[14] +
	// 				SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
	// 		}
	// 	}
	// }
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.

	// TODO: What is this??
	// TODO: How to NOT do this when we're just rendering not training.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return max(result, 0.0);
}

// Forward version of 2D covariance matrix computation
fn computeCov2D(mean: vec3f, focal_x: f32, focal_y: f32, tan_fovx: f32, tan_fovy: f32, cov3D: array<f32>, viewmatrix: mat4x4f) -> vec3f {
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	var t = viewmatrix * vec4f(mean, 1.0);

	// TODO: What is this 1.3 then?
	let limx = 1.3 * tan_fovx;
	let limy = 1.3 * tan_fovy;

	let txtz = t.x / t.z;
	let tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	let J = mat3x3f(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	let W = mat3x3f(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	let T = W * J;

	let Vrk = mat3x3f(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	let cov = transpose(T) * transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return vec3f(float(cov[0][0]), float(cov[0][1]), float(cov[1][1]));
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
fn computeCov3D(scale: vec4f, scale_mod: f32, rot: vec4f, cov3D: array<f32>) {
	// Create scaling matrix
	let S = mat3x3(1.0f);
	S[0][0] = scale_mod * scale.x;
	S[1][1] = scale_mod * scale.y;
	S[2][2] = scale_mod * scale.z;

	// Normalize quaternion to get valid rotation

	// TODO: Hunt down this normalization business.
	let q = rot;// / glm::length(rot);
	let r = q.x;
	let x = q.y;
	let y = q.z;
	let z = q.w;

	// Compute rotation matrix from quaternion
	let R = mat3f(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	let M = S * R;

	// Compute 3D world covariance matrix Sigma
	let Sigma = transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
fn preprocessCUDA(
	P: i32, 
	D: i32, 
	M: i32,
	orig_points: array<f32>,
	scales: array<vec3f>,
	scale_modifier: array<f32>,
	rotations: array<vec4f>,
	opacities: array<f32>,
	shs: array<ShData>,
	clamped: array<bool>,
	viewmatrix: mat4x4f,
	projmatrix: mat4x4f,
	cam_pos: vec3f,
	W: i32, 
	H: i32,
	tan_fovx: f32,  tan_fovy: f32,
	focal_x: f32, focal_y: f32,
	radii: array<i32>,
	points_xy_image: array<vec2f>,
	depths: array<f32>,
	cov3Ds: array<f32>,
	rgb: array<f32>,
	conic_opacity: array<vec4f>,
	grid: vec3i,
	tiles_touched: array<u32>,
	prefiltered: bool,
	rects: array<vec2i>,
	boxmin: vec3f,
	boxmax: vec3f,
	@builtin(global_invocation_id) global_id: vec3<u32>,
	@builtin(local_invocation_index) local_idx: u32,
	@builtin(workgroup_id) workgroup_id: vec3<u32>,
	)
{
	let idx = global_id.x;
	if (idx >= P) {
		return;
	}

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	let p_view: vec3f = 0.0;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view)) {
		return;
	}

	// Transform point by projecting
	// TODO: Orig points can be a vec3 surely.
	let p_orig = vec3f(orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);

	if (p_orig.x < boxmin.x || p_orig.y < boxmin.y || p_orig.z < boxmin.z ||
		p_orig.x > boxmax.x || p_orig.y > boxmax.y || p_orig.z > boxmax.z) {
		return;
	}

	let p_hom = transformPoint4x4(p_orig, projmatrix);
	let p_w = 1.0f / (p_hom.w + 0.0000001f);
	let p_proj = vec4f(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w);

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 

	// TODO: urgh this pointer indexing.
	computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
	cov3D = cov3Ds + idx * 6;

	// Compute 2D screen-space covariance matrix
	let cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	let det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f) {
		return;
	}
	let det_inv = 1.f / det;
	let conic = vec3f(cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv);

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 

	let mid = 0.5f * (cov.x + cov.z);
	let lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	let lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	let my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	let point_image = vec2f(ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H));

	// Slightly more aggressive, might need a math cleanup
	let my_rect = vec2i(ceil(3.f * sqrt(cov.x)), ceil(3.f * sqrt(cov.z)));
	rects[idx] = my_rect;
	getRect(point_image, my_rect, rect_min, rect_max, grid);

	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) {
		return;
	}

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	let result = compute_color_from_sh(idx, D, M, orig_points, cam_pos, shs, clamped);
	rgb[idx * C + 0] = result.x;
	rgb[idx * C + 1] = result.y;
	rgb[idx * C + 2] = result.z;

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = vec4f(conic.x, conic.y, conic.z, opacities[idx]);
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}




// Create workgroup data for renderCUDA
const BLOCK_SIZE : u32 = 8;
var<workgroup> workgroup_data: array<u32, workgroup_len>;


// Allocate storage for batches of collectively fetched data.
var<workgroup> collected_id: array<i32, BLOCK_SIZE>;
var<workgroup> collected_xy: array<vec2f, BLOCK_SIZE>;
var<workgroup> collected_conic_opacity: array<vec4f, BLOCK_SIZE>;

// TODO: Actual values.
const BLOCK_X: u32 = 8;
const BLOCK_Y: u32 = 8;

const W: u32 = 32;
const H: u32 = 32;

var<workgroup> count_done: atomic<u32>;

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
fn renderCUDA(
	ranges: array<vec2u>,
	point_list: array<u32>,
	W: i32, H: i32,
	points_xy_image: array<vec2f>,
	features: array<f32>,
	conic_opacity: array<vec4f>,
	final_T: array<f32>,
	n_contrib: array<u32>,
	bg_color: array<f32>,
	out_color: array<f32>,
	
	@builtin(global_invocation_id) global_id: vec3<u32>,
	@builtin(local_invocation_index) local_idx: u32,
	@builtin(workgroup_id) workgroup_id: vec3<u32>,
)
{
	// Identify current tile and associated min/max pixel range.

	// TODO: I suspect this can be simplified a bit with global/local/group idx.
	let horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	let pix_min = vec2u(workgroup_id.x * BLOCK_X, workgroup_id.y * BLOCK_Y);
	let pix_max = vec2u(min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H));
	let pix = vec2f(pix_min.x + local_idx.x, pix_min.y + local_idx.y);
	let pix_id = W * pix.y + pix.x;
	let pixf = vec2f(pix.x, pix.y);

	// Check if this thread is associated with a valid pixel or outside.
	let inside = (pix.x < W && pix.y < H);

	// Done threads can help with fetching, but don't rasterize.
	// TODO: Or just contribute to memory contention?
	let done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	let range = ranges[workgroup_id.y * horizontal_blocks + workgroup_id.x];
	let rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	let toDo = range.y - range.x;

	// Initialize helper variables
	let T = 1.0f;
	let contributor = 0;
	let last_contributor = 0;
	// let C[CHANNELS] = 0.0;

	// Iterate over batches until all done or range is complete
	toDo += BLOCK_SIZE; // stupic hack :)
	for (let i = 0; i < rounds; i++) {
		toDo -= BLOCK_SIZE;

		// End if entire block votes that it is done rasterizing

		// TODO: Double check that's atomic.
		if (count_done == BLOCK_SIZE) {
			break;
		}

		// Collectively fetch per-Gaussian data from global to shared
		let progress = i * BLOCK_SIZE + local_idx.x;

		if (range.x + progress < range.y) {
			let coll_id = point_list[range.x + progress];
			collected_id[local_idx.x] = coll_id;
			collected_xy[local_idx.x] = points_xy_image[coll_id];
			collected_conic_opacity[local_idx.x] = conic_opacity[coll_id];
		}

		workgroupBarrier();

		// Iterate over current batch
		for (let j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			let xy = collected_xy[j];
			let d = vec2f(xy.x - pixf.x, xy.y - pixf.y);
			let con_o = collected_conic_opacity[j];
			let power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			
			if (power > 0.0f) {
				continue;
			}

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			let alpha = min(0.99f, con_o.w * exp(power));

			if (alpha < 1.0f / 255.0f) {
				continue;
			}

			let test_T = T * (1 - alpha);

			if (test_T < 0.0001f) {
				done = true;
				atomicAdd(&count_done, 1u);
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (let ch = 0; ch < CHANNELS; ch++) {
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			}

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (let ch = 0; ch < CHANNELS; ch++) {
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
	}
}
