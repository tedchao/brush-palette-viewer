#import helpers;

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;

@group(0) @binding(1) var<storage, read> means: array<vec4f>;
@group(0) @binding(2) var<storage, read> scales: array<vec4f>;
@group(0) @binding(3) var<storage, read> quats: array<vec4f>;

@group(0) @binding(4) var<storage, read_write> xys: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> depths: array<f32>;
@group(0) @binding(6) var<storage, read_write> radii: array<u32>;
@group(0) @binding(7) var<storage, read_write> cov2ds: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> num_tiles_hit: array<u32>;

struct Uniforms {
    // View matrix transform world to view position.
    viewmat: mat4x4f,
    // Focal (fx, fy).
    focal: vec2f,
    // Camera center (cx, cy).
    pixel_center: vec2f,
    // Img resolution (w, h)
    img_size: vec2u,
    // Nr. of tiles on each axis (w, h)
    tile_bounds: vec2u,
    // Width of blocks image is divided into.
    block_width: u32,
    // Near clip threshold.
    clip_thresh: f32,
}

fn project_pix(fxfy: vec2f, p_view: vec3f, pp: vec2f) -> vec2f {
    let p_proj = p_view.xy / max(p_view.z, 1e-6f);
    return p_proj * fxfy + pp;
}

// Kernel function for projecting gaussians.
// Each thread processes one gaussian
@compute
@workgroup_size(helpers::SPLATS_PER_GROUP, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;
    let num_points = arrayLength(&means);

    if idx >= num_points {
        return;
    }

    // 0 buffer to mark gaussian as not visible.
    radii[idx] = 0u;
    // Zero out number of tiles hit before cumulative sum.
    num_tiles_hit[idx] = 0u;

    let viewmat = uniforms.viewmat;
    let focal = uniforms.focal;
    let pixel_center = uniforms.pixel_center;

    let clip_thresh = uniforms.clip_thresh;

    // Project world space to camera space.
    let p_world = means[idx].xyz;

    let W = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let p_view = W * p_world + viewmat[3].xyz;

    if p_view.z <= clip_thresh {
        return;
    }

    // compute the projected covariance
    let scale = scales[idx].xyz;
    let quat = quats[idx];

    let R = helpers::quat_to_rotmat(quat);
    let S = helpers::scale_to_mat(scale);
    let M = R * S;
    let V = M * transpose(M);
    
    let tan_fov = 0.5 * vec2f(uniforms.img_size.xy) / focal;
    
    let lims = 1.3 * tan_fov;
    // Get ndc coords +- clipped to the frustum.
    let t = p_view.z * clamp(p_view.xy / p_view.z, -lims, lims);
    let rz = 1.0 / p_view.z;
    let rz2 = rz * rz;

    let J = mat3x3f(
        vec3f(focal.x * rz, 0.0, 0.0),
        vec3f(0.0, focal.y * rz, 0.0),
        vec3f(-focal.x * t.x * rz2, -focal.y * t.y * rz2, 0.0)
    );

    let T = J * W;
    let cov = T * V * transpose(T);

    let c00 = cov[0][0];
    let c11 = cov[1][1];
    let c01 = cov[0][1];

    // add a little blur along axes and save upper triangular elements
    let cov2d = vec3f(c00 + helpers::COV_BLUR, c01, c11 + helpers::COV_BLUR);
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;

    // inverse of 2x2 cov2d matrix
    let b = 0.5 * (cov2d.x + cov2d.z);
    let v1 = b + sqrt(max(0.1f, b * b - det));
    let v2 = b - sqrt(max(0.1f, b * b - det));

    // Render 3 sigma of covariance.
    // TODO: Make this a setting? Could save a good amount of pixels
    // that need to be rendered. Also: pick gaussians that REALLY clip to 0.
    // TODO: Is rounding down better? Eg. for gaussians <pixel size, just skip?
    let radius = u32(ceil(3.0 * sqrt(max(0.0, max(v1, v2)))));

    if radius == 0u {
        // point bbox outside of bounds.
        return;
    }

    // compute the projected mean
    let center = project_pix(focal, p_view, pixel_center);
    let tile_minmax = helpers::get_tile_bbox(center, radius, uniforms.tile_bounds);
    let tile_area = (tile_minmax.z - tile_minmax.x) * (tile_minmax.w - tile_minmax.y);

    // Now write all the data to the buffers.
    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = radius;
    xys[idx] = center;
    cov2ds[idx] = vec4f(cov2d, 1.0);
}
