#import helpers;

@group(0) @binding(0) var<storage, read> means: array<vec4f>;
@group(0) @binding(1) var<storage, read> scales: array<vec4f>;
@group(0) @binding(2) var<storage, read> quats: array<vec4f>;

@group(0) @binding(3) var<storage, read_write> xys: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> depths: array<f32>;
@group(0) @binding(5) var<storage, read_write> radii: array<i32>;
@group(0) @binding(6) var<storage, read_write> conics: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> compensation: array<f32>;
@group(0) @binding(8) var<storage, read_write> num_tiles_hit: array<i32>;

@group(0) @binding(9) var<storage, read> info_array: array<Uniforms>;

struct Uniforms {
    // Number of splats that exist.
    num_points: u32,
    // View matrix transform world to view position.
    viewmat: mat4x4f,
    focal: vec2f,
    pixel_center: vec2f,
    // Img resolution (w, h)
    img_size: vec2u,
    // Total reachable pixels (w, h)
    tile_bounds: vec2u,
    // Near clip threshold.
    clip_thresh: f32,
    // Width of blocks image is divided into.
    block_width: u32,
}

fn project_pix(fxfy: vec2f, p_view: vec3f, pp: vec2f) -> vec2f {
    let p_proj = p_view.xy / max(p_view.z, 1e-6f);
    return p_proj * fxfy + pp;
}

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
@compute
@workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;
    let info = info_array[0];
    let num_points = info.num_points;

    if idx >= num_points {
        return;
    }

    let viewmat = info.viewmat;
    let focal = info.focal;
    let pixel_center = info.pixel_center;

    let img_size = info.img_size;
    let tile_bounds = info.tile_bounds;
    let block_width = info.block_width;
    let clip_thresh = info.clip_thresh;

    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    // Project world space to camera space.
    let mean = means[idx].xyz;
    let W = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    
    let p_view = W * mean + viewmat[3].xyz;

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
    
    let tan_fov = 0.5 * vec2f(img_size.xy) / focal;
    
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

    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.

    // Blur the 2D covariance a bit to prevent blowup for small
    // gaussians.
    // TODO: Is this 0.3 good? Make it configurable?
    let cov2d = vec3f(c00 + 0.3f, c01, c11 + 0.3f);
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;

    if abs(det) < 1e-6 {
        // Cull this gaussian (leaves radii as 0).
        return;
    }

    // inverse of 2x2 cov2d matrix
    let conic = vec3f(cov2d.z, -cov2d.y, cov2d.x) / det;
    let b = 0.5 * (cov2d.x + cov2d.z);
    let v1 = b + sqrt(max(0.1f, b * b - det));
    let v2 = b - sqrt(max(0.1f, b * b - det));

    // Render 3 sigma of covariance.
    // TODO: Make this a setting? Could save a good amount of pixels
    // Touched. Or even better: pick gaussians that REALLY clip to 0.
    // TODO: Is rounding down better?
    let radius = i32(ceil(3.0 * sqrt(max(0.0, max(v1, v2)))));

    // compute the projected mean
    let center = project_pix(focal, p_view, pixel_center);
    let tile_minmax = helpers::get_tile_bbox(center, radius, tile_bounds, block_width);
    let tile_area = (tile_minmax.z - tile_minmax.x) * (tile_minmax.w - tile_minmax.y);

    // TODO: If radius is >0 at all, then tile_area > 0 right?
    // What is this check for?
    if tile_area <= 0 {
        return;
    }

    // Now write all the data to the buffers.
    num_tiles_hit[idx] = i32(tile_area);
    depths[idx] = p_view.z;
    radii[idx] = radius;
    xys[idx] = center;

    conics[idx] = vec4f(conic, 1.0);

    // Add a little blur along axes and save upper triangular elements
    // and compute the density compensation factor due to the blurs.
    let det_orig = c00 * c11 - c01 * c01;
    let det_blur = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    compensation[idx] = sqrt(max(0.0, det_orig / det_blur));
}