#import helpers;

@group(0) @binding(0) var<storage, read> means: array<vec4f>;
@group(0) @binding(1) var<storage, read> scales: array<vec4f>;
@group(0) @binding(2) var<storage, read> quats: array<vec4f>;

@group(0) @binding(3) var<storage, read_write> xys: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> depths: array<f32>;
@group(0) @binding(5) var<storage, read_write> radii: array<f32>;
@group(0) @binding(6) var<storage, read_write> conics: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> compensation: array<f32>;
@group(0) @binding(8) var<storage, read_write> num_tiles_hit: array<i32>;

@group(0) @binding(9) var<storage, read> info_array: array<Uniforms>;

struct Uniforms {
    // Number of splats that exist.
    num_points: u32,
    // View matrix transform world to view position.
    viewmat: mat4x4f,
    // fx, fy, cx, cy
    intrins: vec4f,
    // Img resolution (w, h)
    img_size: vec2u,
    // Total reachable pixels (w, h)
    tile_bounds: vec2u,
    // Global scale for blobs.
    glob_scale: f32,
    // Near clip threshold.
    clip_thresh: f32,
    // Width of blocks image is divided into.
    block_width: u32,
}

fn project_pix(fxfy: vec2f, p_view: vec3f, pp: vec2f) -> vec2f {
    let p_proj = p_view.xy / (p_view.z + 1e-6f);
    let p_pix = p_proj.xy * fxfy.xy + pp;
    return p_pix;
}

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
@compute
@workgroup_size(16, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    let idx = global_id.x;

    // Until burn supports adding in uniforms we read these from a tensor.
    let info = info_array[0];
    let num_points = info.num_points;

    if idx >= num_points {
        return;
    }

    let glob_scale = info.glob_scale;
    let viewmat = info.viewmat;
    let intrins = info.intrins;

    let img_size = info.img_size;
    let tile_bounds = info.tile_bounds;
    let block_width = info.block_width;
    let clip_thresh = info.clip_thresh;

    radii[idx] = 0.0;
    num_tiles_hit[idx] = 0;

    // Project world space to camera space.
    let mean = means[idx].xyz;

    let p_view_proj = viewmat * vec4f(mean, 1.0f);
    let p_view = p_view_proj.xyz / p_view_proj.w;

    if p_view.z <= clip_thresh {
        return;
    }

    // compute the projected covariance
    let scale = scales[idx];
    let quat = quats[idx];

    // save upper right of matrix, as it's symmetric.

    // let scaless = scale_to_mat(scale, glob_scale);
    let R = helpers::quat_to_rotmat(quat);
    // let S = scale_to_mat(scale, glob_scale);

    let scale_total = scale * glob_scale;
    let S = mat3x3(
        vec3f(scale_total.x, 0, 0),
        vec3f(0, scale_total.y, 0), 
        vec3f(0, 0, scale_total.z)
    );

    let M = R * S;
    let V = M * transpose(M);
    
    let fx = intrins.x;
    let fy = intrins.y;
    let cx = intrins.z;
    let cy = intrins.w;

    let tan_fovx = 0.5 * f32(img_size.x) / fx;
    let tan_fovy = 0.5 * f32(img_size.y) / fy;

    let W = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);

    // clip so that the covariance
    // TODO: What does that comment mean... :)
    let lims = 1.3f * vec2f(tan_fovx, tan_fovy);

    let t = p_view.z * clamp(p_view.xy / p_view.z, -lims, lims);
    let rz = 1.0f / p_view.z;
    let rz2 = rz * rz;

    let J = mat3x3f(
        vec3f(fx * rz, 0.0f, 0.0f,),
        vec3f(0.0f, fy * rz, 0.0f),
        vec3f(-fx * t.x * rz2, -fy * t.y * rz2, 0.0f)
    );

    let T = J * W;
    let cov = T * V * transpose(T);

    let c00 = cov[0][0];
    let c11 = cov[1][1];
    let c01 = cov[0][1];
    let cov2d = vec3f(c00 + 0.3f, c01, c11 + 0.3f);

    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;

    if det == 0.0f {
        return;
    }

    // inverse of 2x2 cov2d matrix
    let conic = vec3f(cov2d.z, -cov2d.y, cov2d.x) / det;

    let b = 0.5f * (cov2d.x + cov2d.z);
    let v1 = b + sqrt(max(0.1f, b * b - det));
    let v2 = b - sqrt(max(0.1f, b * b - det));
    
    // take 3 sigma of covariance
    let radius = ceil(3.0 * sqrt(max(v1, v2)));

    // compute the projected mean
    let pixel_center = project_pix(vec2f(fx, fy), p_view.xyz, vec2f(cx, cy));

    // TODO: Block_width? Tile width>
    let tile_minmax = helpers::get_tile_bbox(pixel_center, radius, tile_bounds, block_width);
    let tile_area = (tile_minmax.z - tile_minmax.x) * (tile_minmax.w - tile_minmax.y);

    if tile_area <= 0 {
        return;
    }

    // Now write all the data to the buffers.
    num_tiles_hit[idx] = i32(tile_area);
    depths[idx] = p_view.z;
    radii[idx] = radius;
    xys[idx] = pixel_center;

    conics[idx] = vec4f(conic, 1.0);

    // Add a little blur along axes and save upper triangular elements
    // and compute the density compensation factor due to the blurs.
    let det_orig = c00 * c11 - c01 * c01;
    let det_blur = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    compensation[idx] = sqrt(max(0.0f, det_orig / det_blur));
}