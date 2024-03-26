#import helpers;

@group(0) @binding(0) var<storage, read> means3d: array<vec3f>;
@group(0) @binding(1) var<storage, read> scales: array<vec3f>;
@group(0) @binding(2) var<storage, read> quats: array<vec4f>;

@group(0) @binding(3) var<storage, read_write> covs3d: array<f32>;
@group(0) @binding(4) var<storage, read_write> xys: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> depths: array<f32>;
@group(0) @binding(6) var<storage, read_write> radii: array<i32>;
@group(0) @binding(7) var<storage, read_write> conics: array<vec3f>;
@group(0) @binding(8) var<storage, read_write> compensation: array<f32>;
@group(0) @binding(9) var<storage, read_write> num_tiles_hit: array<i32>;

@group(0) @binding(10) var<storage, read> info_array: array<helpers::InfoBinding>;

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
@compute
@workgroup_size(16, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    // Until burn supports adding in uniforms we read these from a tensor.
    let info = info_array[0];

    let idx = local_id.x;
    let num_points = info.num_points;

    if idx >= num_points {
        return;
    }

    let glob_scale = info.glob_scale;
    let viewmat = info.viewmat;
    let projmat = info.projmat;
    let intrins = info.intrins;

    let img_size = info.img_size;
    let tile_bounds = info.tile_bounds;
    let block_width = info.block_width;
    let clip_thresh = info.clip_thresh;

    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    let p_world = means3d[idx];

    // Project local space to
    let p_view = viewmat * vec4f(p_world, 1.0f);

    if p_view.z <= clip_thresh {
        return;
    }

    // compute the projected covariance
    let scale = scales[idx];
    let quat = quats[idx];

    let tmp = helpers::scale_rot_to_cov3d(scale, glob_scale, quat);

    // save upper right of matrix, as it's symmetric.
    // TODO: Does this match the original order? row vs column major?
    let covs0 = tmp[0][0];
    let covs1 = tmp[0][1];
    let covs2 = tmp[0][2];
    let covs3 = tmp[1][1];
    let covs4 = tmp[1][2];
    let covs5 = tmp[2][2];
    
    covs3d[6 * idx + 0] = covs0;
    covs3d[6 * idx + 1] = covs1;
    covs3d[6 * idx + 2] = covs2;
    covs3d[6 * idx + 3] = covs3;
    covs3d[6 * idx + 4] = covs4;
    covs3d[6 * idx + 5] = covs5;

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

    let t = p_world.z * clamp(p_world.xy / p_world.z, -lims, lims);
    let rz = 1.0f / p_world.z;
    let rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    let J = mat3x3f(
        vec3f(fx * rz, 0.0f, 0.0f,),
        vec3f(0.0f, fy * rz, 0.0f),
        vec3f(-fx * t.x * rz2, -fy * t.y * rz2, 0.0f)
    );

    let T = J * W;

    let V = mat3x3f(
        vec3f(covs0, covs1, covs2),
        vec3f(covs1, covs3, covs4),
        vec3f(covs2, covs4, covs5)
    );

    let cov = T * V * transpose(T);

    // add a little blur along axes and save upper triangular elements
    // and compute the density compensation factor due to the blurs
    let c00 = cov[0][0];
    let c11 = cov[1][1];
    let c01 = cov[0][1];
    let det_orig = c00 * c11 - c01 * c01;
    let cov2d = vec3f(c00 + 0.3f, c01, c11 + 0.3f);
    
    let det_blur = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    let comp = sqrt(max(0.f, det_orig / det_blur));

    let cov2d_bounds = helpers::compute_cov2d_bounds(cov2d);

    if !cov2d_bounds.valid {
        return; // zero determinant
    }
    conics[idx] = cov2d_bounds.conic;

    // compute the projected mean
    let center = helpers::project_pix(projmat, p_world, img_size, vec2f(cx, cy));

    let tile_minmax = helpers::get_tile_bbox(center, cov2d_bounds.radius, tile_bounds, block_width);
    let tile_min = tile_minmax.xy;
    let tile_max = tile_minmax.zw;
    let tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);

    if tile_area <= 0 {
        return;
    }

    num_tiles_hit[idx] = i32(tile_area);
    depths[idx] = p_view.z;
    radii[idx] = i32(cov2d_bounds.radius);
    xys[idx] = center;
    compensation[idx] = comp;
}