#import helpers;

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;

@group(0) @binding(1) var<storage, read> means: array<f32>; // packed vec3
@group(0) @binding(2) var<storage, read> log_scales: array<f32>; // packed vec3
@group(0) @binding(3) var<storage, read> quats: array<vec4f>;

@group(0) @binding(4) var<storage, read_write> global_from_compact_gid: array<u32>;

@group(0) @binding(5) var<storage, read_write> xys: array<vec2f>;
@group(0) @binding(6) var<storage, read_write> conic_comps: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> depths: array<f32>;

@group(0) @binding(8) var<storage, read_write> num_visible: atomic<u32>;


struct Uniforms {
    // View matrix transform world to view position.
    viewmat: mat4x4f,
    // Focal (fx, fy).
    focal: vec2f,
    // Camera center (cx, cy).
    pixel_center: vec2f,
    // Img resolution (w, h)
    img_size: vec4u,
}

fn project_pix(fxfy: vec2f, p_view: vec3f, pp: vec2f) -> vec2f {
    let p_proj = p_view.xy / (p_view.z + 1e-6f);
    return p_proj * fxfy + pp;
}

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let global_gid = global_id.x;

    if global_gid >= arrayLength(&quats) {
        return;
    }

    let viewmat = uniforms.viewmat;
    let focal = uniforms.focal;
    let pixel_center = uniforms.pixel_center;

    // Project world space to camera space.
    let mean = vec3f(means[global_gid * 3 + 0], means[global_gid * 3 + 1], means[global_gid * 3 + 2]);
    let tile_bounds = vec2u(helpers::ceil_div(uniforms.img_size.x, helpers::TILE_WIDTH), helpers::ceil_div(uniforms.img_size.y, helpers::TILE_WIDTH));

    let W = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let p_view = W * mean + viewmat[3].xyz;

    if p_view.z <= 0.01 {
        return;
    }

    // compute the projected covariance
    let scale = exp(vec3f(log_scales[global_gid * 3 + 0], log_scales[global_gid * 3 + 1], log_scales[global_gid * 3 + 2]));
    let quat = quats[global_gid];

    let tan_fov = 0.5 * vec2f(uniforms.img_size.xy) / focal;
    let lims = 1.3 * tan_fov;
    // Get ndc coords +- clipped to the frustum.
    let t = p_view.z * clamp(p_view.xy / p_view.z, -lims, lims);
    
    var M = helpers::quat_to_rotmat(quat);
    M[0] *= scale.x;
    M[1] *= scale.y;
    M[2] *= scale.z;
    var V = M * transpose(M);

    let J = mat3x3f(
        vec3f(focal.x, 0.0, 0.0),
        vec3f(0.0, focal.y, 0.0),
        vec3f(-focal * t / p_view.z, 0.0)
    ) * (1.0 / p_view.z);

    let T = J * W;
    let cov = T * V * transpose(T);

    let c00 = cov[0][0] + helpers::COV_BLUR;
    let c11 = cov[1][1] + helpers::COV_BLUR;
    let c01 = cov[0][1];

    // add a little blur along axes and save upper triangular elements
    let cov2d = vec3f(c00, c01, c11);
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;

    if det == 0.0 {
        return;
    }

    // Calculate ellipse conic.
    let conic = vec3f(cov2d.z, -cov2d.y, cov2d.x) * (1.0 / det);
    // compute the projected mean
    let xy = project_pix(focal, p_view, pixel_center);

    let radius = helpers::radius_from_conic(conic, 1.0);

    let tile_minmax = helpers::get_tile_bbox(xy, radius, tile_bounds);
    let tile_min = tile_minmax.xy;
    let tile_max = tile_minmax.zw;

    if (tile_max.x - tile_min.x) == 0u || (tile_max.y - tile_min.y) == 0u {
        return;
    }

    // Now write all the data to the buffers.
    let write_id = atomicAdd(&num_visible, 1u);
    global_from_compact_gid[write_id] = global_gid;
    
    let comp = helpers::cov_compensation(cov2d);
    depths[write_id] = p_view.z;

    xys[write_id] = xy;
    conic_comps[write_id] = vec4f(conic, comp);
}
