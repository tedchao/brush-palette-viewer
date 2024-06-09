const TILE_WIDTH: u32 = 16u;
const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;

fn get_bbox(center: vec2f, dims: vec2f, bounds: vec2u) -> vec4u {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    let min = vec2u(clamp(vec2i(center - dims), vec2i(0), vec2i(bounds)));
    let max = vec2u(clamp(vec2i(center + dims + vec2f(1.0)), vec2i(0), vec2i(bounds)));
    return vec4u(min, max);
}

fn get_tile_bbox(pix_center: vec2f, pix_radius: u32, tile_bounds: vec2u) -> vec4u {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    let tile_center = pix_center / f32(TILE_WIDTH);
    let tile_radius = f32(pix_radius) / f32(TILE_WIDTH);

    return get_bbox(tile_center, vec2f(tile_radius, tile_radius), tile_bounds);
}

// device helper to get 3D covariance from scale and quat parameters
fn quat_to_rotmat(quat: vec4f) -> mat3x3f {
    // quat to rotation matrix
    let w = quat.x;
    let x = quat.y;
    let y = quat.z;
    let z = quat.w;

    // See https://www.songho.ca/opengl/gl_quaternion.html
    return mat3x3f(
        vec3f(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        vec3f(
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ),
        vec3f(
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y)
        ),
    );
}

fn scale_to_mat(scale: vec3f) -> mat3x3f {
    return mat3x3(
        vec3f(scale.x, 0.0, 0.0),
        vec3f(0.0, scale.y, 0.0), 
        vec3f(0.0, 0.0, scale.z)
    );
}

// TODO: Is this 0.3 good? Make this configurable?
const COV_BLUR: f32 = 0.3;

fn cov_compensation(cov2d: vec3f) -> f32 {
    let cov_orig = cov2d - vec3f(COV_BLUR, 0.0, COV_BLUR);
    let det_orig = cov_orig.x * cov_orig.z - cov_orig.y * cov_orig.y;
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    return sqrt(max(0.0, det_orig / det));
}

fn calc_sigma(pixel_coord: vec2f, conic: vec3f, xy: vec2f) -> f32 {
    let delta = pixel_coord - xy;
    return 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
}

fn calc_vis(pixel_coord: vec2f, conic: vec3f, xy: vec2f) -> f32 {
    return exp(-calc_sigma(pixel_coord, conic, xy));
}

fn inverse(m: mat2x2f) -> mat2x2f {
    let det = determinant(m);
    return mat2x2f(
        m[1][1] / det, -m[1][0] / det, 
        -m[0][1] / det, m[0][0] / det
    );
}

fn simple_sign(x: f32) -> f32 {
    if x >= 0.0 {
        return 1.0;
    }
    return -1.0;
}

// Adopted method from: https://www.geometrictools.com/Documentation/IntersectionRectangleEllipse.pdf
// The pseudocode is a bit broken in their paper, but was adapted to this
// implementation which seems to work.
// TODO: This still has some false positives!
fn ellipse_rect_intersect(Rc: vec2f, Re: vec2f, Ec: vec2f, Em: mat2x2f) -> bool {
    // Compute the increase in extents for R’.
    let L = sqrt(vec2f(Em[1][1], Em[0][0]) / determinant(Em));

    // Transform the ellipse center to rectangle coordinate system.
    let KmC = Ec - Rc;

    // Figure out extends of ellipse + rectangle.
    let extended = Re + L;

    // outside total bounding box.
    if abs(KmC.x) > extended.x || abs(KmC.y) > extended.y {
        return false;
    }

    // Check if point is outside any of the four corners.
    let s = vec2f(simple_sign(KmC.x), simple_sign(KmC.y));
    let delta0 = KmC - s * Re;
    let EmDelta0 = Em * delta0;
    return s.x * EmDelta0.x <= 0.0 || s.y * EmDelta0.y <= 0.0 || dot(delta0, EmDelta0) <= 1.0;
}

fn can_be_visible(tile: vec2u, xy: vec2f, conic: vec3f, opac: f32) -> bool {
    let tile_extent = vec2f(f32(TILE_WIDTH));
    let tile_center = vec2f(tile * TILE_WIDTH) + tile_extent;
    
    // opac * exp(-sigma) < 1.0 / 255.0
    // exp(-sigma) < 1.0 / (opac * 255.0)
    // -sigma < log(1.0 / (opac * 255.0))
    // sigma < log(opac * 255.0);
    let rads = log(opac * 255.0);
    let conic_scaled = conic / (2.0 * rads);
    return ellipse_rect_intersect(tile_center, tile_extent, xy, mat2x2f(conic_scaled.x, conic_scaled.y, conic_scaled.y, conic_scaled.z));
}

fn ceil_div(a: u32, b: u32) -> u32 {
    return (a + b - 1) / b;
}