fn get_bbox(center: vec2f, dims: vec2f, bounds: vec2u) -> vec4u {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    let min = vec2u(clamp(vec2i(center - dims), vec2i(0), vec2i(bounds)));
    let max = vec2u(clamp(vec2i(center + dims + 1), vec2i(0), vec2i(bounds)));
    return vec4u(min, max);
}

fn get_tile_bbox(pix_center: vec2f, pix_radius: u32, tile_bounds: vec2u, block_size: u32) -> vec4u {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    let tile_center = pix_center / f32(block_size);
    let tile_radius = f32(pix_radius) / f32(block_size);

    return get_bbox(tile_center, vec2f(tile_radius, tile_radius), tile_bounds);
}

// device helper to get 3D covariance from scale and quat parameters
fn quat_to_rotmat(quat: vec4f) -> mat3x3f {
    // quat to rotation matrix
    let quat_norm = normalize(quat);

    let w = quat_norm.x;
    let x = quat_norm.y;
    let y = quat_norm.z;
    let z = quat_norm.w;

    // See https://www.songho.ca/opengl/gl_quaternion.html
    return mat3x3f(
        vec3f(
            1.f - 2.f * (y * y + z * z),
            2.f * (x * y + w * z),
            2.f * (x * z - w * y),
        ),
        vec3f(
            2.f * (x * y - w * z),
            1.f - 2.f * (x * x + z * z),
            2.f * (y * z + w * x),
        ),
        vec3f(
            2.f * (x * z + w * y),
            2.f * (y * z - w * x),
            1.f - 2.f * (x * x + y * y)
        ),
    );
}

fn scale_to_mat(scale: vec3f) -> mat3x3f {
    return mat3x3(
        vec3f(scale.x, 0, 0),
        vec3f(0, scale.y, 0), 
        vec3f(0, 0, scale.z)
    );
}
