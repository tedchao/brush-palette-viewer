#import sorting

@compute
@workgroup_size(sorting::US_DIM, 1, 1)
fn main(
    @builtin(local_invocation_id) gtid: vec3u,
    @builtin(workgroup_id) gid: vec3u,
) {
    sorting::Upsweep(gtid, gid);
}