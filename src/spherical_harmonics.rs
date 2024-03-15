// Reference for the spherical harmonics and their coefficients:
// https://mathworld.wolfram.com/SphericalHarmonic.html
use burn::tensor::{backend::Backend, Tensor};

const C0: f32 = 0.28209479177387814;
const C1: f32 = 0.4886025119029199;
const C2: [f32; 5] = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
];

const C3: [f32; 7] = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
];

const C4: [f32; 9] = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
];

struct ShData<B: Backend> {
    c0: Tensor<B, 1>,
    c1: Tensor<B, 1>,
    c2: Tensor<B, 1>,
    c3: Tensor<B, 1>,
    c4: Tensor<B, 1>,
}

/// Evaluates Spherical Harmonics.
///
/// Args:
///   deg: the degress of spherical harmonics upon which the evaluation takes
///     place.
///   sh: the spherical harmonic coefficients.
///   dirs: the directions from which the each sh is observed.
///
/// Returns:
///   An RGB color for every set of sh coefficients that are provided.
fn eval_sh<B: Backend>(
    deg: u32,
    sh: Tensor<B, 2>,
    dirs: Tensor<B, 2>,
    sh_data: ShData<B>,
) -> Tensor<B, 2> {
    if deg > 4 {
        panic!("Invalid sh degress: {deg}");
    }

    let sh_dims = sh.dims()[1] as u32;

    if sh_dims != (deg + 1) * (deg + 1) {
        panic!("Invalid sh shape {sh_dims} for degree {deg}");
    }

    let c0 = sh_data.c0.unsqueeze_dim(1);
    let c1 = sh_data.c1.unsqueeze_dim(1);
    let c2 = sh_data.c2.unsqueeze_dim(1);
    let c3 = sh_data.c3.unsqueeze_dim(1);
    let c4 = sh_data.c4.unsqueeze_dim(1);

    let mut result = c0 * sh;

    let br = 0..sh.dims()[0];

    if deg > 0 {
        let x = dirs.slice([br, 0..1]);
        let y = dirs.slice([br, 1..2]);
        let z = dirs.slice([br, 2..3]);

        result = result - c1 * y * sh.slice([br, 0..1]) + c1 * z * sh.slice([br, 1..2])
            - c1 * x * sh.slice([br, 2..3]);

        if deg > 1 {
            let (xx, yy, zz) = (x * x, y * y, z * z);
            let (xy, yz, xz) = (x * y, y * z, x * z);
            result = result
                + c2.slice([0..1, 0..1]) * xy * sh.slice([br, 3..4])
                + c2.slice([0..1, 0..1]) * yz * sh.slice([br, 4..5])
                + c2.slice([1..2, 0..1]) * (zz * 2 - xx - yy) * sh.slice([br, 5..6])
                + c2.slice([2..3, 0..1]) * xz * sh.slice([br, 6..7])
                + c2.slice([3..4, 0..1]) * (xx - yy) * sh.slice([br, 7..8]);

            if deg > 2 {
                result = result
                    + c3.slice([0..0, 0..1]) * y * (xx * 3 - yy) * sh.slice([br, 8..9])
                    + c3.slice([1..1, 0..1]) * xy * z * sh.slice([br, 9..10])
                    + c3.slice([2..2, 0..1]) * y * (zz * 4 - xx - yy) * sh.slice([br, 10..11])
                    + c3.slice([3..3, 0..1])
                        * z
                        * (zz * 2 - xx * 3 - yy * 3)
                        * sh.slice([br, 11..12])
                    + c3.slice([4..4, 0..1]) * x * (zz * 4 - xx - yy) * sh.slice([br, 12..13])
                    + c3.slice([5..5, 0..1]) * z * (xx - yy) * sh.slice([br, 13..14])
                    + c3.slice([6..6, 0..1]) * x * (xx - yy * 3) * sh.slice([br, 14..15]);

                if deg > 3 {
                    result = result
                        + c4.slice([0..0, 0..1]) * xy * (xx - yy) * sh.slice([br, 15..16])
                        + c4.slice([0..1, 0..1]) * yz * (xx * 3 - yy) * sh.slice([br, 16..17])
                        + c4.slice([0..2, 0..1]) * xy * (zz * 7 - 1) * sh.slice([br, 17..18])
                        + c4.slice([0..3, 0..1]) * yz * (zz * 7 - 3) * sh.slice([br, 18..19])
                        + c4.slice([0..4, 0..1])
                            * (zz * (zz * 35 - 30) + 3)
                            * sh.slice([br, 19..20])
                        + c4.slice([0..5, 0..1]) * xz * (zz * 7 - 3) * sh.slice([br, 20..21])
                        + c4.slice([0..6, 0..1])
                            * (xx - yy)
                            * (zz * 7 - 1)
                            * sh.slice([br, 21..22])
                        + c4.slice([0..7, 0..1]) * xz * (xx - yy * 3) * sh.slice([br, 22..23])
                        + c4.slice([0..8, 0..1])
                            * (xx * (xx - yy * 3) - yy * (xx * 3 - yy))
                            * sh.slice([br, 23..24]);
                }
            }
        }
    }

    result
}

pub fn rgb_to_sh_dc<B: Backend>(rgb: Tensor<B, 2>) -> Tensor<B, 2> {
    return (rgb - 0.5) / C0;
}

pub fn sh_dc_to_rgb<B: Backend>(sh: Tensor<B, 2>) -> Tensor<B, 2> {
    return sh * C0 + 0.5;
}
