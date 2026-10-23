// SPDX-License-Identifier: AGPL-3.0-or-later
// Original C++ code Copyright (c) 2012 Sanjib Sharma and Joss Bland-Hawthorn
// Ported from Cot.cpp and Cot.h
//! Astrometric and coordinate transformations matching C++ Galaxia (`Cot.cpp` / `Cot.h`).
//!
//! Provides:
//! - Equatorial (RA, Dec) <-> Galactic (l, b) transformations.
//! - Cartesian (x, y, z) <-> Spherical (l, b, r) coordinates in 3D and 6D.
//! - Cartesian <-> Cylindrical (phi, z, R) coordinates.
//! - Heliocentric <-> Galactocentric transformations for the Sun.
//! - Distance modulus $\mu = 5 \log_{10}(r_{\mathrm{kpc}} \cdot 100.0) = 5 \log_{10}(r_{\mathrm{pc}}) - 5$.
//! - Exact conversion constant $RADEG = 57.2958$.
//! - 3x3 Rotation matrices for triaxial bulge transformation and coordinate frame rotations.

use std::f64::consts::PI;

/// Exact conversion constant defined in Galaxia `sutils.cpp` and `GInterpolator.h`.
pub const RADEG: f64 = 57.2958;

/// Standard degrees-to-radians conversion factor.
pub const DEG2RAD: f64 = PI / 180.0;

/// Solar position and velocity in Galactocentric Cartesian coordinates (kpc and km/s):
/// `posC = [-8.0, 0.0, 0.015, 11.1, 239.08, 7.25]`
pub const SUN_POS: [f64; 3] = [-8.0, 0.0, 0.015];
pub const SUN_VEL: [f64; 3] = [11.1, 239.08, 7.25];
pub const SUN_POS_VEL: [f64; 6] = [-8.0, 0.0, 0.015, 11.1, 239.08, 7.25];

/// 3x3 matrix for Galactocentric coordinate and velocity rotations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix3x3 {
    pub m: [[f64; 3]; 3],
}

impl Matrix3x3 {
    /// Create identity matrix.
    pub fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Rotation around coordinate axis (0 = X, 1 = Y, 2 = Z).
    /// Matches `Cot::RotationMatrix(axis, theta)`.
    pub fn rotation_axis(axis: usize, theta: f64) -> Self {
        let mut mat = Self::identity();
        let axis1 = (axis + 1) % 3;
        let axis2 = (axis + 2) % 3;
        let c = theta.cos();
        let s = theta.sin();
        mat.m[axis1][axis1] = c;
        mat.m[axis1][axis2] = s;
        mat.m[axis2][axis1] = -s;
        mat.m[axis2][axis2] = c;
        mat
    }

    /// Multiply two 3x3 matrices: self * other.
    pub fn mul(&self, other: &Self) -> Self {
        let mut res = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                res[i][j] = self.m[i][0] * other.m[0][j]
                    + self.m[i][1] * other.m[1][j]
                    + self.m[i][2] * other.m[2][j];
            }
        }
        Self { m: res }
    }

    /// Transform a 3D vector: res = self * pos.
    #[inline]
    pub fn mult_vec(&self, pos: &[f64; 3]) -> [f64; 3] {
        [
            self.m[0][0] * pos[0] + self.m[0][1] * pos[1] + self.m[0][2] * pos[2],
            self.m[1][0] * pos[0] + self.m[1][1] * pos[1] + self.m[1][2] * pos[2],
            self.m[2][0] * pos[0] + self.m[2][1] * pos[1] + self.m[2][2] * pos[2],
        ]
    }
}

/// Distance modulus from heliocentric distance in kiloparsecs.
///
/// $$\mu = 5 \log_{10}(r_{\mathrm{kpc}} \cdot 100.0) = 5 \log_{10}(r_{\mathrm{pc}}) - 5$$
#[inline]
pub fn distance_modulus(r_kpc: f64) -> f64 {
    5.0 * (r_kpc * 100.0).log10()
}

/// Invert distance modulus $\mu$ to distance in kiloparsecs.
#[inline]
pub fn distance_from_dmod(dmod: f64) -> f64 {
    10.0_f64.powf(dmod / 5.0) / 100.0
}

/// Convert Galactic coordinates $(l, b)$ in degrees to Equatorial coordinates $(\alpha, \delta)$ (RA, Dec) in degrees.
///
/// Matches `Cot::lb_to_radec` in `Cot.cpp`:
/// - NGP: $\alpha_{\mathrm{GP}} = 192.85948^\circ$, $\delta_{\mathrm{GP}} = 27.12825^\circ$.
/// - Galactic center ascending node: $l_{\mathrm{CP}} = 122.932^\circ$.
/// - Uses constants $DTOR = 3.14159 / 180.0$ and $RADEG = 180.0 / 3.14159$ as in C++.
pub fn lb_to_radec(l_deg: f64, b_deg: f64) -> (f64, f64) {
    let dtor = 3.14159 / 180.0;
    let radeg = 180.0 / 3.14159;
    let al_gp = 192.85948 * dtor;
    let de_gp = 27.12825 * dtor;
    let l_cp = 122.932 * dtor;

    let l = l_deg * dtor;
    let b = b_deg * dtor;

    let cos_b = b.cos();
    let sin_b = b.sin();
    let cos_de = de_gp.cos();
    let sin_de = de_gp.sin();
    let delta_l = l_cp - l;
    let sin_dl = delta_l.sin();
    let cos_dl = delta_l.cos();

    let denom = cos_b * sin_dl;
    let mut alpha = al_gp + (cos_b * sin_dl).atan2(cos_de * sin_b - sin_de * cos_b * cos_dl);

    let delta_deg = if denom.abs() > 1e-12 {
        let tan_delta = (alpha - al_gp).sin() * (sin_de * sin_b + cos_de * cos_b * cos_dl) / denom;
        tan_delta.atan() * radeg
    } else {
        let sin_delta = sin_de * sin_b + cos_de * cos_b * cos_dl;
        sin_delta.clamp(-1.0, 1.0).asin() * radeg
    };

    alpha *= radeg;
    if alpha >= 360.0 {
        alpha -= 360.0;
    } else if alpha < 0.0 {
        alpha += 360.0;
    }

    (alpha, delta_deg)
}

/// Convert Equatorial coordinates $(\alpha, \delta)$ (RA, Dec) in degrees to Galactic coordinates $(l, b)$ in degrees.
///
/// Inverts `Cot::lb_to_radec` using the same pole constants.
pub fn radec_to_lb(ra_deg: f64, dec_deg: f64) -> (f64, f64) {
    let dtor = 3.14159 / 180.0;
    let radeg = 180.0 / 3.14159;
    let al_gp = 192.85948 * dtor;
    let de_gp = 27.12825 * dtor;
    let l_cp = 122.932 * dtor;

    let a = ra_deg * dtor;
    let d = dec_deg * dtor;

    let sin_d = d.sin();
    let cos_d = d.cos();
    let sin_de = de_gp.sin();
    let cos_de = de_gp.cos();
    let delta_a = a - al_gp;

    let sin_b = sin_de * sin_d + cos_de * cos_d * delta_a.cos();
    let b = sin_b.clamp(-1.0, 1.0).asin();

    let y = cos_d * delta_a.sin();
    let x = cos_de * sin_d - sin_de * cos_d * delta_a.cos();
    let l = l_cp - y.atan2(x);

    let mut l_deg = l * radeg;
    let b_deg = b * radeg;

    while l_deg < 0.0 {
        l_deg += 360.0;
    }
    while l_deg >= 360.0 {
        l_deg -= 360.0;
    }

    (l_deg, b_deg)
}

/// Alias for `radec_to_lb` matching spec nomenclature.
#[inline]
pub fn galactic_lb(ra_deg: f64, dec_deg: f64) -> (f64, f64) {
    radec_to_lb(ra_deg, dec_deg)
}

/// Alias for `lb_to_radec` matching spec nomenclature.
#[inline]
pub fn equatorial_radec(l_deg: f64, b_deg: f64) -> (f64, f64) {
    lb_to_radec(l_deg, b_deg)
}

/// ICRS -> Galactic rotation matrix (Hipparcos/ESA 1997, vol. 1 sect. 1.5.3), row-major.
/// Used for the RA/Dec -> (l, b) conversion of batch fields.
pub const ICRS_TO_GAL: [[f64; 3]; 3] = [
    [
        -0.0548755604162154,
        -0.8734370902348850,
        -0.4838350155487132,
    ],
    [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
];

/// Galactic (l, b) in degrees of an ICRS (RA, Dec) position using Hipparcos rotation.
pub fn icrs_to_galactic(ra_deg: f64, dec_deg: f64) -> (f64, f64) {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let vx = dec.cos() * ra.cos();
    let vy = dec.cos() * ra.sin();
    let vz = dec.sin();

    let x = ICRS_TO_GAL[0][0] * vx + ICRS_TO_GAL[0][1] * vy + ICRS_TO_GAL[0][2] * vz;
    let y = ICRS_TO_GAL[1][0] * vx + ICRS_TO_GAL[1][1] * vy + ICRS_TO_GAL[1][2] * vz;
    let z = ICRS_TO_GAL[2][0] * vx + ICRS_TO_GAL[2][1] * vy + ICRS_TO_GAL[2][2] * vz;

    let b = z.clamp(-1.0, 1.0).asin().to_degrees();
    let mut l = y.atan2(x).to_degrees() % 360.0;
    if l < 0.0 {
        l += 360.0;
    }
    (l, b)
}

/// Deterministic Galaxia seed for a field id.
/// `seed = (tic % 100000) + 1`
#[inline]
pub fn seed_for_tic(tic: u64) -> u32 {
    ((tic % 100000) + 1) as u32
}

/// Convert Cartesian position `[x, y, z]` (kpc) to spherical coordinates `[l, b, r]` (rad, rad, kpc).
///
/// Matches `Cot::xyz_to_lbr(PosC, PosS, 3)` in `Cot.cpp`:
/// - `PosS[0] = atan2(y, x)` (longitude $l \in [-\pi, \pi]$)
/// - `PosS[1] = asin(z / r)` (latitude $b \in [-\pi/2, \pi/2]$)
/// - `PosS[2] = sqrt(x^2 + y^2 + z^2)`
#[inline]
pub fn xyz_to_lbr(pos_c: &[f64; 3]) -> [f64; 3] {
    let r = (pos_c[0] * pos_c[0] + pos_c[1] * pos_c[1] + pos_c[2] * pos_c[2]).sqrt();
    let l = pos_c[1].atan2(pos_c[0]);
    let b = if r > 0.0 {
        (pos_c[2] / r).clamp(-1.0, 1.0).asin()
    } else {
        0.0
    };
    [l, b, r]
}

/// Alias for `xyz_to_lbr` for explicit 3D typing.
#[inline]
pub fn xyz_to_lbr_3d(pos_c: &[f64; 3]) -> [f64; 3] {
    xyz_to_lbr(pos_c)
}

/// Convert spherical coordinates `[l, b, r]` (rad, rad, kpc) to Cartesian `[x, y, z]` (kpc).
///
/// Matches `Cot::lbr_to_xyz(PosS, PosC, 3)` in `Cot.cpp`.
#[inline]
pub fn lbr_to_xyz(pos_s: &[f64; 3]) -> [f64; 3] {
    let l = pos_s[0];
    let b = pos_s[1];
    let r = pos_s[2];

    let x = r * b.cos() * l.cos();
    let y = r * b.cos() * l.sin();
    let z = r * b.sin();

    [x, y, z]
}

/// Alias for `lbr_to_xyz` for explicit 3D typing.
#[inline]
pub fn lbr_to_xyz_3d(pos_s: &[f64; 3]) -> [f64; 3] {
    lbr_to_xyz(pos_s)
}

/// Convert Cartesian position and velocity `[x, y, z, vx, vy, vz]` to spherical `[l, b, r, vl, vb, vr]`.
///
/// Matches `Cot::xyz_to_lbr(PosC, PosS, 6)` in `Cot.cpp`.
/// Velocities are rotated into the local orthonormal spherical basis $(\hat{\mathbf{e}}_l, \hat{\mathbf{e}}_b, \hat{\mathbf{e}}_r)$:
/// - $v_l = -v_x \sin l + v_y \cos l$
/// - $v_b = -v_x \sin b \cos l - v_y \sin b \sin l + v_z \cos b$
/// - $v_r = v_x \cos b \cos l + v_y \cos b \sin l + v_z \sin b$
pub fn xyz_to_lbr_6d(pos_c: &[f64; 6]) -> [f64; 6] {
    let [l, b, r] = xyz_to_lbr(&[pos_c[0], pos_c[1], pos_c[2]]);

    let sin_l = l.sin();
    let cos_l = l.cos();
    let sin_b = b.sin();
    let cos_b = b.cos();

    let vx = pos_c[3];
    let vy = pos_c[4];
    let vz = pos_c[5];

    let vl = -sin_l * vx + cos_l * vy;
    let vb = -sin_b * cos_l * vx - sin_b * sin_l * vy + cos_b * vz;
    let vr = cos_b * cos_l * vx + cos_b * sin_l * vy + sin_b * vz;

    [l, b, r, vl, vb, vr]
}

/// Convert spherical position and velocity `[l, b, r, vl, vb, vr]` to Cartesian `[x, y, z, vx, vy, vz]`.
///
/// Matches `Cot::lbr_to_xyz(PosS, PosC, 6)` in `Cot.cpp`.
pub fn lbr_to_xyz_6d(pos_s: &[f64; 6]) -> [f64; 6] {
    let [x, y, z] = lbr_to_xyz(&[pos_s[0], pos_s[1], pos_s[2]]);

    let l = pos_s[0];
    let b = pos_s[1];

    let sin_l = l.sin();
    let cos_l = l.cos();
    let sin_b = b.sin();
    let cos_b = b.cos();

    let vl = pos_s[3];
    let vb = pos_s[4];
    let vr = pos_s[5];

    let vx = -sin_l * vl - sin_b * cos_l * vb + cos_b * cos_l * vr;
    let vy = cos_l * vl - sin_b * sin_l * vb + cos_b * sin_l * vr;
    let vz = cos_b * vb + sin_b * vr;

    [x, y, z, vx, vy, vz]
}

/// Convert Cartesian `[x, y, z]` to cylindrical coordinates `[phi, z, R]`.
///
/// Matches `Cot::xyz_to_lzr(PosC, PosS, 3)`.
#[inline]
pub fn xyz_to_lzr_3d(pos_c: &[f64; 3]) -> [f64; 3] {
    let phi = pos_c[1].atan2(pos_c[0]);
    let z = pos_c[2];
    let r_cyl = (pos_c[0] * pos_c[0] + pos_c[1] * pos_c[1]).sqrt();
    [phi, z, r_cyl]
}

/// Convert cylindrical coordinates `[phi, z, R]` to Cartesian `[x, y, z]`.
///
/// Matches `Cot::lzr_to_xyz(PosS, PosC, 3)`.
#[inline]
pub fn lzr_to_xyz_3d(pos_s: &[f64; 3]) -> [f64; 3] {
    let phi = pos_s[0];
    let z = pos_s[1];
    let r_cyl = pos_s[2];
    [r_cyl * phi.cos(), r_cyl * phi.sin(), z]
}

/// Convert cylindrical velocity (v_phi, v_z, v_R) to Cartesian velocity (v_x, v_y, v_z).
/// Given position pos = (x, y, z) in kpc.
/// Matches `Cot::xyz_to_lzr` followed by `Cot::lzr_to_xyz` with dim=6.
#[inline]
pub fn vel_cylindrical_to_cartesian(
    pos_xyz: &[f64; 3],
    v_phi: f64,
    v_z: f64,
    v_r: f64,
) -> [f64; 3] {
    let phi = pos_xyz[1].atan2(pos_xyz[0]);
    let sin_phi = phi.sin();
    let cos_phi = phi.cos();
    [
        -v_phi * sin_phi + v_r * cos_phi,
        v_phi * cos_phi + v_r * sin_phi,
        v_z,
    ]
}

/// Convert spherical velocity (v_l, v_b, v_r) to Cartesian velocity (v_x, v_y, v_z).
/// Given position pos = (x, y, z) in kpc.
/// Matches `Cot::xyz_to_lbr` followed by `Cot::lbr_to_xyz` with dim=6.
#[inline]
pub fn vel_spherical_to_cartesian(pos_xyz: &[f64; 3], v_l: f64, v_b: f64, v_r: f64) -> [f64; 3] {
    let [l, b, _] = xyz_to_lbr(pos_xyz);
    let sin_l = l.sin();
    let cos_l = l.cos();
    let sin_b = b.sin();
    let cos_b = b.cos();
    [
        -v_l * sin_l - v_b * sin_b * cos_l + v_r * cos_b * cos_l,
        v_l * cos_l - v_b * sin_b * sin_l + v_r * cos_b * sin_l,
        v_b * cos_b + v_r * sin_b,
    ]
}

/// Convenience converter from Cartesian `[x, y, z]` (heliocentric kpc) to
/// spherical coordinates in degrees $(l, b)$ and heliocentric distance $r$ (kpc).
///
/// Matches the exact logic used in `Galaxia.cpp:64-68` (`append_ext`):
/// - $l = \mathrm{atan2}(y, x) \times 57.2958$
/// - if $l < 0$: $l \leftarrow 360.0 + l$
/// - $b = \arcsin(z / r) \times 57.2958$
#[inline]
pub fn xyz_to_lbr_deg(pos_c: &[f64; 3]) -> (f64, f64, f64) {
    let r = (pos_c[0] * pos_c[0] + pos_c[1] * pos_c[1] + pos_c[2] * pos_c[2]).sqrt();
    let mut l_deg = pos_c[1].atan2(pos_c[0]) * RADEG;
    if l_deg < 0.0 {
        l_deg += 360.0;
    }
    let b_deg = if r > 0.0 {
        (pos_c[2] / r).clamp(-1.0, 1.0).asin() * RADEG
    } else {
        0.0
    };
    (l_deg, b_deg, r)
}

/// Convert Galactocentric Cartesian position to Heliocentric Cartesian position.
///
/// $$\mathbf{r}_{\mathrm{helio}} = \mathbf{r}_{\mathrm{galacto}} - \mathbf{r}_\odot$$
#[inline]
pub fn galactocentric_to_heliocentric_pos(pos_gal: &[f64; 3]) -> [f64; 3] {
    [
        pos_gal[0] - SUN_POS[0],
        pos_gal[1] - SUN_POS[1],
        pos_gal[2] - SUN_POS[2],
    ]
}

/// Convert Heliocentric Cartesian position to Galactocentric Cartesian position.
///
/// $$\mathbf{r}_{\mathrm{galacto}} = \mathbf{r}_{\mathrm{helio}} + \mathbf{r}_\odot$$
#[inline]
pub fn heliocentric_to_galactocentric_pos(pos_hel: &[f64; 3]) -> [f64; 3] {
    [
        pos_hel[0] + SUN_POS[0],
        pos_hel[1] + SUN_POS[1],
        pos_hel[2] + SUN_POS[2],
    ]
}

/// Convert Galactocentric 6D phase space (pos, vel) to Heliocentric 6D.
#[inline]
pub fn galactocentric_to_heliocentric_6d(pv_gal: &[f64; 6]) -> [f64; 6] {
    [
        pv_gal[0] - SUN_POS_VEL[0],
        pv_gal[1] - SUN_POS_VEL[1],
        pv_gal[2] - SUN_POS_VEL[2],
        pv_gal[3] - SUN_POS_VEL[3],
        pv_gal[4] - SUN_POS_VEL[4],
        pv_gal[5] - SUN_POS_VEL[5],
    ]
}

/// Convert Heliocentric 6D phase space (pos, vel) to Galactocentric 6D.
#[inline]
pub fn heliocentric_to_galactocentric_6d(pv_hel: &[f64; 6]) -> [f64; 6] {
    [
        pv_hel[0] + SUN_POS_VEL[0],
        pv_hel[1] + SUN_POS_VEL[1],
        pv_hel[2] + SUN_POS_VEL[2],
        pv_hel[3] + SUN_POS_VEL[3],
        pv_hel[4] + SUN_POS_VEL[4],
        pv_hel[5] + SUN_POS_VEL[5],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radeg_constant() {
        assert_eq!(RADEG, 57.2958);
    }

    #[test]
    fn test_distance_modulus() {
        // At 10 pc (0.01 kpc), dmod = 5 * log10(0.01 * 100) = 0.0
        let dmod_10pc = distance_modulus(0.01);
        assert!((dmod_10pc - 0.0).abs() < 1e-12);

        // At 100 pc (0.1 kpc), dmod = 5 * log10(10) = 5.0
        let dmod_100pc = distance_modulus(0.1);
        assert!((dmod_100pc - 5.0).abs() < 1e-12);

        // At 1 kpc, dmod = 5 * log10(100) = 10.0
        let dmod_1kpc = distance_modulus(1.0);
        assert!((dmod_1kpc - 10.0).abs() < 1e-12);

        // Invert roundtrip
        for &r in &[0.05, 0.25, 1.0, 8.0, 50.0, 1000.0] {
            let mu = distance_modulus(r);
            let r_inv = distance_from_dmod(mu);
            assert!((r - r_inv).abs() / r < 1e-12);
        }
    }

    #[test]
    fn test_xyz_lbr_roundtrip_3d() {
        let test_positions = [
            [1.0, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 0.0, 3.2],
            [-4.0, 5.0, -2.0],
            [8.0, -3.0, 0.5],
        ];

        for pos in &test_positions {
            let lbr = xyz_to_lbr(pos);
            let xyz = lbr_to_xyz(&lbr);
            for i in 0..3 {
                assert!(
                    (pos[i] - xyz[i]).abs() < 1e-12,
                    "Mismatch in pos {:?}: got {:?}",
                    pos,
                    xyz
                );
            }
        }
    }

    #[test]
    fn test_xyz_lbr_roundtrip_6d() {
        let test_pv = [
            [8.0, 0.0, 0.015, 10.0, 220.0, -5.0],
            [-5.0, 3.0, -1.0, -30.0, 150.0, 40.0],
        ];

        for pv in &test_pv {
            let lbr6 = xyz_to_lbr_6d(pv);
            let xyz6 = lbr_to_xyz_6d(&lbr6);
            for i in 0..6 {
                assert!(
                    (pv[i] - xyz6[i]).abs() < 1e-11,
                    "Mismatch in 6d component {}: expected {}, got {}",
                    i,
                    pv[i],
                    xyz6[i]
                );
            }
        }
    }

    #[test]
    fn test_xyz_lzr_roundtrip() {
        let pos = [-3.5, 4.2, 1.8];
        let lzr = xyz_to_lzr_3d(&pos);
        let xyz = lzr_to_xyz_3d(&lzr);
        for i in 0..3 {
            assert!((pos[i] - xyz[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_lb_to_radec_and_inverse() {
        let test_coords = [
            (0.0, 0.0),          // Galactic center
            (122.932, 27.12825), // Near pole / node
            (60.0, 10.0),        // gx_test survey direction
            (180.0, -30.0),
            (270.0, 45.0),
        ];

        for &(l, b) in &test_coords {
            let (ra, dec) = lb_to_radec(l, b);
            let (l_back, b_back) = radec_to_lb(ra, dec);

            assert!(
                (l - l_back).abs() < 1e-4,
                "l roundtrip failed for ({}, {}): got ({}, {})",
                l,
                b,
                l_back,
                b_back
            );
            assert!(
                (b - b_back).abs() < 1e-4,
                "b roundtrip failed for ({}, {}): got ({}, {})",
                l,
                b,
                l_back,
                b_back
            );
        }
    }

    #[test]
    fn test_galactocentric_heliocentric_roundtrip() {
        let pos_gal = [0.0, 0.0, 0.0]; // Galactic center
        let pos_hel = galactocentric_to_heliocentric_pos(&pos_gal);
        assert_eq!(pos_hel, [8.0, 0.0, -0.015]);

        let pos_gal_back = heliocentric_to_galactocentric_pos(&pos_hel);
        assert_eq!(pos_gal_back, pos_gal);

        let pv_gal = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let pv_hel = galactocentric_to_heliocentric_6d(&pv_gal);
        let pv_back = heliocentric_to_galactocentric_6d(&pv_hel);
        for i in 0..6 {
            assert!((pv_gal[i] - pv_back[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_xyz_to_lbr_deg() {
        // Point along positive X axis: l = 0, b = 0, r = 5.0
        let (l, b, r) = xyz_to_lbr_deg(&[5.0, 0.0, 0.0]);
        assert_eq!(l, 0.0);
        assert_eq!(b, 0.0);
        assert_eq!(r, 5.0);

        // Point along negative X axis: l = 180, b = 0, r = 2.0
        let (l, b, r) = xyz_to_lbr_deg(&[-2.0, 0.0, 0.0]);
        assert!((l - 180.0).abs() < 1e-3);
        assert_eq!(b, 0.0);
        assert_eq!(r, 2.0);
    }

    #[test]
    fn test_matrix_rotation() {
        let rot = Matrix3x3::rotation_axis(2, PI / 2.0);
        let v = [1.0, 0.0, 0.0];
        let res = rot.mult_vec(&v);
        assert!((res[0] - 0.0).abs() < 1e-10);
        assert!((res[1] - (-1.0)).abs() < 1e-10);
        assert!((res[2] - 0.0).abs() < 1e-10);
    }
}
