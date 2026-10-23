// SPDX-License-Identifier: AGPL-3.0-or-later
// Original C++ code Copyright (c) 2012 Sanjib Sharma and Joss Bland-Hawthorn
// Ported from SurveyDesign.cpp, SurveyDesign.h, and Geometry.h
//! Survey geometry, cone intersections, and observational filters.
//!
//! Matches C++ Galaxia `Geometry.h` (`GPatch`), `SurveyDesign.cpp`, and `StellarPopulation.cpp`.
//!
//! - Cone geometry check matching `GPatch`:
//!   Center direction vector: $\mathbf{v}_c = (\cos b \cos l, \cos b \sin l, \sin b)$.
//!   Opening angle radius: $d\theta = \sqrt{\Omega / \pi}$ in radians ($\Omega_{\mathrm{rad}} = \Omega_{\mathrm{deg}^2} \times (\pi / 180)^2$).
//!   `check_sphere(pos, radius)`: checks if the bounding sphere of a tree node intersects the cone.
//!   `check_p(pos)`: checks if a point star is inside the cone.
//! - Observational survey cuts:
//!   Apparent magnitude cut: $-100.0 \le J \le 22.0$.
//!   Distance cut: $r_{\mathrm{helio}} - l_{\mathrm{leaf}} \cdot \sqrt{3} < r_{\mathrm{max}}$ ($r_{\mathrm{max}} = 1000.0\,\mathrm{kpc}$).

use crate::coords::SUN_POS_VEL;
use std::f64::consts::PI;

/// Circular survey cone patch matching `GPatch` in `Geometry.h:141-176`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GPatch {
    /// Unit direction vector of the cone center:
    /// `PosU = [cos(b)*cos(l), cos(b)*sin(l), sin(b)]`
    pub center_dir: [f64; 3],
    /// Opening half-angle $d\theta$ in radians.
    pub d_th: f64,
    /// Center galactic longitude $l$ in degrees.
    pub l_deg: f64,
    /// Center galactic latitude $b$ in degrees.
    pub b_deg: f64,
    /// Survey solid angle $\Omega$ in square degrees.
    pub area_sq_deg: f64,
}

impl GPatch {
    /// Create a `GPatch` given center $(l, b)$ in degrees and opening half-angle radius $d\theta$ in degrees.
    ///
    /// Matches `GPatch::GPatch(l, b, d_th1)` in `Geometry.h`.
    pub fn new(l_deg: f64, b_deg: f64, d_th_deg: f64) -> Self {
        let l_rad = l_deg * PI / 180.0;
        let b_rad = b_deg * PI / 180.0;
        let d_th = d_th_deg * PI / 180.0;

        let center_dir = [
            b_rad.cos() * l_rad.cos(),
            b_rad.cos() * l_rad.sin(),
            b_rad.sin(),
        ];

        // Derived equivalent solid angle in square degrees: Omega = 2*pi*(1 - cos(d_th)) * (180/pi)^2
        let area_sq_deg = 2.0 * PI * (1.0 - d_th.cos()) * (180.0 / PI).powi(2);

        Self {
            center_dir,
            d_th,
            l_deg,
            b_deg,
            area_sq_deg,
        }
    }

    /// Create a `GPatch` from a solid angle $\Omega$ in square degrees using the formula:
    ///
    /// $$d\theta = \sqrt{\frac{\Omega_{\mathrm{rad}}}{\pi}}, \quad \Omega_{\mathrm{rad}} = \Omega_{\mathrm{deg}^2} \times \left(\frac{\pi}{180}\right)^2$$
    pub fn from_survey_area(l_deg: f64, b_deg: f64, area_sq_deg: f64) -> Self {
        let omega_rad = area_sq_deg * (PI / 180.0).powi(2);
        let d_th = (omega_rad / PI).sqrt();

        let l_rad = l_deg * PI / 180.0;
        let b_rad = b_deg * PI / 180.0;

        let center_dir = [
            b_rad.cos() * l_rad.cos(),
            b_rad.cos() * l_rad.sin(),
            b_rad.sin(),
        ];

        Self {
            center_dir,
            d_th,
            l_deg,
            b_deg,
            area_sq_deg,
        }
    }

    /// Create a `GPatch` using Galaxia's exact spherical cap formulation (`SurveyDesign.cpp:290-294`):
    ///
    /// $$\text{temp} = \frac{\Omega_{\mathrm{rad}}}{2\pi}, \quad d\theta = \arccos(1 - \text{temp})$$
    pub fn from_survey_area_exact(l_deg: f64, b_deg: f64, area_sq_deg: f64) -> Self {
        let deg2rad = PI / 180.0;
        let temp = area_sq_deg * (deg2rad * deg2rad) / (2.0 * PI);
        let temp_clamped = temp.clamp(0.0, 2.0);
        let d_th = (1.0 - temp_clamped).clamp(-1.0, 1.0).acos();

        let l_rad = l_deg * deg2rad;
        let b_rad = b_deg * deg2rad;

        let center_dir = [
            b_rad.cos() * l_rad.cos(),
            b_rad.cos() * l_rad.sin(),
            b_rad.sin(),
        ];

        Self {
            center_dir,
            d_th,
            l_deg,
            b_deg,
            area_sq_deg,
        }
    }

    /// Check if a sphere with center `pos` and `radius` intersects the survey cone.
    ///
    /// Matches `GPatch::checkSphere` in `Geometry.h:151-161`:
    /// ```cpp
    /// tempr = sqrt(Pos[0]*Pos[0] + Pos[1]*Pos[1] + Pos[2]*Pos[2]);
    /// if (h >= tempr) return 1;
    /// temp = (Pos[0]*PosU[0] + Pos[1]*PosU[1] + Pos[2]*PosU[2]) / tempr;
    /// if ((acos(temp) - asin(h/tempr)) < d_th) return 1;
    /// else return 0;
    /// ```
    #[inline]
    pub fn check_sphere(&self, pos: &[f64; 3], radius: f64) -> bool {
        let tempr = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        if radius >= tempr {
            return true;
        }
        if tempr == 0.0 {
            return true;
        }

        let dot = (pos[0] * self.center_dir[0]
            + pos[1] * self.center_dir[1]
            + pos[2] * self.center_dir[2])
            / tempr;
        let cos_val = dot.clamp(-1.0, 1.0);
        let sin_h = (radius / tempr).clamp(-1.0, 1.0);

        (cos_val.acos() - sin_h.asin()) < self.d_th
    }

    /// C++ matching alias for `check_sphere`.
    #[inline]
    #[allow(non_snake_case)]
    pub fn checkSphere(&self, pos: &[f64; 3], radius: f64) -> bool {
        self.check_sphere(pos, radius)
    }

    /// Check if a point at `pos` lies strictly within the survey cone.
    ///
    /// Matches `GPatch::checkP` in `Geometry.h:163-171`:
    /// ```cpp
    /// tempr = sqrt(Pos[0]*Pos[0] + Pos[1]*Pos[1] + Pos[2]*Pos[2]);
    /// temp = (Pos[0]*PosU[0] + Pos[1]*PosU[1] + Pos[2]*PosU[2]) / tempr;
    /// if (acos(temp) < d_th) return 1;
    /// else return 0;
    /// ```
    #[inline]
    pub fn check_p(&self, pos: &[f64; 3]) -> bool {
        let tempr = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        if tempr == 0.0 {
            return true;
        }

        let dot = (pos[0] * self.center_dir[0]
            + pos[1] * self.center_dir[1]
            + pos[2] * self.center_dir[2])
            / tempr;
        let cos_val = dot.clamp(-1.0, 1.0);

        cos_val.acos() < self.d_th
    }

    /// C++ matching alias for `check_p`.
    #[inline]
    #[allow(non_snake_case)]
    pub fn checkP(&self, pos: &[f64; 3]) -> bool {
        self.check_p(pos)
    }
}

/// Observational filters and geometric survey cut thresholds.
#[derive(Debug, Clone, PartialEq)]
pub struct SurveyCuts {
    /// Apparent magnitude limits `[min, max]` on primary filter (default: `[-100.0, 22.0]`).
    pub app_mag_limits: [f64; 2],
    /// Absolute magnitude limits `[min, max]` (default: `[-1000.0, 1000.0]`).
    pub abs_mag_limits: [f64; 2],
    /// Color limits `[min, max]` on `mag1 - mag2` (default: `[-1000.0, 1000.0]`).
    pub color_limits: [f64; 2],
    /// Maximum heliocentric distance in kpc (default: `1000.0`).
    pub r_max: f64,
}

impl Default for SurveyCuts {
    fn default() -> Self {
        Self {
            app_mag_limits: [-100.0, 22.0],
            abs_mag_limits: [-1000.0, 1000.0],
            color_limits: [-1000.0, 1000.0],
            r_max: 1000.0,
        }
    }
}

impl SurveyCuts {
    /// Create new survey cuts with standard Galaxia defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check leaf node distance against maximum heliocentric distance.
    ///
    /// Matches `StellarPopulation.cpp:72-75`:
    /// $$r_{\mathrm{helio}} - l_{\mathrm{leaf}} \cdot \sqrt{3} < r_{\mathrm{max}}$$
    #[inline]
    pub fn check_leaf_distance(&self, r_helio: f64, leaf_half_side: f64) -> bool {
        let rc = r_helio - leaf_half_side * 3.0_f64.sqrt();
        rc < self.r_max
    }

    /// Check if a generated star satisfies the heliocentric distance limit $r \le r_{\mathrm{max}}$.
    #[inline]
    pub fn check_star_distance(&self, r_helio: f64) -> bool {
        r_helio <= self.r_max
    }

    /// Check apparent magnitude cut on primary band:
    /// `app_mag_limits[0] <= mag <= app_mag_limits[1]`.
    #[inline]
    pub fn check_app_mag(&self, app_mag: f64) -> bool {
        app_mag >= self.app_mag_limits[0] && app_mag <= self.app_mag_limits[1]
    }

    /// Check absolute magnitude cut:
    /// `abs_mag_limits[0] <= abs_mag <= abs_mag_limits[1]`.
    #[inline]
    pub fn check_abs_mag(&self, abs_mag: f64) -> bool {
        abs_mag >= self.abs_mag_limits[0] && abs_mag <= self.abs_mag_limits[1]
    }

    /// Check color cut on `mag1 - mag2`:
    /// `color_limits[0] <= (mag1 - mag2) <= color_limits[1]`.
    #[inline]
    pub fn check_color(&self, color: f64) -> bool {
        color >= self.color_limits[0] && color <= self.color_limits[1]
    }

    /// Combined check matching `SurveyDesign::checkColMag` (`SurveyDesign.cpp:344-347`):
    /// ```cpp
    /// if ((Star.mag(0) >= appMag[0]) && (Star.mag(0) <= appMag[1])
    ///     && ((Star.mag(1)-Star.mag(2)) >= color[0]) && ((Star.mag(1)-Star.mag(2)) <= color[1])
    ///     && ((Star.mag(0)-Star.dmod()) >= absMag[0]) && ((Star.mag(0)-Star.dmod()) <= absMag[1])
    ///     && (Star.rad() <= r_max))
    ///     return 1;
    /// ```
    #[inline]
    pub fn check_col_mag(&self, mag0: f64, mag1: f64, mag2: f64, dmod: f64, r_helio: f64) -> bool {
        mag0 >= self.app_mag_limits[0]
            && mag0 <= self.app_mag_limits[1]
            && (mag1 - mag2) >= self.color_limits[0]
            && (mag1 - mag2) <= self.color_limits[1]
            && (mag0 - dmod) >= self.abs_mag_limits[0]
            && (mag0 - dmod) <= self.abs_mag_limits[1]
            && r_helio <= self.r_max
    }
}

/// Complete survey configuration combining cone geometry, observational cuts, and observer position.
#[derive(Debug, Clone, PartialEq)]
pub struct SurveyDesign {
    /// Patch geometry
    pub patch: GPatch,
    /// Observational cuts
    pub cuts: SurveyCuts,
    /// Observer position and velocity in Galactocentric Cartesian coordinates (default: `SUN_POS_VEL`)
    pub pos_c: [f64; 6],
}

impl SurveyDesign {
    /// Create a new survey design with the given cone center $(l, b)$ in degrees and area in square degrees.
    pub fn new(l_deg: f64, b_deg: f64, area_sq_deg: f64) -> Self {
        Self {
            patch: GPatch::from_survey_area(l_deg, b_deg, area_sq_deg),
            cuts: SurveyCuts::default(),
            pos_c: SUN_POS_VEL,
        }
    }

    /// Create survey design from parameters in `Params`.
    pub fn from_params(params: &crate::params::Params) -> Self {
        let mut cuts = SurveyCuts::default();
        cuts.app_mag_limits = params.app_mag_limits;
        cuts.abs_mag_limits = params.abs_mag_limits;
        cuts.color_limits = params.color_limits;
        cuts.r_max = params.r_max;

        let patch =
            GPatch::from_survey_area_exact(params.longitude, params.latitude, params.survey_area);

        Self {
            patch,
            cuts,
            pos_c: params.pos_c,
        }
    }

    /// Check if a BHTree leaf node should be processed or culled.
    ///
    /// Matches `StellarPopulation.cpp:69-75`:
    /// 1. Shifts leaf Galactocentric position to Heliocentric frame: $\mathbf{r}_{\mathrm{helio}} = \mathbf{r}_{\mathrm{leaf}} - \mathbf{r}_\odot$.
    /// 2. Tests distance cut: $r_{\mathrm{helio}} - l_{\mathrm{leaf}} \cdot \sqrt{3} < r_{\mathrm{max}}$.
    /// 3. Tests cone intersection: `patch.check_sphere(&r_helio, l_leaf * sqrt(3))`.
    #[inline]
    pub fn check_leaf(&self, leaf_gal_pos: &[f64; 3], leaf_half_side: f64) -> bool {
        let r_helio = [
            leaf_gal_pos[0] - self.pos_c[0],
            leaf_gal_pos[1] - self.pos_c[1],
            leaf_gal_pos[2] - self.pos_c[2],
        ];

        let dist =
            (r_helio[0] * r_helio[0] + r_helio[1] * r_helio[1] + r_helio[2] * r_helio[2]).sqrt();
        let h = leaf_half_side * 3.0_f64.sqrt();

        if dist - h >= self.cuts.r_max {
            return false;
        }

        self.patch.check_sphere(&r_helio, h)
    }

    /// Check if a generated star at heliocentric position satisfies both cone geometry and magnitude/color cuts.
    #[inline]
    pub fn check_star(
        &self,
        star_helio_pos: &[f64; 3],
        mag0: f64,
        mag1: f64,
        mag2: f64,
        dmod: f64,
    ) -> bool {
        let dist = (star_helio_pos[0] * star_helio_pos[0]
            + star_helio_pos[1] * star_helio_pos[1]
            + star_helio_pos[2] * star_helio_pos[2])
            .sqrt();

        if !self.cuts.check_col_mag(mag0, mag1, mag2, dmod, dist) {
            return false;
        }

        self.patch.check_p(star_helio_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cone_center_direction() {
        // Center at l=0, b=0 -> [1, 0, 0]
        let patch0 = GPatch::new(0.0, 0.0, 1.0);
        assert!((patch0.center_dir[0] - 1.0).abs() < 1e-12);
        assert!(patch0.center_dir[1].abs() < 1e-12);
        assert!(patch0.center_dir[2].abs() < 1e-12);

        // Center at l=90, b=0 -> [0, 1, 0]
        let patch90 = GPatch::new(90.0, 0.0, 1.0);
        assert!(patch90.center_dir[0].abs() < 1e-12);
        assert!((patch90.center_dir[1] - 1.0).abs() < 1e-12);
        assert!(patch90.center_dir[2].abs() < 1e-12);

        // Center at North Galactic Pole b=90 -> [0, 0, 1]
        let patch_ngp = GPatch::new(0.0, 90.0, 1.0);
        assert!(patch_ngp.center_dir[0].abs() < 1e-12);
        assert!(patch_ngp.center_dir[1].abs() < 1e-12);
        assert!((patch_ngp.center_dir[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_opening_angle_from_survey_area() {
        let area = 0.1; // 0.1 sq deg
        let patch = GPatch::from_survey_area(60.0, 10.0, area);

        // Omega in rad: 0.1 * (pi / 180)^2 = 3.046174e-5 steradians
        let omega_rad = area * (PI / 180.0).powi(2);
        let expected_dth = (omega_rad / PI).sqrt();
        assert!((patch.d_th - expected_dth).abs() < 1e-12);

        // Compare with exact spherical cap formula:
        let patch_exact = GPatch::from_survey_area_exact(60.0, 10.0, area);
        // For 0.1 sq deg, difference between small angle sqrt(Omega/pi) and acos(1 - Omega/2pi) is < 1e-8
        assert!((patch.d_th - patch_exact.d_th).abs() < 1e-8);
    }

    #[test]
    fn test_check_p() {
        let patch = GPatch::new(0.0, 0.0, 5.0); // 5 degree half angle along X axis

        // Along cone center axis (X > 0) -> inside
        assert!(patch.check_p(&[10.0, 0.0, 0.0]));

        // Within 4 degrees (< 5 deg) -> inside
        let angle_4deg = 4.0 * PI / 180.0;
        let pos_in = [10.0 * angle_4deg.cos(), 10.0 * angle_4deg.sin(), 0.0];
        assert!(patch.check_p(&pos_in));

        // Outside at 6 degrees (> 5 deg) -> outside
        let angle_6deg = 6.0 * PI / 180.0;
        let pos_out = [10.0 * angle_6deg.cos(), 10.0 * angle_6deg.sin(), 0.0];
        assert!(!patch.check_p(&pos_out));

        // Opposite direction (-X) -> outside
        assert!(!patch.check_p(&[-10.0, 0.0, 0.0]));
    }

    #[test]
    fn test_check_sphere() {
        let patch = GPatch::new(0.0, 0.0, 5.0); // 5 degree opening along X

        // Sphere at center axis -> intersects
        assert!(patch.check_sphere(&[10.0, 0.0, 0.0], 1.0));

        // Sphere center at 7 degrees, radius 3 degrees: (7 - 3) = 4 < 5 -> intersects
        let angle_7deg = 7.0 * PI / 180.0;
        let center_7deg = [10.0 * angle_7deg.cos(), 10.0 * angle_7deg.sin(), 0.0];
        // radius = 10.0 * sin(3 deg) ~ 0.523
        let r_3deg = 10.0 * (3.0 * PI / 180.0).sin();
        assert!(patch.check_sphere(&center_7deg, r_3deg));

        // Sphere center at 10 degrees, radius 1 degree: (10 - 1) = 9 > 5 -> does NOT intersect
        let angle_10deg = 10.0 * PI / 180.0;
        let center_10deg = [10.0 * angle_10deg.cos(), 10.0 * angle_10deg.sin(), 0.0];
        let r_1deg = 10.0 * (1.0 * PI / 180.0).sin();
        assert!(!patch.check_sphere(&center_10deg, r_1deg));

        // Sphere large enough to contain origin (radius >= tempr) always intersects
        assert!(patch.check_sphere(&[0.0, 10.0, 0.0], 15.0));
    }

    #[test]
    fn test_survey_cuts() {
        let cuts = SurveyCuts::default();

        // Apparent magnitude: [-100.0, 22.0]
        assert!(cuts.check_app_mag(15.0));
        assert!(cuts.check_app_mag(-100.0));
        assert!(cuts.check_app_mag(22.0));
        assert!(!cuts.check_app_mag(22.1));
        assert!(!cuts.check_app_mag(-100.1));

        // Distance cut for leaf: r_helio - leaf_l * sqrt(3) < 1000
        assert!(cuts.check_leaf_distance(500.0, 10.0));
        assert!(cuts.check_leaf_distance(1010.0, 10.0)); // 1010 - 17.32 = 992.68 < 1000 -> true
        assert!(!cuts.check_leaf_distance(1020.0, 10.0)); // 1020 - 17.32 = 1002.68 > 1000 -> false

        // Star distance: r <= 1000
        assert!(cuts.check_star_distance(1000.0));
        assert!(!cuts.check_star_distance(1000.01));

        // Combined check_col_mag
        let dmod = 10.0;
        assert!(cuts.check_col_mag(18.0, 18.0, 17.5, dmod, 50.0));
        assert!(!cuts.check_col_mag(23.0, 23.0, 22.5, dmod, 50.0)); // J > 22
    }

    #[test]
    fn test_survey_design_leaf_and_star() {
        // Survey at l=60, b=10, area=0.1 sq deg
        let survey = SurveyDesign::new(60.0, 10.0, 0.1);

        // Leaf along line of sight at heliocentric dist 10 kpc:
        // helio pos = 10 * [cos(10)*cos(60), cos(10)*sin(60), sin(10)]
        let l_rad = 60.0 * PI / 180.0;
        let b_rad = 10.0 * PI / 180.0;
        let helio_pos = [
            10.0 * b_rad.cos() * l_rad.cos(),
            10.0 * b_rad.cos() * l_rad.sin(),
            10.0 * b_rad.sin(),
        ];
        // Galactocentric leaf pos = helio_pos + sun_pos
        let gal_pos = [
            helio_pos[0] + survey.pos_c[0],
            helio_pos[1] + survey.pos_c[1],
            helio_pos[2] + survey.pos_c[2],
        ];

        // Leaf right on line of sight passes
        assert!(survey.check_leaf(&gal_pos, 0.1));

        // Star on line of sight with valid mag passes
        assert!(survey.check_star(&helio_pos, 15.0, 15.0, 14.5, 10.0));

        // Leaf far in opposite direction fails
        let gal_pos_opp = [
            -helio_pos[0] + survey.pos_c[0],
            -helio_pos[1] + survey.pos_c[1],
            -helio_pos[2] + survey.pos_c[2],
        ];
        assert!(!survey.check_leaf(&gal_pos_opp, 0.1));
    }
}
