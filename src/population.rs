// SPDX-License-Identifier: AGPL-3.0-or-later
// Original C++ code Copyright (c) 2012 Sanjib Sharma and Joss Bland-Hawthorn
// Ported from Population.h, Population.cpp, and StellarPopulation.cpp
//! Milky Way stellar populations, density laws, kinematics, and spatial generators.
//!
//! Matches Galaxia 0.7.2's `Population.h`, `Population.cpp`, and `StellarPopulation.cpp`.

use crate::bhtree::BHNode;
use crate::coords::{vel_cylindrical_to_cartesian, vel_spherical_to_cartesian, Matrix3x3, DEG2RAD};
use crate::imf::Sampler;
use crate::interp::{interpolate, VCirc};
use crate::rng::Philox;

/// C++ Population.h warp/geometry literal constant
pub const PI_WARP: f64 = 3.14159;

pub const RSUN: f64 = 8.0;
pub const ZSUN: f64 = 0.015;
pub const VCONST: f64 = 226.84;

/// Star particle in Galaxia simulation.
/// Matches internal structure from `StarParticle.h`.
#[derive(Debug, Clone, PartialEq)]
pub struct StarParticle {
    /// Heliocentric or Galactocentric coordinates (x, y, z) in kpc.
    pub pos: [f64; 3],
    /// Heliocentric or Galactocentric velocities (vx, vy, vz) in km/s.
    pub vel: [f64; 3],
    /// Metallicity [Fe/H].
    pub feh: f64,
    /// Alpha enhancement [alpha/Fe].
    pub alpha: f64,
    /// Age stored as log10(age / yr).
    pub age: f64,
    /// Initial mass M_sun.
    pub smass: f64,
    /// Heliocentric distance in kpc.
    pub rad: f64,
    /// Apparent magnitudes (J, J, Ks).
    pub mags: [f64; 3],
    /// Absolute magnitudes (J, J, Ks).
    pub abs_mags: [f64; 3],
    /// Bolometric luminosity log10(L / L_sun).
    pub lum: f64,
    /// Effective temperature log10(Teff / K).
    pub teff: f64,
    /// Surface gravity log10(g / (cm/s^2)).
    pub grav: f64,
    /// Distance modulus.
    pub dmod: f64,
    /// Current stellar mass M_sun (after mass loss).
    pub mact: f64,
    /// RGB tip mass M_sun.
    pub mtip: f64,
    /// Population ID (0..9).
    pub popid: i32,
    /// Satellite ID (0 for Milky Way).
    pub satid: i32,
    /// Particle ID (0).
    pub partid: i32,
}

impl Default for StarParticle {
    fn default() -> Self {
        Self {
            pos: [0.0; 3],
            vel: [0.0; 3],
            feh: 0.0,
            alpha: 0.0,
            age: 0.0,
            smass: 0.0,
            rad: 0.0,
            mags: [1e6; 3],
            abs_mags: [1e6; 3],
            lum: 0.0,
            teff: 0.0,
            grav: 0.0,
            dmod: 0.0,
            mact: 0.0,
            mtip: 0.0,
            popid: 0,
            satid: 0,
            partid: 0,
        }
    }
}

impl StarParticle {
    /// Convert absolute magnitudes to apparent magnitudes via `+ dmod`.
    /// Matches `StarParticle::AbsToAppMag()` in `StarParticle.h:69-74`.
    #[inline]
    pub fn abs_to_app_mag(&mut self) {
        self.rad =
            (self.pos[0] * self.pos[0] + self.pos[1] * self.pos[1] + self.pos[2] * self.pos[2])
                .sqrt();
        self.dmod = 5.0 * (self.rad * 100.0).log10();
        self.mags[0] = self.abs_mags[0] + self.dmod;
        self.mags[1] = self.abs_mags[1] + self.dmod;
        self.mags[2] = self.abs_mags[2] + self.dmod;
    }
}

/// Trait defining a Milky Way stellar population.
pub trait Population: Send + Sync {
    /// Population identifier (0..9).
    fn id(&self) -> usize;
    /// BHTree age-expansion option (1 for ThinDisk 0..6, 0 for ThickDisk, Spheroid, Bulge).
    fn option_e(&self) -> usize;
    /// Central population age in yr.
    fn age(&self) -> f64;
    /// Half-width age window in yr.
    fn dage(&self) -> f64;
    /// Mean metallicity [Fe/H].
    fn feh(&self) -> f64;
    /// Metallicity dispersion sigma_[Fe/H].
    fn dfeh(&self) -> f64;
    /// Scale height ratio epsilon.
    fn eps(&self) -> f64;
    /// Intrinsic velocity dispersions (sigma_U, sigma_V, sigma_W) in km/s.
    fn sigma_v(&self) -> [f64; 3];

    /// Evaluate 3D spatial stellar density rho(x, y, z, age) in M_sun / kpc^3.
    fn density(&self, pos: &[f64; 3], age: f64) -> f64;

    /// Compute warp and flare transformed coordinate `PosB = [x, y, z_warp]` and flare factor `k_flare`.
    fn set_pos_b(&self, pos: &[f64; 3]) -> ([f64; 3], f64);

    /// Age-Metallicity Relation [Fe/H](R, age).
    fn amr(&self, pos: &[f64; 3], age: f64) -> f64;

    /// Velocity dispersion sigma_R(age, R) in km/s.
    fn sigmar_func(&self, age: f64, r_c: f64) -> f64;

    /// Asymmetric drift correction v_a in km/s.
    fn asymmetric_drift(&self, pos: &[f64; 3], v_c: f64, age: f64, sigma_r: f64) -> f64;

    /// Sample a star position inside a leaf bounding box using rejection sampling.
    /// Matches `BHNode::generatePos` in `BHTree.h:152-166`.
    fn generate_pos(&self, leaf: &BHNode, rng: &mut Philox) -> [f64; 4] {
        loop {
            let u1 = rng.uniform_f64();
            let u2 = rng.uniform_f64();
            let u3 = rng.uniform_f64();
            let x = leaf.pos[0] + (u1 - 0.5) * 2.0 * leaf.l;
            let y = leaf.pos[1] + (u2 - 0.5) * 2.0 * leaf.l;
            let z = leaf.pos[2] + (u3 - 0.5) * 2.0 * leaf.l;
            let age = if self.option_e() == 1 {
                let u4 = rng.uniform_f64();
                leaf.pos[3] + (u4 - 0.5) * 2.0 * leaf.lage
            } else {
                self.age()
            };

            let rho = self.density(&[x, y, z], age);
            let u5 = rng.uniform_f64();
            if u5 * (2.0 * leaf.rho_max) <= rho {
                return [x, y, z, age];
            }
        }
    }

    /// Assign age, metallicity, and mass to a star particle inside a leaf.
    /// Matches `Population::setAgeFehMass` in `Population.h:141-149`.
    fn set_age_feh_mass(&self, star: &mut StarParticle, imf: &Sampler, rng: &mut Philox) {
        let age_linear = if self.option_e() == 0 {
            self.age() + (rng.uniform_f64() - 0.5) * 2.0 * self.dage()
        } else {
            star.age
        };

        star.feh = self.amr(&star.pos, age_linear) + rng.standard_normal() * self.dfeh();
        star.alpha = star.feh;
        star.smass = imf.sample(rng);
        star.age = age_linear.log10();
    }

    /// Assign 3D kinematic velocities (vx, vy, vz) in km/s.
    /// Matches `Population::setVel` in `Population.cpp:117-145`.
    fn set_vel(&self, star: &mut StarParticle, vcirc: &VCirc, rng: &mut Philox) {
        let age_temp = 10.0f64.powf(star.age);
        let rc = (star.pos[0] * star.pos[0] + star.pos[1] * star.pos[1]).sqrt();
        let vc = vcirc.f(rc);
        let temp = self.sigmar_func(age_temp, rc);

        let z1 = rng.standard_normal();
        let z2 = rng.standard_normal();
        let z3 = rng.standard_normal();

        let sig_u = self.sigma_v()[0];
        let sig_v = self.sigma_v()[1];
        let sig_w = self.sigma_v()[2];

        let va = self.asymmetric_drift(&star.pos, vc, age_temp, temp);

        let v3 = z1 * temp * sig_v / sig_u - (vc - va);
        let v4 = z2 * temp * sig_w / sig_u;
        let v5 = z3 * temp;

        if self.id() == 8 {
            // Spheroid uses spherical coordinates (v_l, v_b, v_r)
            star.vel = vel_spherical_to_cartesian(&star.pos, v3, v4, v5);
        } else {
            // All other populations use cylindrical coordinates (v_phi, v_z, v_R)
            star.vel = vel_cylindrical_to_cartesian(&star.pos, v3, v4, v5);
        }
    }
}

// ----------------------------------------------------------------------------
// Base ThinDisk logic common to ThinDisk0 and ThinDisk1..6
// ----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ThinDiskBase {
    pub th_max: f64,
    pub g_warp: f64,
    pub r_warp: f64,
    pub r_flare: f64,
    pub g_flare: f64,
    pub sfr: f64,
    pub qparam: f64,
    pub bparam: f64,
    pub agevec: [f64; 7],
    pub fehvec: [f64; 7],
    pub dfehvec: [f64; 7],
    pub dagevec: [f64; 7],
    pub epsvec: [f64; 7],
    pub sigma_v: [f64; 3],
}

impl ThinDiskBase {
    pub fn new(warp_flare_on: usize) -> Self {
        let mut r_warp = 8.4 * RSUN / 8.5;
        let mut r_flare = 9.5 * RSUN / 8.5;
        if warp_flare_on == 0 {
            r_warp = 1e6;
            r_flare = 1e6;
        }

        Self {
            th_max: PI_WARP * 0.5,
            g_warp: 0.18,
            r_warp,
            r_flare,
            g_flare: 5.4e-4,
            sfr: 2.37,
            qparam: 0.33,
            bparam: 0.33,
            agevec: [0.075e9, 0.575e9, 1.5e9, 2.5e9, 4e9, 6e9, 8.5e9],
            fehvec: [0.01, 0.03, 0.03, 0.01, -0.07, -0.14, -0.37],
            dfehvec: [0.12, 0.12, 0.10, 0.11, 0.18, 0.17, 0.2],
            dagevec: [0.075e9, 0.425e9, 0.5e9, 0.5e9, 1e9, 1e9, 1.5e9],
            epsvec: [0.014, 0.0268, 0.0375, 0.0551, 0.0696, 0.0785, 0.0791],
            sigma_v: [50.0, 50.0 / 1.548, 50.0 / 2.38],
        }
    }

    #[inline]
    pub fn eps_func(&self, age: f64) -> f64 {
        0.079f64.min(0.104 * ((age + 0.1e9) / 10.1e9).powf(0.5))
    }

    #[inline]
    pub fn set_pos_b(&self, pos: &[f64; 3]) -> ([f64; 3], f64) {
        let r = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
        let mut th = pos[1].atan2(pos[0]);
        if th < 0.0 {
            th += 2.0 * PI_WARP;
        }

        let mut zw = pos[2];
        if r > self.r_warp {
            zw = pos[2] - self.g_warp * (r - self.r_warp) * (th - self.th_max).cos();
        }

        let k_flare = if r > self.r_flare {
            1.0 + self.g_flare * (r - self.r_flare)
        } else {
            1.0
        };

        ([pos[0], pos[1], zw], k_flare)
    }

    #[inline]
    pub fn amr(&self, pos: &[f64; 3], age: f64) -> f64 {
        let r = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
        interpolate(&self.agevec, &self.fehvec, age) - 0.07 * (r - RSUN)
    }

    #[inline]
    pub fn sigmar_func(&self, age: f64, rc: f64) -> f64 {
        43.1f64.min(50.0 * ((age + 0.1e9) / 10.1e9).powf(self.bparam))
            * ((RSUN - rc) * self.qparam / 2.5).exp()
    }
}

// ----------------------------------------------------------------------------
// Population 0: ThinDisk0
// ----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ThinDisk0 {
    pub base: ThinDiskBase,
    pub option_e: usize,
    pub rho0: f64,
    pub hr1: f64,
    pub hr2: f64,
}

impl ThinDisk0 {
    pub fn new(warp_flare_on: usize, option_e: usize) -> Self {
        let base = ThinDiskBase::new(warp_flare_on);
        let hr1 = 5.0;
        let hr2 = 3.0;
        let mut rho0 = base.sfr / (5.5683282 * (hr1 * hr1 * hr1 - hr2 * hr2 * hr2));
        if option_e == 0 {
            rho0 *= 2.0 * base.dagevec[0];
        }

        Self {
            base,
            option_e,
            rho0,
            hr1: hr1 * hr1, // 25.0
            hr2: hr2 * hr2, // 9.0
        }
    }

    fn dlnrho_dln_r(&self, pos: &[f64; 3], age: f64) -> f64 {
        let dr = 1e-3;
        let p2 = [pos[0] * (1.0 + dr), pos[1] * (1.0 + dr), pos[2]];
        let p1 = [pos[0] * (1.0 - dr), pos[1] * (1.0 - dr), pos[2]];
        let d2 = self.density(&p2, age);
        let d1 = self.density(&p1, age);
        (d2 - d1) / (dr * (d1 + d2))
    }
}

impl Population for ThinDisk0 {
    fn id(&self) -> usize {
        0
    }
    fn option_e(&self) -> usize {
        self.option_e
    }
    fn age(&self) -> f64 {
        self.base.agevec[0]
    }
    fn dage(&self) -> f64 {
        self.base.dagevec[0]
    }
    fn feh(&self) -> f64 {
        self.base.fehvec[0]
    }
    fn dfeh(&self) -> f64 {
        self.base.dfehvec[0]
    }
    fn eps(&self) -> f64 {
        self.base.epsvec[0]
    }
    fn sigma_v(&self) -> [f64; 3] {
        self.base.sigma_v
    }

    fn set_pos_b(&self, pos: &[f64; 3]) -> ([f64; 3], f64) {
        self.base.set_pos_b(pos)
    }

    fn density(&self, pos: &[f64; 3], age: f64) -> f64 {
        let (pos_b, k_flare) = self.set_pos_b(pos);
        // Option E=1 uses age-dependent flattening; Option E=0 uses fixed eps (Population.h:205-211)
        let eps1 = if self.option_e == 1 {
            self.base.eps_func(age)
        } else {
            self.eps()
        };
        let a2 = pos_b[0] * pos_b[0]
            + pos_b[1] * pos_b[1]
            + (pos_b[2] * pos_b[2]) / (eps1 * eps1 * k_flare * k_flare);
        (self.rho0 / eps1) * ((-a2 / self.hr1).exp() - (-a2 / self.hr2).exp()) / k_flare
    }

    fn amr(&self, pos: &[f64; 3], age: f64) -> f64 {
        self.base.amr(pos, age)
    }

    fn sigmar_func(&self, age: f64, r_c: f64) -> f64 {
        self.base.sigmar_func(age, r_c)
    }

    fn asymmetric_drift(&self, pos: &[f64; 3], v_c: f64, age: f64, sigma_r: f64) -> f64 {
        let rc = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
        let sig = self.sigma_v();
        let mut temp = (1.0 - sig[1] * sig[1] / (sig[0] * sig[0]))
            + (1.0 - sig[2] * sig[2] / (sig[0] * sig[0]));
        temp += self.dlnrho_dln_r(pos, age) - 2.0 * self.base.qparam * rc / 2.5;
        temp *= sigma_r * sigma_r * 0.5 / v_c;
        -temp
    }
}

// ----------------------------------------------------------------------------
// Population 1..6: ThinDisk1
// ----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ThinDisk1 {
    pub id: usize,
    pub base: ThinDiskBase,
    pub option_e: usize,
    pub rho0: f64,
    pub hr1: f64,
    pub hr2: f64,
}

impl ThinDisk1 {
    pub fn new(id: usize, warp_flare_on: usize, option_e: usize) -> Self {
        assert!(id >= 1 && id <= 6, "ThinDisk1 requires popID in 1..=6");
        let base = ThinDiskBase::new(warp_flare_on);
        let hr1 = 2.530;
        let hr2 = 1.320;
        let mut rho0 = base.sfr / (23.719602 * (hr1 * hr1 * hr1 - hr2 * hr2 * hr2));
        if id == 6 {
            rho0 /= 1.25;
        }
        if option_e == 0 {
            rho0 *= 2.0 * base.dagevec[id];
        }

        Self {
            id,
            base,
            option_e,
            rho0,
            hr1: hr1 * hr1, // 6.4009
            hr2: hr2 * hr2, // 1.7424
        }
    }

    fn dlnrho_dln_r(&self, pos: &[f64; 3], age: f64) -> f64 {
        let dr = 1e-3;
        let p2 = [pos[0] * (1.0 + dr), pos[1] * (1.0 + dr), pos[2]];
        let p1 = [pos[0] * (1.0 - dr), pos[1] * (1.0 - dr), pos[2]];
        let d2 = self.density(&p2, age);
        let d1 = self.density(&p1, age);
        (d2 - d1) / (dr * (d1 + d2))
    }
}

impl Population for ThinDisk1 {
    fn id(&self) -> usize {
        self.id
    }
    fn option_e(&self) -> usize {
        self.option_e
    }
    fn age(&self) -> f64 {
        self.base.agevec[self.id]
    }
    fn dage(&self) -> f64 {
        self.base.dagevec[self.id]
    }
    fn feh(&self) -> f64 {
        self.base.fehvec[self.id]
    }
    fn dfeh(&self) -> f64 {
        self.base.dfehvec[self.id]
    }
    fn eps(&self) -> f64 {
        self.base.epsvec[self.id]
    }
    fn sigma_v(&self) -> [f64; 3] {
        self.base.sigma_v
    }

    fn set_pos_b(&self, pos: &[f64; 3]) -> ([f64; 3], f64) {
        self.base.set_pos_b(pos)
    }

    fn density(&self, pos: &[f64; 3], age: f64) -> f64 {
        let (pos_b, k_flare) = self.set_pos_b(pos);
        // Option E=1 uses age-dependent flattening; Option E=0 uses fixed eps (Population.h:205-211)
        let eps1 = if self.option_e == 1 {
            self.base.eps_func(age)
        } else {
            self.eps()
        };
        let a2 = pos_b[0] * pos_b[0]
            + pos_b[1] * pos_b[1]
            + (pos_b[2] * pos_b[2]) / (eps1 * eps1 * k_flare * k_flare);
        let term1 = (-(0.25 + a2 / self.hr1).sqrt()).exp();
        let term2 = (-(0.25 + a2 / self.hr2).sqrt()).exp();
        (self.rho0 / eps1) * (term1 - term2) / k_flare
    }

    fn amr(&self, pos: &[f64; 3], age: f64) -> f64 {
        self.base.amr(pos, age)
    }

    fn sigmar_func(&self, age: f64, r_c: f64) -> f64 {
        self.base.sigmar_func(age, r_c)
    }

    fn asymmetric_drift(&self, pos: &[f64; 3], v_c: f64, age: f64, sigma_r: f64) -> f64 {
        let rc = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
        let sig = self.sigma_v();
        let mut temp = (1.0 - sig[1] * sig[1] / (sig[0] * sig[0]))
            + (1.0 - sig[2] * sig[2] / (sig[0] * sig[0]));
        temp += self.dlnrho_dln_r(pos, age) - 2.0 * self.base.qparam * rc / 2.5;
        temp *= sigma_r * sigma_r * 0.5 / v_c;
        -temp
    }
}

// ----------------------------------------------------------------------------
// Population 7: ThickDisk
// ----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ThickDisk {
    pub rho0: f64,
    pub hr1: f64,
    pub hr2: f64,
    pub xl: f64,
    pub th_max: f64,
    pub g_warp: f64,
    pub r_warp: f64,
    pub r_flare: f64,
    pub g_flare: f64,
    pub qparam: f64,
    pub sigma_v: [f64; 3],
}

impl ThickDisk {
    pub fn new(warp_flare_on: usize) -> Self {
        let mut r_warp = 8.51 * RSUN / 8.5;
        let mut r_flare = 9.5 * RSUN / 8.5;
        if warp_flare_on == 0 {
            r_warp = 1e6;
            r_flare = 1e6;
        }

        let hr1 = 2.5;
        let hr2 = 0.8;
        let xl = 0.4;
        let base_rho0 = 1.34e6 * 1157.68;

        let mut td = Self {
            rho0: base_rho0,
            hr1,
            hr2,
            xl,
            th_max: PI_WARP * 0.5,
            g_warp: 0.18,
            r_warp,
            r_flare,
            g_flare: 5.4e-4,
            qparam: 0.33,
            sigma_v: [67.0, 51.0, 42.0],
        };

        // Normalize so density at pos_c matches base_rho0 exactly
        let pos_c = [-RSUN, 0.0, ZSUN];
        let d0 = td.density(&pos_c, 11e9) / base_rho0;
        if d0 > 0.0 {
            td.rho0 = base_rho0 / d0;
        }
        td
    }
}

impl Population for ThickDisk {
    fn id(&self) -> usize {
        7
    }
    fn option_e(&self) -> usize {
        0
    }
    fn age(&self) -> f64 {
        11.0e9
    }
    fn dage(&self) -> f64 {
        0.0
    }
    fn feh(&self) -> f64 {
        -0.78
    }
    fn dfeh(&self) -> f64 {
        0.30
    }
    fn eps(&self) -> f64 {
        0.0
    }
    fn sigma_v(&self) -> [f64; 3] {
        self.sigma_v
    }

    fn set_pos_b(&self, pos: &[f64; 3]) -> ([f64; 3], f64) {
        let r = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
        let mut th = pos[1].atan2(pos[0]);
        if th < 0.0 {
            th += 2.0 * PI_WARP;
        }

        let mut zw = pos[2];
        if r > self.r_warp {
            zw = pos[2] - self.g_warp * (r - self.r_warp) * (th - self.th_max).cos();
        }

        let k_flare = if r > self.r_flare {
            1.0 + self.g_flare * (r - self.r_flare)
        } else {
            1.0
        };

        ([pos[0], pos[1], zw], k_flare)
    }

    fn density(&self, pos: &[f64; 3], _age: f64) -> f64 {
        let (pos_b, k_flare) = self.set_pos_b(pos);
        let a2 = (pos_b[0] * pos_b[0] + pos_b[1] * pos_b[1]).sqrt();

        let hr2 = self.hr2;
        let xl = self.xl;
        let temp = (2.0 * hr2 + xl)
            * (3.0 * k_flare * hr2 * xl + 3.0 * k_flare * k_flare * hr2 * hr2 + xl * xl)
            / ((2.0 * hr2 * k_flare + xl) * (3.0 * hr2 * xl + 3.0 * hr2 * hr2 + xl * xl));

        let radial_part = self.rho0 * (-(a2 - RSUN) / self.hr1).exp();

        // C++ Galaxia quirk: condition tests unwarped |pos[2]|, formula uses warped pos_b[2]
        if pos[2].abs() < xl {
            radial_part * (1.0 - pos_b[2] * pos_b[2] / (xl * (2.0 * hr2 * k_flare + xl))) / temp
        } else {
            let z_scale = hr2 * k_flare;
            let factor = (xl / z_scale).exp() / (1.0 + 0.5 * xl / z_scale);
            radial_part * factor * (-pos_b[2].abs() / z_scale).exp() / temp
        }
    }

    fn amr(&self, _pos: &[f64; 3], _age: f64) -> f64 {
        self.feh()
    }

    fn sigmar_func(&self, _age: f64, _rc: f64) -> f64 {
        self.sigma_v[0]
    }

    fn asymmetric_drift(&self, pos: &[f64; 3], v_c: f64, _age: f64, sigma_r: f64) -> f64 {
        let rc = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
        let sig = self.sigma_v;
        let mut temp = (1.0 - sig[1] * sig[1] / (sig[0] * sig[0]))
            + (1.0 - sig[2] * sig[2] / (sig[0] * sig[0]));
        temp -= rc * (1.0 / self.hr1 + 2.0 * self.qparam / 2.5);
        temp *= sigma_r * sigma_r * 0.5 / v_c;
        -temp
    }
}

// ----------------------------------------------------------------------------
// Population 8: Spheroid (Halo)
// ----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Spheroid {
    pub rho0: f64,
    pub rhoc: f64,
    pub ac: f64,
    pub eps: f64,
    pub sigma_v: [f64; 3],
}

impl Spheroid {
    pub fn new() -> Self {
        let ac = 0.5;
        let eps = 0.76;
        let base_rho0 = 9.32e3 * 1407.52 * 1.1;

        let mut sph = Self {
            rho0: base_rho0,
            rhoc: 0.0,
            ac,
            eps,
            sigma_v: [135.0, 85.0, 85.0],
        };

        let pos_c = [-RSUN, 0.0, ZSUN];
        let d0 = sph.density(&pos_c, 14.0e9) / base_rho0;
        if d0 > 0.0 {
            sph.rho0 = base_rho0 / d0;
        }
        sph.rhoc = sph.rho0 * (ac / RSUN).powf(-2.44);
        sph
    }
}

impl Population for Spheroid {
    fn id(&self) -> usize {
        8
    }
    fn option_e(&self) -> usize {
        0
    }
    fn age(&self) -> f64 {
        14.0e9
    }
    fn dage(&self) -> f64 {
        0.0
    }
    fn feh(&self) -> f64 {
        -1.78
    }
    fn dfeh(&self) -> f64 {
        0.50
    }
    fn eps(&self) -> f64 {
        self.eps
    }
    fn sigma_v(&self) -> [f64; 3] {
        self.sigma_v
    }

    fn set_pos_b(&self, pos: &[f64; 3]) -> ([f64; 3], f64) {
        (*pos, 1.0)
    }

    fn density(&self, pos: &[f64; 3], _age: f64) -> f64 {
        let a =
            (pos[0] * pos[0] + pos[1] * pos[1] + (pos[2] * pos[2]) / (self.eps * self.eps)).sqrt();
        if a < self.ac {
            self.rhoc
        } else {
            self.rho0 * (a / RSUN).powf(-2.44)
        }
    }

    fn amr(&self, _pos: &[f64; 3], _age: f64) -> f64 {
        self.feh()
    }

    fn sigmar_func(&self, _age: f64, _rc: f64) -> f64 {
        self.sigma_v[0]
    }

    fn asymmetric_drift(&self, _pos: &[f64; 3], v_c: f64, _age: f64, _sigma_r: f64) -> f64 {
        // Galaxia Population.h:100: virtual asymmetricDrift returns v_c, so (v_c - v_a) == 0
        v_c
    }
}

// ----------------------------------------------------------------------------
// Population 9: Bulge
// ----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Bulge {
    pub rho0: f64,
    pub x0: f64,
    pub y0: f64,
    pub z0: f64,
    pub rc: f64,
    pub rot_matrix: Matrix3x3,
    pub sigma_v: [f64; 3],
}

impl Bulge {
    pub fn new() -> Self {
        let alpha = 78.9;
        let beta = 3.5;
        let gamma = 91.3;

        // In C++ Galaxia (Population.h:795-797):
        // TM = Cot::RotationMatrix(0, gamma * PI / 180.0);
        // TM *= Cot::RotationMatrix(1, -beta * PI / 180.0);
        // TM *= Cot::RotationMatrix(2, (alpha - 90.0) * PI / 180.0);
        let m0 = Matrix3x3::rotation_axis(0, gamma * DEG2RAD);
        let m1 = Matrix3x3::rotation_axis(1, -beta * DEG2RAD);
        let m2 = Matrix3x3::rotation_axis(2, (alpha - 90.0) * DEG2RAD);

        let rot_matrix = m0.mul(&m1).mul(&m2);

        Self {
            rho0: 0.255 * 13.7e9,
            x0: 1.59,
            y0: 0.424,
            z0: 0.424,
            rc: 2.54,
            rot_matrix,
            sigma_v: [110.0, 110.0, 100.0],
        }
    }
}

impl Population for Bulge {
    fn id(&self) -> usize {
        9
    }
    fn option_e(&self) -> usize {
        0
    }
    fn age(&self) -> f64 {
        10.0e9
    }
    fn dage(&self) -> f64 {
        0.0
    }
    fn feh(&self) -> f64 {
        0.0
    }
    fn dfeh(&self) -> f64 {
        0.40
    }
    fn eps(&self) -> f64 {
        0.0
    }
    fn sigma_v(&self) -> [f64; 3] {
        self.sigma_v
    }

    fn set_pos_b(&self, pos: &[f64; 3]) -> ([f64; 3], f64) {
        let pos_b = self.rot_matrix.mult_vec(pos);
        (pos_b, 1.0)
    }

    fn density(&self, pos: &[f64; 3], _age: f64) -> f64 {
        let (pos_b, _) = self.set_pos_b(pos);
        let term_xy =
            pos_b[0] * pos_b[0] / (self.x0 * self.x0) + pos_b[1] * pos_b[1] / (self.y0 * self.y0);
        let term_z = pos_b[2] * pos_b[2] / (self.z0 * self.z0);
        let a2 = (term_xy * term_xy + term_z * term_z).sqrt();

        let cut = (pos_b[0] * pos_b[0] + pos_b[1] * pos_b[1]).sqrt() - self.rc;
        if cut < 0.0 {
            self.rho0 * (-0.5 * a2).exp()
        } else {
            self.rho0 * (-0.5 * a2).exp() * (-cut * cut / 0.5).exp()
        }
    }

    fn amr(&self, _pos: &[f64; 3], _age: f64) -> f64 {
        self.feh()
    }

    fn sigmar_func(&self, _age: f64, _rc: f64) -> f64 {
        self.sigma_v[0]
    }

    fn asymmetric_drift(&self, pos: &[f64; 3], v_c: f64, _age: f64, _sigma_r: f64) -> f64 {
        let rc = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
        v_c - rc * 71.62
    }
}

/// Create a heap-allocated `Population` trait object for the specified `pop_id` (0..9).
pub fn create_population(
    pop_id: usize,
    warp_flare_on: usize,
    option_e: usize,
) -> Box<dyn Population> {
    match pop_id {
        0 => Box::new(ThinDisk0::new(warp_flare_on, option_e)),
        1..=6 => Box::new(ThinDisk1::new(pop_id, warp_flare_on, option_e)),
        7 => Box::new(ThickDisk::new(warp_flare_on)),
        8 => Box::new(Spheroid::new()),
        9 => Box::new(Bulge::new()),
        _ => panic!("Invalid pop_id: {}, must be 0..=9", pop_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_10_populations_create_and_density() {
        let pos = [-8.0, 0.0, 0.015]; // Solar position
        for id in 0..=9 {
            let opt_e = if id <= 6 { 1 } else { 0 };
            let pop = create_population(id, 1, opt_e);
            assert_eq!(pop.id(), id);
            assert_eq!(pop.option_e(), opt_e);

            let rho = pop.density(&pos, pop.age());
            assert!(
                rho > 0.0 && rho.is_finite(),
                "Pop {} density at solar pos must be positive and finite: {}",
                id,
                rho
            );
        }
    }

    #[test]
    fn test_warp_and_flare() {
        let pop = ThinDiskBase::new(1);
        // Inside warp and flare radius (R < 7.9) -> no warp, no flare
        let (pos_b, k_flare) = pop.set_pos_b(&[5.0, 0.0, 0.2]);
        assert_eq!(pos_b, [5.0, 0.0, 0.2]);
        assert_eq!(k_flare, 1.0);

        // Outside flare radius (R = 12 kpc, R_flare ~ 8.94 kpc) -> k_flare > 1.0
        let (_, k_flare_outer) = pop.set_pos_b(&[12.0, 0.0, 0.0]);
        assert!(k_flare_outer > 1.0);
    }

    #[test]
    fn test_spheroid_power_law_density() {
        let sph = Spheroid::new();
        // At R=10 kpc in plane: rho should scale as R^-2.44
        let rho10 = sph.density(&[10.0, 0.0, 0.0], 14e9);
        let rho20 = sph.density(&[20.0, 0.0, 0.0], 14e9);
        let ratio = rho10 / rho20;
        let expected_ratio = 2.0f64.powf(2.44);
        assert!((ratio - expected_ratio).abs() < 1e-6);
    }
}
