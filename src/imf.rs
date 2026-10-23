// SPDX-License-Identifier: AGPL-3.0-or-later
// Original C++ code Copyright (c) 2012 Sanjib Sharma and Joss Bland-Hawthorn
// Ported from Sampler.h, Sampler.cpp, and Population.cpp
//! Initial Mass Function (IMF) sampling and cumulative distribution functions.
//!
//! Matches Galaxia 0.7.2's `Sampler.h`, `Sampler.cpp`, and IMF formulas in `Population.cpp`.

use crate::interp::interpolate;
use crate::rng::Philox;

/// IMF for ThinDisk (popID 0..6).
/// Matches `exp_imf0` / `Chabrier_exp_imf0` from Galaxia 0.7.2.
#[inline]
pub fn exp_imf0(m: f64) -> f64 {
    if m < 1.0 {
        m.powf(-1.6)
    } else {
        m.powf(-3.0)
    }
}

/// Alias for `exp_imf0` matching the specification name.
#[inline]
pub fn chabrier_exp_imf0(m: f64) -> f64 {
    exp_imf0(m)
}

/// Continuous Chabrier exponential IMF.
/// Matches `Chabrier_exp_imf` in `Population.cpp:17`.
#[inline]
pub fn chabrier_exp_imf(m: f64) -> f64 {
    22.8978 * (-((716.4 / m).powf(0.25))).exp() * m.powf(-3.3)
}

/// IMF for ThickDisk (popID 7).
/// Matches `exp_imf7` in `Population.cpp:32`.
#[inline]
pub fn exp_imf7(m: f64) -> f64 {
    m.powf(-0.5)
}

/// IMF for Spheroid / Halo (popID 8).
/// Matches `exp_imf8` in `Population.cpp:37`.
#[inline]
pub fn exp_imf8(m: f64) -> f64 {
    m.powf(-0.5)
}

/// IMF for Bulge (popID 9).
/// Matches `exp_imf9` in `Population.cpp:42`.
#[inline]
pub fn exp_imf9(m: f64) -> f64 {
    m.powf(-2.35)
}

/// Numerical probability distribution sampler for initial stellar masses.
///
/// Implements trapezoidal integration, cumulative probability distribution (CPD) construction,
/// active mass window bounding, and inverse CDF sampling.
#[derive(Debug, Clone)]
pub struct Sampler {
    pub x: Vec<f64>,
    pub px: Vec<f64>,
    pub cpd: Vec<f64>,
    pub meanx: f64,
    pub x_min: f64,
    pub x_max: f64,
    pub cpd_min: f64,
    pub cpd_max: f64,
}

impl Sampler {
    /// Construct a new Sampler with `nsize` grid points between `xmin` and `xmax`.
    /// `optionlinlog == 1` for logarithmic spacing (default for Galaxia IMFs).
    pub fn new<F>(nsize: usize, xmin: f64, xmax: f64, func: F, optionlinlog: usize) -> Self
    where
        F: Fn(f64) -> f64,
    {
        assert!(nsize >= 2, "Sampler requires at least 2 points");
        assert!(xmax > xmin, "xmax must be greater than xmin");

        let mut x = Vec::with_capacity(nsize);
        if optionlinlog == 0 {
            // Linear spacing
            let step = (xmax - xmin) / ((nsize - 1) as f64);
            for i in 0..nsize {
                x.push(xmin + (i as f64) * step);
            }
        } else {
            // Logarithmic spacing: logspace(x, xmin, xmax, nsize)
            assert!(xmin > 0.0, "Logarithmic spacing requires xmin > 0");
            let log_min = xmin.log10();
            let log_max = xmax.log10();
            let step = (log_max - log_min) / ((nsize - 1) as f64);
            x.push(xmin);
            for i in 1..nsize - 1 {
                x.push(10.0f64.powf(log_min + (i as f64) * step));
            }
            x.push(xmax);
        }

        let mut px = Vec::with_capacity(nsize);
        for &xi in &x {
            px.push(func(xi));
        }

        let mut sampler = Self {
            x,
            px,
            cpd: Vec::new(),
            meanx: 0.0,
            x_min: xmin,
            x_max: xmax,
            cpd_min: 0.0,
            cpd_max: 0.0,
        };

        sampler.normalize();
        sampler
    }

    /// Factory for ThinDisk IMF (popID 0..6): `exp_imf0` on [0.07, 100.0] with 10,000 points.
    pub fn new_thin_disk() -> Self {
        Self::new(10000, 0.07, 100.0, exp_imf0, 1)
    }

    /// Factory for ThickDisk IMF (popID 7): `exp_imf7` on [0.07, 100.0] with 10,000 points.
    pub fn new_thick_disk() -> Self {
        Self::new(10000, 0.07, 100.0, exp_imf7, 1)
    }

    /// Factory for Spheroid IMF (popID 8): `exp_imf8` on [0.07, 100.0] with 10,000 points.
    pub fn new_spheroid() -> Self {
        Self::new(10000, 0.07, 100.0, exp_imf8, 1)
    }

    /// Factory for Bulge IMF (popID 9): `exp_imf9` on [0.07, 100.0] with 10,000 points.
    pub fn new_bulge() -> Self {
        Self::new(10000, 0.07, 100.0, exp_imf9, 1)
    }

    /// Factory by population ID (0..9).
    pub fn for_pop_id(pop_id: usize) -> Self {
        match pop_id {
            7 => Self::new_thick_disk(),
            8 => Self::new_spheroid(),
            9 => Self::new_bulge(),
            _ => Self::new_thin_disk(),
        }
    }

    /// Normalize the probability density `px` and compute `meanx` and cumulative distribution `cpd`.
    fn normalize(&mut self) {
        let n = self.x.len();
        let mut norm = 0.0;
        for i in 1..n {
            norm += (self.x[i] - self.x[i - 1]) * (self.px[i] + self.px[i - 1]) * 0.5;
        }

        if norm > 0.0 {
            for val in &mut self.px {
                *val /= norm;
            }
        }

        let mut meanx = 0.0;
        for i in 1..n {
            let p0 = self.px[i - 1] * self.x[i - 1];
            let p1 = self.px[i] * self.x[i];
            meanx += (self.x[i] - self.x[i - 1]) * (p0 + p1) * 0.5;
        }
        self.meanx = meanx;
        self.x_min = self.x[0];
        self.x_max = self.x[n - 1];

        self.calculate_cpd();
    }

    /// Calculate the Cumulative Probability Distribution.
    /// Matches `Sampler::calculateCpd()` in `Sampler.cpp:354`.
    fn calculate_cpd(&mut self) {
        let n = self.x.len();
        self.cpd = vec![0.0; n];

        if self.px[0] < self.px[n - 1] {
            // Ascending density: integrate forward from x_min
            self.cpd[0] = 0.0;
            let mut temp = 0.0;
            for i in 1..n {
                temp += (self.x[i] - self.x[i - 1]) * (self.px[i] + self.px[i - 1]) * 0.5;
                self.cpd[i] = temp;
            }
        } else {
            // Descending density (standard for stellar IMFs): integrate backwards from x_max
            self.cpd[n - 1] = 0.0;
            let mut temp = 0.0;
            for i in (0..n - 1).rev() {
                temp += (self.x[i + 1] - self.x[i]) * (self.px[i + 1] + self.px[i]) * 0.5;
                self.cpd[i] = temp;
            }
        }

        self.cpd_min = self.cpd[0];
        self.cpd_max = self.cpd[n - 1];
    }

    /// Evaluate cumulative fraction between mass limits: `|C(m1) - C(m2)|`.
    /// Matches `Sampler::getFac(xmin1, xmax1)` in `Sampler.cpp:328`.
    #[inline]
    pub fn get_fac(&self, xmin1: f64, xmax1: f64) -> f64 {
        let c1 = interpolate(&self.x, &self.cpd, xmin1);
        let c2 = interpolate(&self.x, &self.cpd, xmax1);
        (c1 - c2).abs()
    }

    /// C++ casing alias for `get_fac`.
    #[inline]
    #[allow(non_snake_case)]
    pub fn getFac(&self, xmin1: f64, xmax1: f64) -> f64 {
        self.get_fac(xmin1, xmax1)
    }

    /// Set active sampling range [xmin1, xmax1].
    /// Matches `Sampler::setRange(xmin1, xmax1)` in `Sampler.cpp:226`.
    pub fn set_range(&mut self, mut xmin1: f64, mut xmax1: f64) {
        if xmax1 < xmin1 {
            std::mem::swap(&mut xmin1, &mut xmax1);
        }

        let n = self.x.len();
        if xmax1 >= self.x_max {
            self.cpd_max = self.cpd[n - 1];
        } else if xmax1 <= self.x_min {
            self.cpd_max = self.cpd[0];
        } else {
            let up = self.x.partition_point(|v| *v <= xmax1).saturating_sub(1);
            let up = up.min(n - 2);
            self.cpd_max = self.cpd[up]
                + (self.cpd[up + 1] - self.cpd[up]) * (xmax1 - self.x[up])
                    / (self.x[up + 1] - self.x[up]);
        }

        if xmin1 >= self.x_max {
            self.cpd_min = self.cpd[n - 1];
        } else if xmin1 <= self.x_min {
            self.cpd_min = self.cpd[0];
        } else {
            let up = self.x.partition_point(|v| *v <= xmin1).saturating_sub(1);
            let up = up.min(n - 2);
            self.cpd_min = self.cpd[up]
                + (self.cpd[up + 1] - self.cpd[up]) * (xmin1 - self.x[up])
                    / (self.x[up + 1] - self.x[up]);
        }

        // Boundary adjustments matching C++ Galaxia (Jul 28 2009 update)
        if xmax1 <= self.x_min && n > 1 {
            self.cpd_min = self.cpd[1];
        }
        if xmin1 >= self.x_max && n > 2 {
            self.cpd_max = self.cpd[n - 2];
        }
    }

    /// Draw a random stellar mass given a uniform random deviate `u` in [0, 1).
    /// Matches `Sampler::rand()` in `Sampler.cpp:334`.
    pub fn sample_u(&self, uniform_draw: f64) -> f64 {
        let u = self.cpd_min + uniform_draw * (self.cpd_max - self.cpd_min);
        let n = self.cpd.len();

        let up = if self.cpd[0] == 0.0 {
            // Ascending CPD
            let p = self.cpd.partition_point(|v| *v <= u);
            p.saturating_sub(1).min(n - 2)
        } else {
            // Descending CPD: cpd[0] == 1.0, cpd[n-1] == 0.0
            // Find first element strictly less than u (i.e. cpd[i] < u)
            let p = self.cpd.partition_point(|v| *v >= u);
            p.saturating_sub(1).min(n - 2)
        };

        let denom = self.cpd[up + 1] - self.cpd[up];
        if denom.abs() < 1e-30 {
            self.x[up]
        } else {
            self.x[up] + (self.x[up + 1] - self.x[up]) * (u - self.cpd[up]) / denom
        }
    }

    /// Sample a random stellar mass using the given pseudo-random generator.
    #[inline]
    pub fn sample(&self, rng: &mut Philox) -> f64 {
        self.sample_u(rng.uniform_f64())
    }

    /// C++ matching method `rand()`.
    #[inline]
    pub fn rand(&self, rng: &mut Philox) -> f64 {
        self.sample(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thin_disk_sampler_normalisation_and_meanx() {
        let sampler = Sampler::new_thin_disk();
        // Integration domain: [0.07, 100.0] M_sun
        assert_eq!(sampler.x.len(), 10000);
        assert!((sampler.x_min - 0.07).abs() < 1e-6);
        assert!((sampler.x_max - 100.0).abs() < 1e-6);

        // In backwards cumulative integration, cpd[0] should be ~1.0, cpd.last() should be 0.0
        assert!((sampler.cpd[0] - 1.0).abs() < 1e-4);
        assert_eq!(sampler.cpd[sampler.cpd.len() - 1], 0.0);

        // getFac over full range should be 1.0
        let fac_full = sampler.get_fac(0.07, 100.0);
        assert!((fac_full - 1.0).abs() < 1e-4);

        // Active fraction in [0.1, 10.0]
        let fac_mid = sampler.get_fac(0.1, 10.0);
        assert!(fac_mid > 0.0 && fac_mid < 1.0);

        // mean mass for thin disk (Chabrier/exp_imf0) should be around 0.3 - 0.5 M_sun
        println!("Thin disk meanx: {}", sampler.meanx);
        assert!(sampler.meanx > 0.2 && sampler.meanx < 0.6);
    }

    #[test]
    fn test_sampling_bounds() {
        let mut sampler = Sampler::new_thin_disk();
        sampler.set_range(1.0, 5.0);

        let mut rng = Philox::from_stream(12345, 0, 0, 0, 0, 1);
        for _ in 0..1000 {
            let m = sampler.sample(&mut rng);
            assert!(
                m >= 0.999 && m <= 5.001,
                "Sampled mass out of bounds: {}",
                m
            );
        }
    }

    #[test]
    fn test_all_population_samplers() {
        let pops = [
            Sampler::new_thin_disk(),
            Sampler::new_thick_disk(),
            Sampler::new_spheroid(),
            Sampler::new_bulge(),
        ];
        for s in &pops {
            assert_eq!(s.x.len(), 10000);
            assert!((s.cpd[0] - 1.0).abs() < 1e-4);
            assert!(s.meanx > 0.0);
        }
    }
}
