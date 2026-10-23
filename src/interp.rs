// SPDX-License-Identifier: AGPL-3.0-or-later
// Original C++ code Copyright (c) 2012 Sanjib Sharma and Joss Bland-Hawthorn
// Ported from Interp.h, Interp.cpp, and GInterpolator.h
//! Numerical interpolation routines and tabulated grid helpers.
//!
//! Matches Galaxia 0.7.2's `Interp.h`, `Interp.cpp`, and `GInterpolator.h`.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Find index `i` such that `x[i] <= x1 < x[i+1]` in an ascending sorted slice.
/// Clamped to `[0, x.len() - 2]`.
#[inline]
pub fn locate(x: &[f64], x1: f64) -> usize {
    if x.len() < 2 || x1 <= x[0] {
        return 0;
    }
    let last = x.len() - 1;
    if x1 >= x[last] {
        return last - 1;
    }
    // partition_point finds first index where x[i] > x1
    let p = x.partition_point(|v| *v <= x1);
    let i = p.saturating_sub(1);
    i.min(last - 1)
}

/// Find nearest index `i` in an ascending sorted slice `x`.
/// Returns `0` if `x1 <= x[0]`, `x.len() - 1` if `x1 >= x[last]`.
#[inline]
pub fn locate_nearest(x: &[f64], x1: f64) -> usize {
    if x.is_empty() {
        return 0;
    }
    if x.len() == 1 || x1 <= x[0] {
        return 0;
    }
    let last = x.len() - 1;
    if x1 >= x[last] {
        return last;
    }
    let i = locate(x, x1);
    let frac = (x1 - x[i]) / (x[i + 1] - x[i]);
    if frac < 0.5 {
        i
    } else {
        i + 1
    }
}

/// Linear 1D interpolation with endpoint clamping.
/// Matches `Interp.h::interpolate(x, y, x1)`.
#[inline]
pub fn interpolate(x: &[f64], y: &[f64], x1: f64) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    if x.is_empty() {
        return 0.0;
    }
    if x.len() == 1 {
        return y[0];
    }
    if x1 >= x[x.len() - 1] {
        return y[y.len() - 1];
    }
    if x1 <= x[0] {
        return y[0];
    }
    let i = locate(x, x1);
    let dx = x[i + 1] - x[i];
    if dx.abs() < 1e-30 {
        return y[i];
    }
    y[i] + (y[i + 1] - y[i]) * (x1 - x[i]) / dx
}

/// Tabulated circular velocity curve `vcirc(R)`.
/// Matches Galaxia `Interp` over `Model/vcirc.dat`.
#[derive(Debug, Clone)]
pub struct VCirc {
    pub r: Vec<f64>,
    pub v: Vec<f64>,
}

impl Default for VCirc {
    fn default() -> Self {
        // Flat circular velocity fallback (v = 226.84 km/s)
        Self {
            r: vec![0.0, 1000.0],
            v: vec![226.84, 226.84],
        }
    }
}

impl VCirc {
    /// Read `vcirc.dat` ASCII table.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let f = File::open(path)?;
        let reader = BufReader::new(f);
        let mut tokens = Vec::new();
        for line in reader.lines() {
            let line = line?;
            for part in line.split_whitespace() {
                tokens.push(part.to_string());
            }
        }
        if tokens.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Empty vcirc.dat file",
            ));
        }

        let n: usize = tokens[0]
            .parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        if tokens.len() < 1 + 2 * n {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "Not enough tokens in vcirc.dat: expected {}, found {}",
                    1 + 2 * n,
                    tokens.len()
                ),
            ));
        }

        let mut r = Vec::with_capacity(n);
        for tok in &tokens[1..1 + n] {
            r.push(
                tok.parse::<f64>()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
            );
        }

        let mut v = Vec::with_capacity(n);
        for tok in &tokens[1 + n..1 + 2 * n] {
            v.push(
                tok.parse::<f64>()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
            );
        }

        Ok(Self { r, v })
    }

    /// Evaluate circular velocity at Galactocentric radius `r_kpc`.
    #[inline]
    pub fn f(&self, r_kpc: f64) -> f64 {
        interpolate(&self.r, &self.v, r_kpc)
    }
}

/// Regular grid linear interpolator for 1D, 2D, or 3D datasets.
/// Matches Galaxia's `GInterpolator`.
#[derive(Debug, Clone)]
pub struct GInterpolator {
    pub data: Vec<f32>,
    pub dims: Vec<usize>,
    /// Per-dimension: [min, max, step]
    pub xmms: Vec<[f32; 3]>,
}

impl GInterpolator {
    pub fn new(data: Vec<f32>, dims: Vec<usize>, xmms: Vec<[f32; 3]>) -> Self {
        Self { data, dims, xmms }
    }

    /// Interpolate in multi-dimensional space.
    pub fn interpol(&self, pos: &[f64]) -> f64 {
        if self.dims.is_empty() || self.data.is_empty() {
            return 0.0;
        }

        let mut key = vec![0.0f64; self.dims.len()];
        for i in 0..self.dims.len() {
            let p = if i < pos.len() { pos[i] as f32 } else { 0.0 };
            let min = self.xmms[i][0];
            let step = self.xmms[i][2];
            let mut k = (p - min) / step;
            if k < 0.0 {
                k = 0.0;
            }
            let max_k = (self.dims[i] - 1) as f32;
            if k >= max_k {
                k = (self.dims[i] as f32) - 2.0 + 0.999999;
            }
            key[i] = k as f64;
        }

        if self.dims.len() == 1 {
            let idx = key[0].floor() as usize;
            let frac = key[0] - (idx as f64);
            let idx2 = (idx + 1).min(self.data.len() - 1);
            let d0 = self.data[idx] as f64;
            let d1 = self.data[idx2] as f64;
            d0 * (1.0 - frac) + d1 * frac
        } else if self.dims.len() == 2 {
            let mut val = 0.0;
            let k0 = key[0];
            let k1 = key[1];
            let i0 = k0.floor() as usize;
            let i1 = k1.floor() as usize;
            for i in 0..2 {
                for j in 0..2 {
                    let idx0 = (i0 + i).min(self.dims[0] - 1);
                    let idx1 = (i1 + j).min(self.dims[1] - 1);
                    let index = idx0 * self.dims[1] + idx1;
                    let w0 = (1.0 - (k0 - (i0 as f64)) - (i as f64)).abs();
                    let w1 = (1.0 - (k1 - (i1 as f64)) - (j as f64)).abs();
                    val += (self.data[index] as f64) * w0 * w1;
                }
            }
            val
        } else if self.dims.len() == 3 {
            let mut val = 0.0;
            let k0 = key[0];
            let k1 = key[1];
            let k2 = key[2];
            let i0 = k0.floor() as usize;
            let i1 = k1.floor() as usize;
            let i2 = k2.floor() as usize;
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        let idx0 = (i0 + i).min(self.dims[0] - 1);
                        let idx1 = (i1 + j).min(self.dims[1] - 1);
                        let idx2 = (i2 + k).min(self.dims[2] - 1);
                        let index = idx0 * self.dims[1] * self.dims[2] + idx1 * self.dims[2] + idx2;
                        let w0 = (1.0 - (k0 - (i0 as f64)) - (i as f64)).abs();
                        let w1 = (1.0 - (k1 - (i1 as f64)) - (j as f64)).abs();
                        let w2 = (1.0 - (k2 - (i2 as f64)) - (k as f64)).abs();
                        val += (self.data[index] as f64) * w0 * w1 * w2;
                    }
                }
            }
            val
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_locate_and_locate_nearest() {
        let x = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        assert_eq!(locate(&x, 0.5), 0);
        assert_eq!(locate(&x, 1.0), 0);
        assert_eq!(locate(&x, 1.5), 0);
        assert_eq!(locate(&x, 2.0), 1);
        assert_eq!(locate(&x, 3.0), 1);
        assert_eq!(locate(&x, 20.0), 3);

        assert_eq!(locate_nearest(&x, 0.5), 0);
        assert_eq!(locate_nearest(&x, 1.4), 0);
        assert_eq!(locate_nearest(&x, 1.6), 1);
        assert_eq!(locate_nearest(&x, 2.9), 1);
        assert_eq!(locate_nearest(&x, 3.1), 2);
        assert_eq!(locate_nearest(&x, 20.0), 4);
    }

    #[test]
    fn test_interpolate_1d() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 10.0, 20.0, 30.0];
        assert_eq!(interpolate(&x, &y, -1.0), 0.0);
        assert_eq!(interpolate(&x, &y, 0.5), 5.0);
        assert_eq!(interpolate(&x, &y, 1.0), 10.0);
        assert_eq!(interpolate(&x, &y, 2.75), 27.5);
        assert_eq!(interpolate(&x, &y, 4.0), 30.0);
    }

    #[test]
    fn test_ginterpolator_1d() {
        let data = vec![10.0, 20.0, 30.0, 40.0];
        let dims = vec![4];
        let xmms = vec![[0.0, 3.0, 1.0]];
        let gi = GInterpolator::new(data, dims, xmms);
        let val = gi.interpol(&[1.5]);
        assert!((val - 25.0).abs() < 1e-4);
    }
}
