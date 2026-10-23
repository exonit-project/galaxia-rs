// SPDX-License-Identifier: AGPL-3.0-or-later
// Original C++ code Copyright (c) 2012 Sanjib Sharma and Joss Bland-Hawthorn
// Ported from GInterpolator.h, GInterpolator.cpp, and Galaxia.cpp
//! Extinction modeling and multi-dimensional grid interpolation.
//!
//! Matches C++ Galaxia `GInterpolator.h`, `GInterpolator.cpp`, and `Galaxia.cpp` (`append_ext`).
//!
//! - `GInterpolator`: trilinear and bilinear interpolator over regular grids stored in EBF format.
//! - `ExtinctionMap`: reads 3D/2D extinction maps from:
//!   - `Extinction/ExMap3d_1024.ebf` (`/ExMap3d` and `/ExMap2d`)
//!   - `Extinction/Schlegel_4096.ebf` (`/ExMap2d`)
//! - Evaluates:
//!   $$E(B-V)_{\mathrm{solar}} = f_{\mathrm{ext}}(l, b, \log_{10} d) \times E(B-V)_{\odot,\infty}(l, b)$$
//!   $$E(B-V)_{\mathrm{schlegel}} = f_{\mathrm{ext}}(l, b, \log_{10} d) \times E(B-V)_{\mathrm{SFD},\infty}(l, b)$$

use crate::coords::xyz_to_lbr_deg;
use crate::ebf::{EbfError, EbfFile};
use std::path::Path;

/// Multi-dimensional regular grid interpolator matching `GInterpolator` in `GInterpolator.h`.
///
/// Supports 1D, 2D (bilinear), and 3D (trilinear) interpolation on rectilinear grids
/// with coordinate bounds given by `xmms` `[min, max, step]`.
#[derive(Debug, Clone, PartialEq)]
pub struct GInterpolator {
    /// Flattened data array (row-major C-order: dim[0] fastest stride = dim[1]*dim[2])
    pub data: Vec<f32>,
    /// Dimensions for each axis
    pub dims: Vec<usize>,
    /// Axis coordinate specifications: `[min, max, step]` for each dimension
    pub xmms: Vec<[f32; 3]>,
}

impl GInterpolator {
    /// Create a new `GInterpolator` from raw data, dimensions, and axis bounds.
    pub fn new(data: Vec<f32>, dims: Vec<usize>, xmms: Vec<[f32; 3]>) -> Self {
        Self { data, dims, xmms }
    }

    /// Read a `GInterpolator` dataset from an EBF file given its tag name (e.g. `"/ExMap3d"` or `"/ExMap2d"`).
    ///
    /// Reads:
    /// - Data: `{tagname}.data` or `{tagname}`
    /// - Grid bounds: `{tagname}.xmms`
    pub fn read_from_file(path: impl AsRef<Path>, tagname: &str) -> Result<Self, EbfError> {
        let mut ebf = EbfFile::open(path)?;
        Self::read_from_ebf(&mut ebf, tagname)
    }

    /// Read a `GInterpolator` dataset from an already opened `EbfFile`.
    pub fn read_from_ebf(ebf: &mut EbfFile, tagname: &str) -> Result<Self, EbfError> {
        let tag_clean = tagname.trim_end_matches(".data");
        let data_tag = format!("{}.data", tag_clean);
        let xmms_tag = format!("{}.xmms", tag_clean);

        let ds = if ebf.contains_key(&data_tag) {
            ebf.read(&data_tag)?
        } else if ebf.contains_key(tagname) {
            ebf.read(tagname)?
        } else {
            return Err(EbfError::DatasetNotFound(data_tag));
        };

        let dims = ds.dims.clone();
        let data = ds.data.as_f32_vec()?;

        let xmms_ds = if ebf.contains_key(&xmms_tag) {
            ebf.read(&xmms_tag)?
        } else {
            return Err(EbfError::DatasetNotFound(xmms_tag));
        };

        let xmms_raw = xmms_ds.data.as_f32_vec()?;
        let num_dims = dims.len();

        let mut xmms = Vec::with_capacity(num_dims);
        if num_dims > 1 {
            let cols = if xmms_ds.dims.len() > 1 {
                xmms_ds.dims[1]
            } else {
                xmms_raw.len() / num_dims
            };
            for i in 0..num_dims {
                let off = i * cols;
                let min = xmms_raw[off];
                let max = if cols > 1 { xmms_raw[off + 1] } else { min };
                let step = if cols > 2 { xmms_raw[off + 2] } else { 1.0 };
                xmms.push([min, max, step]);
            }
        } else {
            let min = xmms_raw[0];
            let max = if xmms_raw.len() > 1 { xmms_raw[1] } else { min };
            let step = if xmms_raw.len() > 2 { xmms_raw[2] } else { 1.0 };
            xmms.push([min, max, step]);
        }

        Ok(Self { data, dims, xmms })
    }

    /// Interpolate value at coordinates `pos` matching `GInterpolator::interpol` in `GInterpolator.h:103-154`.
    ///
    /// For 2D grids, `pos[0..2]` = `[glon, glat]`.
    /// For 3D grids, `pos[0..3]` = `[glon, glat, log10(d)]`.
    pub fn interpol(&self, pos: &[f64]) -> f64 {
        let n = self.dims.len();
        if n == 0 || self.data.is_empty() {
            return 0.0;
        }

        let mut key = vec![0.0f64; n];
        for i in 0..n {
            let mut k = (pos[i] - self.xmms[i][0] as f64) / (self.xmms[i][2] as f64);
            if k < 0.0 {
                k = 0.0;
            }
            let max_k = (self.dims[i] - 1) as f64;
            if k >= max_k {
                k = (self.dims[i] as f64 - 2.0) + 0.999999;
            }
            key[i] = k;
        }

        match n {
            1 => {
                let i0 = key[0] as usize;
                let f0 = key[0] - i0 as f64;
                (self.data[i0] as f64) * (1.0 - f0) + (self.data[i0 + 1] as f64) * f0
            }
            2 => {
                let i0 = key[0] as usize;
                let j0 = key[1] as usize;
                let f0 = key[0] - i0 as f64;
                let f1 = key[1] - j0 as f64;

                let d1 = self.dims[1];
                let mut value = 0.0;
                for i in 0..2 {
                    let w0 = (1.0 - f0 - i as f64).abs();
                    for j in 0..2 {
                        let w1 = (1.0 - f1 - j as f64).abs();
                        let idx = (i0 + i) * d1 + (j0 + j);
                        value += (self.data[idx] as f64) * w0 * w1;
                    }
                }
                value
            }
            3 => {
                let i0 = key[0] as usize;
                let j0 = key[1] as usize;
                let k0 = key[2] as usize;
                let f0 = key[0] - i0 as f64;
                let f1 = key[1] - j0 as f64;
                let f2 = key[2] - k0 as f64;

                let d1 = self.dims[1];
                let d2 = self.dims[2];
                let mut value = 0.0;
                for i in 0..2 {
                    let w0 = (1.0 - f0 - i as f64).abs();
                    for j in 0..2 {
                        let w1 = (1.0 - f1 - j as f64).abs();
                        for k in 0..2 {
                            let w2 = (1.0 - f2 - k as f64).abs();
                            let idx = ((i0 + i) * d1 + (j0 + j)) * d2 + (k0 + k);
                            value += (self.data[idx] as f64) * w0 * w1 * w2;
                        }
                    }
                }
                value
            }
            _ => 0.0,
        }
    }

    /// Interpolate value directly from Cartesian heliocentric position `[x, y, z]` in kpc.
    ///
    /// Matches `GInterpolator::interpol_xyz` in `GInterpolator.h:91-102`.
    pub fn interpol_xyz(&self, pos_xyz: &[f64; 3]) -> f64 {
        let (l, b, r) = xyz_to_lbr_deg(pos_xyz);
        let log_r = if r > 0.0 { r.log10() } else { -2.0 };
        let pos_s = [l, b, log_r];
        self.interpol(&pos_s)
    }
}

/// Extinction values evaluated for a star.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtinctionValues {
    /// $E(B-V)$ with SFD98 asymptotic scaling: $f_{\mathrm{ext}} \times E(B-V)_{\mathrm{SFD},\infty}$
    pub exbv_schlegel: f32,
    /// $E(B-V)$ with solar asymptotic scaling: $f_{\mathrm{ext}} \times E(B-V)_{\odot,\infty}$
    pub exbv_solar: f32,
    /// Full asymptotic SFD98 $E(B-V)$ at infinity
    pub exbv_schlegel_inf: f32,
    /// Galactic longitude in degrees $[0, 360)$
    pub glon: f32,
    /// Galactic latitude in degrees $[-90, 90]$
    pub glat: f32,
}

/// Extinction service combining 3D and 2D dust maps.
pub struct ExtinctionMap {
    /// 3D extinction fraction map $f_{\mathrm{ext}}(l, b, \log_{10} d)$ (`ExMap3d_1024.ebf:/ExMap3d`)
    pub gi_3d: GInterpolator,
    /// Solar asymptotic 2D normalization $E(B-V)_{\odot,\infty}(l, b)$ (`ExMap3d_1024.ebf:/ExMap2d`)
    pub ge_solar: GInterpolator,
    /// SFD98 asymptotic 2D map $E(B-V)_{\mathrm{SFD},\infty}(l, b)$ (`Schlegel_4096.ebf:/ExMap2d`)
    pub ge_schlegel: GInterpolator,
}

impl ExtinctionMap {
    /// Load extinction maps from standard Galaxia data directory.
    ///
    /// Expects:
    /// - `{data_dir}/Extinction/ExMap3d_1024.ebf`
    /// - `{data_dir}/Extinction/Schlegel_4096.ebf`
    pub fn load_from_dir(data_dir: impl AsRef<Path>) -> Result<Self, EbfError> {
        let dir = data_dir.as_ref();
        let exmap3d_path = dir.join("Extinction/ExMap3d_1024.ebf");
        let schlegel_path = dir.join("Extinction/Schlegel_4096.ebf");

        let mut ebf_3d = EbfFile::open(&exmap3d_path)?;
        let gi_3d = GInterpolator::read_from_ebf(&mut ebf_3d, "/ExMap3d")?;
        let ge_solar = GInterpolator::read_from_ebf(&mut ebf_3d, "/ExMap2d")?;

        let mut ebf_schlegel = EbfFile::open(&schlegel_path)?;
        let ge_schlegel = GInterpolator::read_from_ebf(&mut ebf_schlegel, "/ExMap2d")?;

        Ok(Self {
            gi_3d,
            ge_solar,
            ge_schlegel,
        })
    }

    /// Compute extinction parameters for a star given its heliocentric Cartesian position $[x, y, z]$ in kpc.
    ///
    /// Matches `Galaxia.cpp:60-77` (`append_ext`):
    /// 1. $(l, b, r)$ from `xyz_to_lbr_deg`: $l \in [0, 360)^\circ$, $b \in [-90, 90]^\circ$, $r$ in kpc.
    /// 2. $d_{\log} = \log_{10}(r)$.
    /// 3. $f_{\mathrm{ext}} = \text{gi\_3d.interpol}(l, b, d_{\log})$.
    /// 4. $E(B-V)_{\mathrm{schlegel}} = f_{\mathrm{ext}} \times \text{ge\_schlegel.interpol}(l, b)$.
    /// 5. $E(B-V)_{\mathrm{solar}} = f_{\mathrm{ext}} \times \text{ge\_solar.interpol}(l, b)$.
    /// 6. $E(B-V)_{\mathrm{schlegel,inf}} = \text{ge\_schlegel.interpol}(l, b)$.
    pub fn compute(&self, pos_helio_kpc: &[f64; 3]) -> ExtinctionValues {
        let (glon, glat, r) = xyz_to_lbr_deg(pos_helio_kpc);
        let log_r = if r > 0.0 { r.log10() } else { -2.0 };
        let pos_s = [glon, glat, log_r];

        let f_ext = self.gi_3d.interpol(&pos_s);
        let sfd_inf = self.ge_schlegel.interpol(&pos_s);
        let solar_inf = self.ge_solar.interpol(&pos_s);

        ExtinctionValues {
            exbv_schlegel: (f_ext * sfd_inf) as f32,
            exbv_solar: (f_ext * solar_inf) as f32,
            exbv_schlegel_inf: sfd_inf as f32,
            glon: glon as f32,
            glat: glat as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpol_1d() {
        // Linear grid: x in [0.0, 10.0, step=2.0], dims=[6]
        // data = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
        let data = vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0];
        let dims = vec![6];
        let xmms = vec![[0.0, 10.0, 2.0]];
        let interp = GInterpolator::new(data, dims, xmms);

        // At exact grid points:
        assert!((interp.interpol(&[0.0]) - 0.0).abs() < 1e-6);
        assert!((interp.interpol(&[2.0]) - 20.0).abs() < 1e-6);
        assert!((interp.interpol(&[4.0]) - 40.0).abs() < 1e-6);

        // Halfway between 2.0 and 4.0: x = 3.0 -> 30.0
        assert!((interp.interpol(&[3.0]) - 30.0).abs() < 1e-6);

        // Clamping below min:
        assert!((interp.interpol(&[-5.0]) - 0.0).abs() < 1e-6);

        // Clamping above max:
        assert!((interp.interpol(&[15.0]) - 100.0).abs() < 1e-3);
    }

    #[test]
    fn test_interpol_2d_bilinear() {
        // 2x2 grid: x in [0, 1, step=1], y in [0, 1, step=1]
        // data: [0,0]=1.0, [0,1]=2.0, [1,0]=3.0, [1,1]=4.0
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let dims = vec![2, 2];
        let xmms = vec![[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]];
        let interp = GInterpolator::new(data, dims, xmms);

        // Corner tests (boundary has 0.999999 clamp matching C++ Galaxia)
        assert!((interp.interpol(&[0.0, 0.0]) - 1.0).abs() < 1e-5);
        assert!((interp.interpol(&[0.0, 1.0]) - 2.0).abs() < 1e-5);
        assert!((interp.interpol(&[1.0, 0.0]) - 3.0).abs() < 1e-5);
        assert!((interp.interpol(&[1.0, 1.0]) - 4.0).abs() < 1e-5);

        // Center test: (1 + 2 + 3 + 4) / 4 = 2.5
        assert!((interp.interpol(&[0.5, 0.5]) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_interpol_3d_trilinear() {
        // 2x2x2 unit cube:
        // z=0: [0,0,0]=0, [0,1,0]=2, [1,0,0]=4, [1,1,0]=6
        // z=1: [0,0,1]=1, [0,1,1]=3, [1,0,1]=5, [1,1,1]=7
        // value = 4*x + 2*y + 1*z
        let mut data = vec![0.0f32; 8];
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    let idx = (x * 2 + y) * 2 + z;
                    data[idx] = (4 * x + 2 * y + 1 * z) as f32;
                }
            }
        }
        let dims = vec![2, 2, 2];
        let xmms = vec![[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]];
        let interp = GInterpolator::new(data, dims, xmms);

        // Center test: x=0.5, y=0.5, z=0.5 -> 4(0.5) + 2(0.5) + 1(0.5) = 3.5
        assert!((interp.interpol(&[0.5, 0.5, 0.5]) - 3.5).abs() < 1e-6);

        // Arbitrary point: x=0.2, y=0.7, z=0.4 -> 4(0.2) + 2(0.7) + 1(0.4) = 0.8 + 1.4 + 0.4 = 2.6
        assert!((interp.interpol(&[0.2, 0.7, 0.4]) - 2.6).abs() < 1e-6);
    }

    #[test]
    fn test_real_extinction_maps_if_present() {
        let data_dir = crate::require_data_dir!();

        let ext = ExtinctionMap::load_from_dir(&data_dir).expect("Failed to load extinction maps");
        assert_eq!(ext.gi_3d.dims, vec![1024, 1024, 100]);
        assert_eq!(ext.ge_solar.dims, vec![1024, 1024]);
        assert_eq!(ext.ge_schlegel.dims, vec![4096, 4096]);

        // Evaluate at a known test direction: l = 60.0, b = 10.0, d = 1.0 kpc
        // pos_c = 1.0 * [cos(10)*cos(60), cos(10)*sin(60), sin(10)]
        let (glon, glat, r) = (60.0f64, 10.0f64, 1.0f64);
        let dtor = std::f64::consts::PI / 180.0;
        let pos = [
            r * (glat * dtor).cos() * (glon * dtor).cos(),
            r * (glat * dtor).cos() * (glon * dtor).sin(),
            r * (glat * dtor).sin(),
        ];

        let res = ext.compute(&pos);
        println!(
            "Extinction at (l=60, b=10, r=1 kpc): E(B-V)_schlegel={:.4}, E(B-V)_solar={:.4}",
            res.exbv_schlegel, res.exbv_solar
        );

        assert!((res.glon - 60.0).abs() < 0.01);
        assert!((res.glat - 10.0).abs() < 0.01);
        assert!(res.exbv_schlegel > 0.0);
        assert!(res.exbv_solar > 0.0);
        assert!(res.exbv_schlegel_inf > 0.0);
    }
}
