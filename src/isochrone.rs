// SPDX-License-Identifier: AGPL-3.0-or-later
// Original C++ code Copyright (c) 2012 Sanjib Sharma and Joss Bland-Hawthorn
// Ported from Isochrone.h, Isochrone.cpp, IsochroneBase.h, IsochroneBase.cpp, IsochroneDB.h, IsochroneDB.cpp, and TableInterpolator.h
//! Padova isochrone grid reader, Chabrier low-mass dwarf interpolators, and stellar parameter lookups.
//!
//! Matches Galaxia 0.7.2's `Isochrone.h`, `Isochrone.cpp`, `IsochroneBase.h`, `IsochroneBase.cpp`,
//! `IsochroneDB.h`, `IsochroneDB.cpp`, and `TableInterpolator.h`.

use crate::imf::Sampler;
use crate::interp::{interpolate, locate, locate_nearest};
use crate::population::StarParticle;
use crate::rng::Philox;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Error function and complementary error function matching Numerical Recipes in Galaxia `IsochroneDB.h:19-88`.
pub struct ErfNR;

impl ErfNR {
    const COF: [f64; 28] = [
        -1.3026537197817094,
        6.4196979235649026e-1,
        1.9476473204185836e-2,
        -9.561514786808631e-3,
        -9.46595344482036e-4,
        3.66839497852761e-4,
        4.2523324806907e-5,
        -2.0278578112534e-5,
        -1.624290004647e-6,
        1.303655835580e-6,
        1.5626441722e-8,
        -8.5238095915e-8,
        6.529054439e-9,
        5.059343495e-9,
        -9.91364156e-10,
        -2.27365122e-10,
        9.6467911e-11,
        2.394038e-12,
        -6.886027e-12,
        8.94487e-13,
        3.13092e-13,
        -1.12708e-13,
        3.81e-16,
        7.106e-15,
        -1.523e-15,
        -9.4e-17,
        1.21e-16,
        -2.8e-17,
    ];

    #[inline]
    fn erfccheb(z: f64) -> f64 {
        let t = 2.0 / (2.0 + z);
        let ty = 4.0 * t - 2.0;
        let mut d = 0.0;
        let mut dd = 0.0;
        for &c in Self::COF[1..].iter().rev() {
            let tmp = d;
            d = ty * d - dd + c;
            dd = tmp;
        }
        t * (-z * z + 0.5 * (Self::COF[0] + ty * d) - dd).exp()
    }

    #[inline]
    pub fn erfc(x: f64) -> f64 {
        if x >= 0.0 {
            Self::erfccheb(x)
        } else {
            2.0 - Self::erfccheb(-x)
        }
    }

    /// Standard Gaussian CDF: $\Phi(z) = \frac{1}{2} \text{erfc}(-z / \sqrt{2})$.
    #[inline]
    pub fn normal_cdf(z: f64) -> f64 {
        0.5 * Self::erfc(-0.707106781186547524 * z)
    }
}

/// Extinction table row from `alavtab_*.dat`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlavRow {
    pub n: i32,
    pub teff: f64,
    pub logg: f64,
    pub av: f64,
    pub rv: f64,
    pub a_i: f64,
    pub a_j: f64,
    pub a_h: f64,
    pub a_ks: f64,
}

/// Reader for `alavtab_*.dat` extinction tables.
pub fn read_alavtab(path: impl AsRef<Path>) -> Result<Vec<AlavRow>, std::io::Error> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() >= 9 {
            if let (Ok(n), Ok(teff), Ok(logg), Ok(av), Ok(rv), Ok(ai), Ok(aj), Ok(ah), Ok(aks)) = (
                parts[0].parse::<i32>(),
                parts[1].parse::<f64>(),
                parts[2].parse::<f64>(),
                parts[3].parse::<f64>(),
                parts[4].parse::<f64>(),
                parts[5].parse::<f64>(),
                parts[6].parse::<f64>(),
                parts[7].parse::<f64>(),
                parts[8].parse::<f64>(),
            ) {
                rows.push(AlavRow {
                    n,
                    teff,
                    logg,
                    av,
                    rv,
                    a_i: ai,
                    a_j: aj,
                    a_h: ah,
                    a_ks: aks,
                });
            }
        }
    }
    Ok(rows)
}

/// 2D ASCII table representing mass or gravity columns.
#[derive(Debug, Clone, Default)]
pub struct AsciiTable2D {
    pub cols: Vec<Vec<f64>>,
}

impl AsciiTable2D {
    pub fn read_from_file(path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        let mut tokens = content.split_whitespace();

        // Find first "<head>"
        let mut found_first = false;
        while let Some(tok) = tokens.next() {
            if tok == "<head>" {
                found_first = true;
                break;
            }
        }
        if !found_first {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Missing opening <head>",
            ));
        }

        let _rows: usize = tokens
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing rows"))?
            .parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let dim1: usize = tokens
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing dim1"))?
            .parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let _massid: i32 = tokens
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing massid"))?
            .parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let columns: usize = tokens
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing columns"))?
            .parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut col_indices = Vec::with_capacity(columns);
        for _ in 0..columns {
            let idx: usize = tokens
                .next()
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "missing dimid")
                })?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            col_indices.push(idx);
        }

        let next_head = tokens.next().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing closing <head>")
        })?;
        if next_head != "<head>" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected closing <head>, got {}", next_head),
            ));
        }

        let mut cols: Vec<Vec<f64>> = vec![Vec::new(); columns];
        let mut row_buf = Vec::with_capacity(dim1);

        for tok in tokens {
            if tok.starts_with('#') {
                continue;
            }
            // Parse as f32 then promote to f64 matching C++ TableInterpolator.cpp:159-161
            if let Ok(v) = tok.parse::<f32>() {
                row_buf.push(v as f64);
                if row_buf.len() == dim1 {
                    for (k, &dim_idx) in col_indices.iter().enumerate() {
                        if dim_idx < row_buf.len() {
                            cols[k].push(row_buf[dim_idx]);
                        }
                    }
                    row_buf.clear();
                }
            }
        }

        Ok(Self { cols })
    }
}

/// Chabrier low-mass star interpolator (`Isochrones/Chabrier/`).
#[derive(Debug, Clone)]
pub struct ChabrierInterpolator {
    pub age_keys: Vec<f64>,
    pub tables: Vec<AsciiTable2D>,
}

impl ChabrierInterpolator {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let dir = dir.as_ref();
        let keys_file = dir.join("interp_keys.txt");
        let f = File::open(&keys_file)?;
        let reader = BufReader::new(f);

        let mut file_list = Vec::new();
        let mut age_keys = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.starts_with("files=") {
                let rest = &trimmed[6..];
                for part in rest.split(',') {
                    let clean = part.trim().trim_matches('"').trim_matches('\'');
                    if !clean.is_empty() {
                        file_list.push(clean.to_string());
                    }
                }
            } else if trimmed.starts_with("keys=") {
                let rest = &trimmed[5..];
                for part in rest.split_whitespace().flat_map(|p| p.split(',')) {
                    let clean = part.trim();
                    if !clean.is_empty() {
                        if let Ok(v) = clean.parse::<f64>() {
                            age_keys.push(v);
                        }
                    }
                }
            }
        }

        let mut tables = Vec::with_capacity(file_list.len());
        for fname in &file_list {
            let t = AsciiTable2D::read_from_file(dir.join(fname))?;
            tables.push(t);
        }

        Ok(Self { age_keys, tables })
    }

    /// Interpolate `(Teff_linear, Lum, Grav)` at `(log_age, mass)`.
    pub fn interpolate_dwarf(&self, log_age: f64, mass: f64) -> [f64; 3] {
        let mut c_pos = log_age;
        if c_pos >= *self.age_keys.last().unwrap() {
            c_pos = *self.age_keys.last().unwrap() * 0.99999;
        }
        if c_pos <= self.age_keys[0] {
            c_pos = self.age_keys[0] * 1.00001;
        }

        let iv = locate(&self.age_keys, c_pos).min(self.age_keys.len() - 2);
        let du = (c_pos - self.age_keys[iv]) / (self.age_keys[iv + 1] - self.age_keys[iv]);
        let probs = [1.0 - du, du];

        let mut res = [0.0; 3];
        for (k, &prob) in probs.iter().enumerate() {
            let tidx = iv + k;
            let tab = &self.tables[tidx];
            // col 0: Mass, col 1: Teff, col 2: Lum, col 3: Grav
            res[0] += prob * interpolate(&tab.cols[0], &tab.cols[1], mass);
            res[1] += prob * interpolate(&tab.cols[0], &tab.cols[2], mass);
            res[2] += prob * interpolate(&tab.cols[0], &tab.cols[3], mass);
        }
        res
    }
}

/// Bolometric Correction interpolator for DCMC (`Isochrones/BolCorr/DCMC/`).
#[derive(Debug, Clone)]
pub struct BolCorrInterpolator {
    pub feh_keys: Vec<f64>,
    pub teff_list: Vec<f64>,
    pub tables: Vec<AsciiTable2D>,
}

impl BolCorrInterpolator {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let dir = dir.as_ref();
        let keys_file = dir.join("interp_keys.txt");
        let f = File::open(&keys_file)?;
        let reader = BufReader::new(f);

        let mut file_list = Vec::new();
        let mut feh_keys = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.starts_with("files=") {
                let rest = &trimmed[6..];
                for part in rest.split(',') {
                    let clean = part.trim().trim_matches('"').trim_matches('\'');
                    if !clean.is_empty() {
                        file_list.push(clean.to_string());
                    }
                }
            } else if trimmed.starts_with("keys=") {
                let rest = &trimmed[5..];
                for part in rest.split_whitespace().flat_map(|p| p.split(',')) {
                    let clean = part.trim();
                    if !clean.is_empty() {
                        if let Ok(v) = clean.parse::<f64>() {
                            feh_keys.push(v);
                        }
                    }
                }
            }
        }

        // Read teff_list.dat
        let teff_file = dir.join("teff_list.dat");
        let tf = File::open(teff_file)?;
        let t_reader = BufReader::new(tf);
        let mut teff_list = Vec::new();
        let mut head_count = 0;
        for line in t_reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed == "<head>" {
                head_count += 1;
                continue;
            }
            if head_count >= 2 {
                for part in trimmed.split_whitespace() {
                    if let Ok(v) = part.parse::<f32>() {
                        teff_list.push(v as f64);
                    }
                }
            }
        }

        let mut tables = Vec::with_capacity(file_list.len());
        for fname in &file_list {
            let t = AsciiTable2D::read_from_file(dir.join(fname))?;
            tables.push(t);
        }

        Ok(Self {
            feh_keys,
            teff_list,
            tables,
        })
    }

    /// Evaluate bolometric corrections for [I, J, H, Ks] at `(feh, Teff_linear, grav)`.
    pub fn interpolate_bc(&self, feh: f64, teff_linear: f64, grav: f64) -> [f64; 4] {
        let mut c_feh = feh;
        if c_feh >= *self.feh_keys.last().unwrap() {
            c_feh = *self.feh_keys.last().unwrap() * 0.99999;
        }
        if c_feh <= self.feh_keys[0] {
            c_feh = self.feh_keys[0] * 1.00001;
        }

        let iv = locate(&self.feh_keys, c_feh).min(self.feh_keys.len() - 2);
        let du = (c_feh - self.feh_keys[iv]) / (self.feh_keys[iv + 1] - self.feh_keys[iv]);
        let probs = [1.0 - du, du];

        let jj = locate(&self.teff_list, teff_linear).min(self.teff_list.len() - 2);
        let t0 = self.teff_list[jj];
        let t1 = self.teff_list[jj + 1];
        let temp1 = 1.0 - (teff_linear.log10() - t0.log10()) / (t1.log10() - t0.log10());

        let mut val = [0.0; 4];
        for (k, &p_feh) in probs.iter().enumerate() {
            let tab = &self.tables[iv + k];
            // col 0: Teff + Grav
            // col 1..4: BC for I, J, H, Ks
            for b in 0..4 {
                let v0 = interpolate(&tab.cols[0], &tab.cols[b + 1], t0 + grav);
                let v1 = interpolate(&tab.cols[0], &tab.cols[b + 1], t1 + grav);
                val[b] += p_feh * (v0 * temp1 + v1 * (1.0 - temp1));
            }
        }
        val
    }
}

/// Single Age-Metallicity isochrone curve.
/// Matches `Isochrone` in `Isochrone.h`.
#[derive(Debug, Clone, Default)]
pub struct IsochroneTrack {
    pub age: f64,
    pub feh: f64,
    pub alpha: f64,
    pub m: Vec<f64>,
    pub mact: Vec<f64>,
    pub mag0: Vec<f64>,
    pub mag1: Vec<f64>,
    pub mag2: Vec<f64>,
    pub teff: Vec<f64>,
    pub grav: Vec<f64>,
    pub lum: Vec<f64>,
    /// Extra photometric bands: [I, J, H, Ks]
    pub mags: Vec<Vec<f64>>,
    /// Monotonically decreasing envelope of `mag0` for rapid magnitude cuts.
    pub vmon: Vec<f64>,
    pub m_min: f64,
    pub m_max: f64,
    pub mtip: f64,
}

impl IsochroneTrack {
    /// Compute the non-increasing upper envelope `vmon` of primary magnitude `mag0`.
    pub fn set_mon(&mut self) {
        if self.mag0.is_empty() {
            return;
        }
        self.vmon = self.mag0.clone();
        for i in 1..self.vmon.len() {
            if self.vmon[i] > self.vmon[i - 1] {
                self.vmon[i] = self.vmon[i - 1];
            }
        }
        self.m_min = self.m[0];
        self.m_max = *self.m.last().unwrap();
    }

    /// Identify the tip of the Red Giant Branch (RGB tip mass `mtip`).
    pub fn set_tip(&mut self) {
        if self.m.is_empty() {
            return;
        }
        self.mtip = self.m[0];
        for i in 1..self.m.len() {
            if self.lum[i] < self.lum[i - 1] && self.teff[i] > self.teff[i - 1] && self.m[i] > 0.15
            {
                self.mtip = self.m[i];
                break;
            }
        }
    }

    /// Minimum mass required to satisfy limiting absolute magnitude `max_v`.
    #[inline]
    pub fn min_mass_for_max_v(&self, max_v: f64) -> f64 {
        if self.vmon.is_empty() {
            return 0.07;
        }
        if max_v >= self.vmon[0] {
            self.m[0]
        } else if max_v <= *self.vmon.last().unwrap() {
            *self.m.last().unwrap()
        } else {
            let p = self.vmon.partition_point(|v| *v >= max_v);
            let idx = p.saturating_sub(1).min(self.m.len() - 1);
            self.m[idx]
        }
    }
}

/// Output parameters from `interpolate_tgm` in append1 step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StellarParams {
    pub lum: f64,
    pub teff: f64,
    pub grav: f64,
    pub mact: f64,
    pub mtip: f64,
    pub mags: [f64; 4], // [I, J, H, Ks]
}

/// Active mass window result from `set_isochrones1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActiveMassWindow {
    pub box_mmin: f64,
    pub box_mmax: f64,
    pub n_tot: usize,
}

/// Per-field isochrone cache state maintaining C++ IsochroneDB maxV and per-track thresholds.
///
/// Matches C++ `IsochroneDB.h:117-118,148` (`maxV=30`, `delta_maxV=0.01`) and `Isochrone.h:46` (`m_min`).
/// Must be instantiated freshly per field and never shared across fields or threads.
#[derive(Debug, Clone)]
pub struct IsochroneFieldState {
    pub max_v: f64,
    pub delta_max_v: f64,
    pub track_max_v: Vec<f64>,
    pub track_m_min: Vec<f64>,
}

impl IsochroneFieldState {
    pub fn new(db: &IsochroneDB) -> Self {
        let n = db.tracks.len();
        let mut track_max_v = Vec::with_capacity(n);
        let mut track_m_min = Vec::with_capacity(n);
        for t in &db.tracks {
            // Initialized to NaN so first access triggers setmaxV(maxV) matching C++
            track_max_v.push(f64::NAN);
            track_m_min.push(if t.m.is_empty() { 0.07 } else { t.m[0] });
        }
        Self {
            max_v: 30.0,
            delta_max_v: 0.01,
            track_max_v,
            track_m_min,
        }
    }
}

/// Padova Isochrone Database for DCMC photometric system.
#[derive(Debug, Clone)]
pub struct IsochroneDB {
    pub age_grid: Vec<f64>,
    pub feh_grid: Vec<f64>,
    pub tracks: Vec<IsochroneTrack>,
    pub d_age: f64,
}

impl IsochroneDB {
    /// Load the full Padova DCMC isochrone grid (182 ages x 34 metallicities).
    pub fn load(data_dir: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let data_dir = data_dir.as_ref();
        let padova_dir = data_dir.join("Isochrones/padova/DCMC");

        // 34 metallicities matching C++ IsochroneBase.cpp:197
        let fe_z: [f64; 34] = [
            0.0001, 0.0002, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0012, 0.0014,
            0.0016, 0.0018, 0.002, 0.0022, 0.0024, 0.0026, 0.003, 0.0034, 0.004, 0.005, 0.006,
            0.007, 0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.024, 0.028, 0.03,
        ];
        let feh_grid: Vec<f64> = fe_z.iter().map(|&z| (z / 0.019f64).log10()).collect();

        // Load dwarf patches if available
        let chabrier_dir = data_dir.join("Isochrones/Chabrier");
        let chabrier = if chabrier_dir.exists() {
            ChabrierInterpolator::load(&chabrier_dir).ok()
        } else {
            None
        };

        let bolcorr_dir = data_dir.join("Isochrones/BolCorr/DCMC");
        let bolcorr = if bolcorr_dir.exists() {
            BolCorrInterpolator::load(&bolcorr_dir).ok()
        } else {
            None
        };

        let dwarf_masses = [
            0.010, 0.012, 0.015, 0.020, 0.030, 0.040, 0.050, 0.055, 0.060, 0.070, 0.072, 0.075,
            0.080, 0.090, 0.100,
        ];

        let mut tracks = Vec::new();
        let mut age_grid = Vec::new();

        for (q, &z) in fe_z.iter().enumerate() {
            let fname = format!("output_{:.6}.dat", z);
            let file_path = padova_dir.join(&fname);
            let f = File::open(&file_path)?;
            let reader = BufReader::new(f);

            let mut current_track: Option<IsochroneTrack> = None;

            for line in reader.lines() {
                let line = line?;
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                let tokens: Vec<&str> = trimmed.split_whitespace().collect();
                if tokens.len() < 11 {
                    continue;
                }
                if tokens[2] == "nan" {
                    continue;
                }

                // Parse as f32 then promote to f64 matching C++ IsochroneBase.cpp:324,350
                let log_age: f64 = tokens[0].parse::<f32>().map(|v| v as f64).unwrap_or(-1.0);
                let m_ini: f64 = tokens[1].parse::<f32>().map(|v| v as f64).unwrap_or(0.0);
                let m_act: f64 = tokens[2].parse::<f32>().map(|v| v as f64).unwrap_or(0.0);
                let lum: f64 = tokens[3].parse::<f32>().map(|v| v as f64).unwrap_or(0.0);
                let teff: f64 = tokens[4].parse::<f32>().map(|v| v as f64).unwrap_or(0.0);
                let grav: f64 = tokens[5].parse::<f32>().map(|v| v as f64).unwrap_or(0.0);
                let mag_i: f64 = tokens[7].parse::<f32>().map(|v| v as f64).unwrap_or(1e6);
                let mag_j: f64 = tokens[8].parse::<f32>().map(|v| v as f64).unwrap_or(1e6);
                let mag_h: f64 = tokens[9].parse::<f32>().map(|v| v as f64).unwrap_or(1e6);
                let mag_ks: f64 = tokens[10].parse::<f32>().map(|v| v as f64).unwrap_or(1e6);

                let is_new_age = match &current_track {
                    Some(t) => (t.age - log_age).abs() > 1e-5,
                    None => true,
                };

                if is_new_age {
                    if let Some(mut completed_track) = current_track.take() {
                        completed_track.set_mon();
                        completed_track.set_tip();
                        tracks.push(completed_track);
                    }

                    let mut new_track = IsochroneTrack {
                        age: log_age,
                        feh: feh_grid[q],
                        alpha: 0.0,
                        m: Vec::new(),
                        mact: Vec::new(),
                        mag0: Vec::new(),
                        mag1: Vec::new(),
                        mag2: Vec::new(),
                        teff: Vec::new(),
                        grav: Vec::new(),
                        lum: Vec::new(),
                        mags: vec![Vec::new(); 4],
                        vmon: Vec::new(),
                        m_min: 0.0,
                        m_max: 0.0,
                        mtip: 0.0,
                    };

                    // Prepend Chabrier dwarfs if interpolators available
                    if let (Some(ch), Some(bc)) = (&chabrier, &bolcorr) {
                        for &dm in &dwarf_masses {
                            let [t_lin, l_val, g_val] = ch.interpolate_dwarf(log_age, dm);
                            let bc_mags = bc.interpolate_bc(feh_grid[q], t_lin, g_val);

                            new_track.m.push(dm);
                            new_track.mact.push(dm);
                            new_track.teff.push(t_lin.log10());
                            new_track.lum.push(l_val);
                            new_track.grav.push(g_val);

                            // Primary: J (col 8, index 1 in BC)
                            // Secondary: J (col 8, index 1 in BC)
                            // Color: Ks (col 10, index 3 in BC)
                            let j_mag = 4.77 - 2.5 * l_val - bc_mags[1];
                            let ks_mag = 4.77 - 2.5 * l_val - bc_mags[3];
                            new_track.mag0.push(j_mag);
                            new_track.mag1.push(j_mag);
                            new_track.mag2.push(ks_mag);

                            for b in 0..4 {
                                new_track.mags[b].push(4.77 - 2.5 * l_val - bc_mags[b]);
                            }
                        }
                    }

                    if q == 0 {
                        age_grid.push(log_age);
                    }
                    current_track = Some(new_track);
                }

                if let Some(track) = &mut current_track {
                    track.m.push(m_ini);
                    track.mact.push(m_act);
                    track.lum.push(lum);
                    track.teff.push(teff);
                    track.grav.push(grav);
                    track.mag0.push(mag_j);
                    track.mag1.push(mag_j);
                    track.mag2.push(mag_ks);
                    track.mags[0].push(mag_i);
                    track.mags[1].push(mag_j);
                    track.mags[2].push(mag_h);
                    track.mags[3].push(mag_ks);
                }
            }

            if let Some(mut completed_track) = current_track.take() {
                completed_track.set_mon();
                completed_track.set_tip();
                tracks.push(completed_track);
            }
        }

        // Reconstruct Age grid matching C++ IsochroneBase.cpp:307-310:
        // temp = icv[1].age - icv[0].age;
        // for (int i = 0; i < n_ages; i++) Age.push_back(icv[0].age + i * temp);
        let (reconstructed_age_grid, d_age) = if age_grid.len() >= 2 {
            let temp = age_grid[1] - age_grid[0];
            let base_age = age_grid[0];
            let grid: Vec<f64> = (0..age_grid.len())
                .map(|i| base_age + (i as f64) * temp)
                .collect();
            (grid, temp)
        } else {
            (age_grid, 0.02)
        };

        Ok(Self {
            age_grid: reconstructed_age_grid,
            feh_grid,
            tracks,
            d_age,
        })
    }

    /// Compute index in flattened track array: `q * n_ages + p`.
    #[inline]
    pub fn track_index(&self, p_age: usize, q_feh: usize) -> usize {
        q_feh * self.age_grid.len() + p_age
    }

    /// Calculate age selection probability weights $W_{\mathrm{Age}}$.
    pub fn calculate_age_prob(&self, age_linear: f64, dage_linear: f64) -> Vec<f64> {
        let n = self.age_grid.len();
        let mut w = vec![0.0; n];

        if dage_linear > 0.0 {
            let age_lo = (age_linear - dage_linear).max(1.0).log10();
            let age_hi = (age_linear + dage_linear).log10();
            let mut sum = 0.0;
            for i in 0..n {
                let temp_lo = if i == 0 {
                    age_lo.min(self.age_grid[i])
                } else {
                    (self.age_grid[i] + self.age_grid[i - 1]) * 0.5
                };
                if temp_lo > age_hi {
                    continue;
                }

                let temp_up = if i == n - 1 {
                    age_hi.max(self.age_grid[i])
                } else {
                    (self.age_grid[i] + self.age_grid[i + 1]) * 0.5
                };
                if temp_up < age_lo {
                    continue;
                }

                let t_lo = temp_lo.max(age_lo);
                let t_up = temp_up.min(age_hi);
                let val = (10.0f64.powf(t_up) - 10.0f64.powf(t_lo)) / (2.0 * dage_linear);
                w[i] = val;
                sum += val;
            }
            if sum > 0.0 {
                for v in &mut w {
                    *v /= sum;
                }
            }
        } else {
            let age_log = age_linear.max(1.0).log10();
            let i = ((age_log - self.age_grid[0]) / self.d_age).floor() as isize;
            if i < 0 {
                w[0] = 1.0;
            } else if (i as usize) >= n - 1 {
                w[n - 1] = 1.0;
            } else {
                let u = i as usize;
                let frac = (age_log - self.age_grid[u]) / (self.age_grid[u + 1] - self.age_grid[u]);
                w[u] = 1.0 - frac;
                w[u + 1] = frac;
            }
        }
        w
    }

    /// Calculate metallicity selection probability weights $W_{\mathrm{FeH}}$.
    pub fn calculate_feh_prob(&self, feh: f64, dfeh: f64) -> Vec<f64> {
        let n = self.feh_grid.len();
        let mut w = vec![0.0; n];

        if dfeh > 0.0 {
            let feh_lo = feh - 3.0 * dfeh;
            let feh_hi = feh + 3.0 * dfeh;
            let mut sum = 0.0;
            for q in 0..n {
                let temp_lo = if q == 0 {
                    feh_lo.min(self.feh_grid[q])
                } else {
                    (self.feh_grid[q] + self.feh_grid[q - 1]) * 0.5
                };
                if temp_lo > feh_hi {
                    continue;
                }

                let temp_up = if q == n - 1 {
                    feh_hi.max(self.feh_grid[q])
                } else {
                    (self.feh_grid[q] + self.feh_grid[q + 1]) * 0.5
                };
                if temp_up < feh_lo {
                    continue;
                }

                let t_lo = temp_lo.max(feh_lo);
                let t_up = temp_up.min(feh_hi);
                let cdf_up = ErfNR::normal_cdf((t_up - feh) / dfeh);
                let cdf_lo = ErfNR::normal_cdf((t_lo - feh) / dfeh);
                let val = cdf_up - cdf_lo;
                w[q] = val;
                sum += val;
            }
            if sum > 0.0 {
                for v in &mut w {
                    *v /= sum;
                }
            }
        } else {
            let q = locate(&self.feh_grid, feh).min(n - 2);
            let frac = (feh - self.feh_grid[q]) / (self.feh_grid[q + 1] - self.feh_grid[q]);
            w[q] = 1.0 - frac;
            w[q + 1] = frac;
        }
        w
    }

    /// Calculate active mass window $[m_{\mathrm{box,min}}, m_{\mathrm{box,max}}]$ and stochastic star count $N_{\mathrm{tot}}$.
    /// Matches `IsochroneDB::setIsochrones1` in `IsochroneDB.cpp:233-300`.
    pub fn set_isochrones1(
        &self,
        state: &mut IsochroneFieldState,
        age_linear: f64,
        dage_linear: f64,
        feh: f64,
        dfeh: f64,
        max_v1: f64,
        mass: f64,
        star_type: i32,
        imf: &mut Sampler,
        rng: &mut Philox,
    ) -> ActiveMassWindow {
        let w_age = self.calculate_age_prob(age_linear, dage_linear);
        let w_feh = self.calculate_feh_prob(feh, dfeh);

        let mut box_mmin = 1000.0f64;
        let mut box_mmax = 0.0f64;

        // Update maxV only if difference exceeds delta_maxV (0.01) (IsochroneDB.cpp:252-253)
        if (state.max_v - max_v1).abs() > state.delta_max_v {
            state.max_v = max_v1;
        }

        for (p, &wa) in w_age.iter().enumerate() {
            if wa > 0.0 {
                for (q, &wf) in w_feh.iter().enumerate() {
                    if wf > 0.0 {
                        let idx = self.track_index(p, q);
                        let track = &self.tracks[idx];
                        // Update track maxV and m_min if changed (IsochroneDB.cpp:267-274)
                        if state.track_max_v[idx] != state.max_v {
                            state.track_max_v[idx] = state.max_v;
                            state.track_m_min[idx] = track.min_mass_for_max_v(state.max_v);
                        }
                        let m_min = state.track_m_min[idx];
                        if m_min < box_mmin {
                            box_mmin = m_min;
                        }
                        let track_mmax = *track.m.last().unwrap_or(&100.0);
                        if track_mmax > box_mmax {
                            box_mmax = track_mmax;
                        }
                    }
                }
            }
        }

        box_mmin = box_mmin.max(imf.x_min);
        if star_type == 1 || star_type == 2 {
            box_mmin = 0.75;
            box_mmax = 1.0;
        }

        let fac = imf.get_fac(box_mmin, box_mmax);
        let mu_stars = mass * fac / imf.meanx;
        let base_n = mu_stars.floor() as usize;
        let remainder = mu_stars - (base_n as f64);
        let n_tot = if rng.uniform_f64() < remainder {
            base_n + 1
        } else {
            base_n
        };

        imf.set_range(box_mmin, box_mmax);

        ActiveMassWindow {
            box_mmin,
            box_mmax,
            n_tot,
        }
    }

    /// Interpolate physical properties and primary absolute magnitudes for a generated star particle.
    /// Matches `IsochroneDB::interpolateStar` in `IsochroneDB.cpp:302-341` and `Isochrone.cpp:75-87`.
    pub fn interpolate_star(&self, star: &mut StarParticle, state: &IsochroneFieldState) {
        // Reproduce C++ float age and feh quantization (IsochroneDB.cpp:306-316)
        let temp_age = star.age as f32;
        // Compare float promoted to double against double grid)
        let i = if (temp_age as f64) <= self.age_grid[0] {
            0
        } else if (temp_age as f64) >= *self.age_grid.last().unwrap() {
            self.age_grid.len() - 1
        } else {
            let idx = ((temp_age as f64 - self.age_grid[0]) / self.d_age + 0.5).floor() as usize;
            idx.min(self.age_grid.len() - 1)
        };

        let temp_feh = star.feh as f32;
        let j = locate_nearest(&self.feh_grid, temp_feh as f64);
        let track_idx = self.track_index(i, j);
        let track = &self.tracks[track_idx];

        // Reproduce C++ float mass casting and track leaf minimum rejection (Isochrone.cpp:77-87)
        let smass_t = star.smass as f32;
        let track_m_min = state.track_m_min[track_idx];
        let track_m_max = *track.m.last().unwrap_or(&100.0);

        if (smass_t as f64) <= track_m_min || (smass_t as f64) >= track_m_max {
            star.abs_mags = [1e6; 3];
            star.lum = 1e6;
            star.teff = 1e6;
            star.grav = 1e6;
        } else {
            let smass_d = smass_t as f64;
            let k = locate(&track.m, smass_d);
            let frac = (smass_t as f64 - track.m[k]) / (track.m[k + 1] - track.m[k]);
            star.abs_mags[0] = track.mag0[k] + (track.mag0[k + 1] - track.mag0[k]) * frac;
            star.abs_mags[1] = track.mag1[k] + (track.mag1[k + 1] - track.mag1[k]) * frac;
            star.abs_mags[2] = track.mag2[k] + (track.mag2[k + 1] - track.mag2[k]) * frac;
            star.lum = track.lum[k] + (track.lum[k + 1] - track.lum[k]) * frac;
            star.teff = track.teff[k] + (track.teff[k + 1] - track.teff[k]) * frac;
            star.grav = track.grav[k] + (track.grav[k + 1] - track.grav[k]) * frac;
        }
    }

    /// Interpolate physical parameters and all filter magnitudes in post-processing (`append1`).
    /// Matches `IsochroneDB::interpolateTGM` in `IsochroneDB.cpp:344-386` and `Isochrone.cpp:103-155`.
    pub fn interpolate_tgm(&self, log_age: f64, feh: f64, smass: f64) -> StellarParams {
        // C++ takes float age1, float feh1, float smass1 (IsochroneDB.cpp:344)
        let age1 = log_age as f32;
        let feh1 = feh as f32;
        let smass1 = smass as f32;

        // Compare float promoted to double against double grid)
        let i = if (age1 as f64) <= self.age_grid[0] {
            0
        } else if (age1 as f64) >= *self.age_grid.last().unwrap() {
            self.age_grid.len() - 1
        } else {
            let idx = ((age1 as f64 - self.age_grid[0]) / self.d_age + 0.5).floor() as usize;
            idx.min(self.age_grid.len() - 1)
        };

        let j = locate_nearest(&self.feh_grid, feh1 as f64);
        let track_idx = self.track_index(i, j);
        let track = &self.tracks[track_idx];

        let mtip = track.mtip;
        let last = track.m.len() - 1;

        if (smass1 as f64) >= track.m[last] {
            StellarParams {
                lum: track.lum[last],
                teff: track.teff[last],
                grav: track.grav[last],
                mact: track.mact[last],
                mtip,
                mags: [
                    track.mags[0][last],
                    track.mags[1][last],
                    track.mags[2][last],
                    track.mags[3][last],
                ],
            }
        } else if (smass1 as f64) <= track.m[0] {
            StellarParams {
                lum: track.lum[0],
                teff: track.teff[0],
                grav: track.grav[0],
                mact: track.mact[0],
                mtip,
                mags: [
                    track.mags[0][0],
                    track.mags[1][0],
                    track.mags[2][0],
                    track.mags[3][0],
                ],
            }
        } else {
            let k = locate(&track.m, smass1 as f64);
            let frac = (smass1 as f64 - track.m[k]) / (track.m[k + 1] - track.m[k]);
            StellarParams {
                lum: track.lum[k] + (track.lum[k + 1] - track.lum[k]) * frac,
                teff: track.teff[k] + (track.teff[k + 1] - track.teff[k]) * frac,
                grav: track.grav[k] + (track.grav[k + 1] - track.grav[k]) * frac,
                mact: track.mact[k] + (track.mact[k + 1] - track.mact[k]) * frac,
                mtip,
                mags: [
                    track.mags[0][k] + (track.mags[0][k + 1] - track.mags[0][k]) * frac,
                    track.mags[1][k] + (track.mags[1][k + 1] - track.mags[1][k]) * frac,
                    track.mags[2][k] + (track.mags[2][k + 1] - track.mags[2][k]) * frac,
                    track.mags[3][k] + (track.mags[3][k + 1] - track.mags[3][k]) * frac,
                ],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf_normal_cdf() {
        assert!((ErfNR::normal_cdf(0.0) - 0.5).abs() < 1e-12);
        // At 1 sigma: ~0.8413447
        assert!((ErfNR::normal_cdf(1.0) - 0.841344746).abs() < 1e-6);
        // At -1 sigma: ~0.1586552
        assert!((ErfNR::normal_cdf(-1.0) - 0.158655253).abs() < 1e-6);
    }

    #[test]
    fn test_read_alavtab_if_present() {
        let data_dir = crate::require_data_dir!();
        let p = data_dir.join("Isochrones/BolCorr/DCMC/alavtab_p00.dat");
        let p = p.as_path();
        assert!(p.exists(), "alavtab_p00.dat not found at {:?}", p);
        let rows = read_alavtab(p).expect("Failed to read alavtab_p00.dat");
        assert_eq!(rows.len(), 6800);
        assert_eq!(rows[0].teff, 500.0);
        assert_eq!(rows[0].logg, 3.50);
        assert_eq!(rows[0].av, 1.0);
    }

    #[test]
    fn test_chabrier_and_bolcorr_interpolator() {
        let data_dir = crate::require_data_dir!();
        let ch_dir = data_dir.join("Isochrones/Chabrier");
        let bc_dir = data_dir.join("Isochrones/BolCorr/DCMC");
        assert!(ch_dir.exists(), "Chabrier dir not found: {:?}", ch_dir);
        assert!(bc_dir.exists(), "BolCorr dir not found: {:?}", bc_dir);

        let ch = ChabrierInterpolator::load(&ch_dir).expect("Failed to load Chabrier tables");
        assert_eq!(ch.age_keys.len(), 5);
        assert_eq!(ch.tables.len(), 5);

        let [t_lin, lum, grav] = ch.interpolate_dwarf(9.0, 0.05);
        assert!(t_lin > 1000.0 && t_lin < 5000.0);
        assert!(lum < 0.0);
        assert!(grav > 3.0);

        let bc = BolCorrInterpolator::load(&bc_dir).expect("Failed to load BolCorr tables");
        assert_eq!(bc.feh_keys.len(), 7);
        assert_eq!(bc.teff_list.len(), 148);
        assert_eq!(bc.tables.len(), 7);

        let bc_mags = bc.interpolate_bc(0.0, t_lin, grav);
        assert_eq!(bc_mags.len(), 4);
    }

    #[test]
    fn test_isochrone_db_load_and_interpolate() {
        let data_dir = crate::require_data_dir!();

        let ic = IsochroneDB::load(&data_dir).expect("Failed to load Padova DCMC isochrone DB");
        assert_eq!(ic.age_grid.len(), 182);
        assert_eq!(ic.feh_grid.len(), 34);
        assert_eq!(ic.tracks.len(), 182 * 34);

        // Test set_isochrones1
        let mut state = IsochroneFieldState::new(&ic);
        let mut imf = Sampler::new_thin_disk();
        let mut rng = Philox::from_stream(42, 0, 0, 0, 0, 1);
        let win = ic.set_isochrones1(
            &mut state, 0.075e9, 0.075e9, 0.01, 0.12, 10.0, 1000.0, 0, &mut imf, &mut rng,
        );
        assert!(win.box_mmin >= 0.07);
        assert!(win.box_mmax <= 100.0);
        assert!(win.box_mmin < win.box_mmax);

        // Test interpolate_star
        let mut star = StarParticle::default();
        star.age = (1.5e9f64).log10();
        star.feh = 0.0;
        star.smass = 1.0;
        ic.interpolate_star(&mut star, &state);
        assert!(star.abs_mags[0] < 100.0, "Absolute J mag must be finite");
        assert!(
            star.teff > 3.0 && star.teff < 5.0,
            "log Teff must be physical"
        );

        // Test interpolate_tgm
        let params = ic.interpolate_tgm((1.5e9f64).log10(), 0.0, 1.0);
        assert!(params.mact > 0.5 && params.mact <= 1.0);
        assert!(params.teff > 3.0);
    }
}
