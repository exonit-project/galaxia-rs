// SPDX-License-Identifier: AGPL-3.0-or-later
// Original C++ code Copyright (c) 2012 Sanjib Sharma and Joss Bland-Hawthorn
// Ported from Galaxia.cpp and StellarPopulation.cpp
//! Full end-to-end simulation pipeline for Galaxia.
//!
//! Orchestrates tree leaf filtering, isochrone window determination, stellar parameter
//! generation, coordinate transformations, magnitude evaluation, extinction interpolation,
//! and canonical EBF catalog serialization matching C++ Galaxia 0.7.2 `runModel`.

use crate::backend::{create_backend, AcceleratorBackend, ActiveLeaf, BackendKind};
use crate::bhtree::BHTree;
use crate::ebf::EbfWriter;
use crate::extinction::{ExtinctionMap, ExtinctionValues};
use crate::imf::Sampler;
use crate::interp::VCirc;
use crate::isochrone::{IsochroneDB, IsochroneFieldState, StellarParams};
use crate::params::Params;
use crate::population::{create_population, StarParticle, RSUN};
use crate::rng::Philox;
use crate::survey::SurveyDesign;
use std::path::Path;
use std::time::{Duration, Instant};

/// Star particle combined with physical observables and line-of-sight extinction.
#[derive(Debug, Clone)]
pub struct SpawnedStar {
    pub star: StarParticle,
    pub tgm: StellarParams,
    pub ext: ExtinctionValues,
}

/// Generation and timing statistics for a simulation run.
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Total stars spawned across all populations.
    pub total_stars: usize,
    /// Stars spawned per population (0..9).
    pub pop_stars: [usize; 10],
    /// Active leaves passing geometric and distance survey cuts.
    pub active_leafs: [usize; 10],
    /// Total leaves in octree per population.
    pub total_leafs: [usize; 10],
    /// Duration of BHTree octree loading.
    pub tree_load_duration: Duration,
    /// Duration of star spawning and survey filtering.
    pub spawn_duration: Duration,
    /// Duration of physical parameter lookups and extinction calculation.
    pub post_process_duration: Duration,
    /// Duration of EBF file serialization.
    pub write_duration: Duration,
    /// Total pipeline wall-clock execution time.
    pub total_duration: Duration,
    /// Output file path written.
    pub output_path: String,
}

/// TESS magnitude (Tmag) from 2MASS-like J, Ks (Stassun et al. 2018, TESS Input Catalog v8 relation).
#[inline]
pub fn stassun_tmag(j: f64, ks: f64) -> f64 {
    let c = j - ks;
    if c < -0.1 {
        j + 0.5
    } else if c <= 0.70 {
        j + 1.22163 * c.powi(3) - 1.74299 * c.powi(2) + 1.89115 * c + 0.0563
    } else if c <= 1.0 {
        j - 269.372 * c.powi(3) + 668.453 * c.powi(2) - 545.64 * c + 147.811
    } else {
        j + 1.75
    }
}

/// Extincted Tmag from heliocentric distance (kpc), DCMC J, DCMC Ks, and solar E(B-V).
/// Ceiling filter:
/// `RV = 3.1, A_J/A_V = 0.282, A_Ks/A_V = 0.112`.
#[inline]
pub fn extincted_tmag(rad_kpc: f64, dcmc_j: f64, dcmc_ks: f64, ebv_solar: f64) -> f64 {
    let dm = 5.0 * (rad_kpc * 100.0).log10();
    let j_ext = dcmc_j + dm + 0.282 * 3.1 * ebv_solar;
    let ks_ext = dcmc_ks + dm + 0.112 * 3.1 * ebv_solar;
    stassun_tmag(j_ext, ks_ext)
}

/// Galaxia simulation generator holding resident databases in memory.
pub struct Generator {
    pub isochrones: IsochroneDB,
    pub extinction: ExtinctionMap,
    pub vcirc: VCirc,
    pub backend: Box<dyn AcceleratorBackend>,
    pub resident_trees: Option<Vec<BHTree>>,
}

impl Generator {
    /// Load resident databases from `data_dir` and initialize the requested accelerator backend.
    pub fn new(data_dir: impl AsRef<Path>, backend_kind: BackendKind) -> anyhow::Result<Self> {
        let backend = create_backend(backend_kind)?;
        Self::with_backend(data_dir, backend)
    }

    /// Load resident databases with an explicitly specified accelerator backend.
    pub fn with_backend(
        data_dir: impl AsRef<Path>,
        backend: Box<dyn AcceleratorBackend>,
    ) -> anyhow::Result<Self> {
        let dir = data_dir.as_ref();

        let isochrones = IsochroneDB::load(dir).map_err(|e| {
            anyhow::anyhow!("Failed to load IsochroneDB from {}: {}", dir.display(), e)
        })?;

        let extinction = ExtinctionMap::load_from_dir(dir).map_err(|e| {
            anyhow::anyhow!("Failed to load ExtinctionMap from {}: {}", dir.display(), e)
        })?;

        let vcirc_path = dir.join("Model/vcirc.dat");
        let vcirc = if vcirc_path.exists() {
            VCirc::load_from_file(&vcirc_path).map_err(|e| {
                anyhow::anyhow!("Failed to load vcirc from {}: {}", vcirc_path.display(), e)
            })?
        } else {
            VCirc::default()
        };

        Ok(Self {
            isochrones,
            extinction,
            vcirc,
            backend,
            resident_trees: None,
        })
    }

    /// Preload all 10 BHTree octrees into memory once (for batch mode).
    pub fn preload_resident_trees(&mut self, data_dir: impl AsRef<Path>) -> anyhow::Result<()> {
        let dir = data_dir.as_ref();
        let mut trees = Vec::with_capacity(10);
        for pop_id in 0..10 {
            let option_e = if pop_id <= 6 { 1 } else { 0 };
            let tree = BHTree::load_for_pop(dir, pop_id, option_e, 1, &crate::coords::SUN_POS)
                .map_err(|e| {
                    anyhow::anyhow!("Failed to load resident BHTree for pop {}: {}", pop_id, e)
                })?;
            trees.push(tree);
        }
        self.resident_trees = Some(trees);
        Ok(())
    }

    /// Create a Generator with all resident data preloaded (IsochroneDB, ExtinctionMap, VCirc, and 10 BHTrees).
    pub fn with_resident_data(
        data_dir: impl AsRef<Path>,
        backend: Box<dyn AcceleratorBackend>,
    ) -> anyhow::Result<Self> {
        let mut gen = Self::with_backend(data_dir.as_ref(), backend)?;
        gen.preload_resident_trees(data_dir.as_ref())?;
        Ok(gen)
    }

    /// Run the full simulation pipeline matching C++ `runModel`.
    pub fn run_model(&self, params: &Params) -> anyhow::Result<GenerationStats> {
        self.run_model_ext_quiet(params, None, false)
            .map(|(stats, _)| stats)
    }

    /// Run the simulation pipeline with an optional extincted Tmag ceiling filter.
    pub fn run_model_ext(
        &self,
        params: &Params,
        tmag_ceiling: Option<f64>,
    ) -> anyhow::Result<(GenerationStats, usize)> {
        self.run_model_ext_quiet(params, tmag_ceiling, false)
    }

    /// Run the simulation pipeline with an optional extincted Tmag ceiling filter and quiet flag.
    pub fn run_model_ext_quiet(
        &self,
        params: &Params,
        tmag_ceiling: Option<f64>,
        quiet: bool,
    ) -> anyhow::Result<(GenerationStats, usize)> {
        let total_start = Instant::now();
        let survey = SurveyDesign::from_params(params);

        let (i_min, i_max) = if params.pop_id == -1 {
            (0, 10)
        } else {
            let p = params.pop_id.clamp(0, 9) as usize;
            (p, p + 1)
        };

        let mut stats = GenerationStats {
            output_path: params.output_ebf_path(),
            ..Default::default()
        };

        let mut spawned_stars: Vec<SpawnedStar> = Vec::new();

        // Fresh isochrone cache state per field
        let mut isochrone_state = IsochroneFieldState::new(&self.isochrones);

        if !quiet {
            println!("Generating populations................");
            println!("--------------------------------------------------------");
        }

        for i in i_min..i_max {
            let option_e = if i <= 6 { 1 } else { 0 };
            let pop = create_population(i, params.warp_flare_on as usize, option_e);

            let local_tree;
            let bhtree = if let Some(trees) = &self.resident_trees {
                &trees[i]
            } else {
                let t_tree = Instant::now();
                local_tree = BHTree::load_for_pop(
                    &params.code_data_dir,
                    i,
                    option_e,
                    params.warp_flare_on as usize,
                    &[params.pos_c[0], params.pos_c[1], params.pos_c[2]],
                )
                .map_err(|e| anyhow::anyhow!("Failed to load BHTree for pop {}: {}", i, e))?;
                stats.tree_load_duration += t_tree.elapsed();
                &local_tree
            };
            stats.total_leafs[i] = bhtree.leafs.len();

            let mut imf = Sampler::for_pop_id(i);
            let mut pop_stars_count = 0;
            let mut active_leafs_count = 0;

            let t_spawn = Instant::now();

            let is_cpu = self.backend.name() == "CPU Backend";

            if is_cpu {
                // CPU fp64 path: direct leaf-by-leaf execution preserving C++ maxV cache sequence
                for (leaf_idx, leaf) in bhtree.leafs.iter().enumerate() {
                    let r_helio = [
                        leaf.pos[0] - params.pos_c[0],
                        leaf.pos[1] - params.pos_c[1],
                        leaf.pos[2] - params.pos_c[2],
                    ];
                    let dist = (r_helio[0] * r_helio[0]
                        + r_helio[1] * r_helio[1]
                        + r_helio[2] * r_helio[2])
                        .sqrt();
                    let h = leaf.l * 1.732;
                    let rc = dist - h;

                    if rc >= params.r_max || !survey.patch.check_sphere(&r_helio, h) {
                        continue;
                    }

                    active_leafs_count += 1;

                    let rc_xy = (leaf.pos[0] * leaf.pos[0] + leaf.pos[1] * leaf.pos[1]).sqrt();
                    let mut feh_temp = pop.feh();
                    if pop.id() < 7 {
                        feh_temp -= 0.07 * (rc_xy - RSUN);
                    }

                    let max_v = params.abs_mag_limits[1].min(params.app_mag_limits[1] - leaf.dmod);

                    let mut leaf_rng =
                        Philox::from_stream(params.seed as u64, 0, i as u32, leaf_idx as u32, 0, 0);

                    let active_window = self.isochrones.set_isochrones1(
                        &mut isochrone_state,
                        pop.age(),
                        pop.dage(),
                        feh_temp,
                        pop.dfeh(),
                        max_v,
                        leaf.mass * params.f_sample,
                        params.star_type,
                        &mut imf,
                        &mut leaf_rng,
                    );

                    let mut n_tot = active_window.n_tot;
                    if n_tot == 1 && leaf.mass < 1e-2 && (leaf.rho_max / leaf.rho_min > 250.0) {
                        n_tot = 0;
                    }

                    if n_tot == 0 {
                        continue;
                    }

                    let mut leaf_imf = imf.clone();
                    leaf_imf.cpd_min = imf.cpd_min;
                    leaf_imf.cpd_max = imf.cpd_max;

                    for star_idx in 0..n_tot {
                        let mut star_rng = Philox::from_stream(
                            params.seed as u64,
                            0,
                            i as u32,
                            leaf_idx as u32,
                            star_idx as u32,
                            1,
                        );

                        let mut star = StarParticle {
                            popid: i as i32,
                            satid: 0,
                            partid: 0,
                            ..Default::default()
                        };

                        let bh_node = crate::bhtree::BHNode {
                            pos: leaf.pos,
                            l: leaf.l,
                            lage: leaf.lage,
                            rho_max: leaf.rho_max,
                            rho_min: leaf.rho_min,
                            dmod: leaf.dmod,
                            mass: leaf.mass,
                            delta_mass: 0.0,
                            left: -1,
                        };

                        if option_e == 1 {
                            let pos_age = pop.generate_pos(&bh_node, &mut star_rng);
                            star.pos = [pos_age[0], pos_age[1], pos_age[2]];
                            star.age = pos_age[3];
                        }

                        pop.set_age_feh_mass(&mut star, &leaf_imf, &mut star_rng);
                        self.isochrones
                            .interpolate_star(&mut star, &isochrone_state);
                        star.partid = 0;

                        if star.abs_mags[0] < 1e4 {
                            if option_e == 0 {
                                let pos_age = pop.generate_pos(&bh_node, &mut star_rng);
                                star.pos = [pos_age[0], pos_age[1], pos_age[2]];
                            }

                            pop.set_vel(&mut star, &self.vcirc, &mut star_rng);

                            star.pos[0] -= params.pos_c[0];
                            star.pos[1] -= params.pos_c[1];
                            star.pos[2] -= params.pos_c[2];

                            star.vel[0] -= params.pos_c[3];
                            star.vel[1] -= params.pos_c[4];
                            star.vel[2] -= params.pos_c[5];

                            star.abs_to_app_mag();

                            if survey.check_star(
                                &star.pos,
                                star.mags[0],
                                star.mags[1],
                                star.mags[2],
                                star.dmod,
                            ) {
                                let tgm = self
                                    .isochrones
                                    .interpolate_tgm(star.age, star.feh, star.smass);
                                // Extinction uses float XYZ matching Galaxia.cpp:55-75
                                let pos_xyz = [
                                    star.pos[0] as f32 as f64,
                                    star.pos[1] as f32 as f64,
                                    star.pos[2] as f32 as f64,
                                ];
                                let ext = self.extinction.compute(&pos_xyz);
                                spawned_stars.push(SpawnedStar { star, tgm, ext });
                                pop_stars_count += 1;
                            }
                        }
                    }
                }
            } else {
                let mut active_leaves = Vec::new();
                let mut cumulative_star_offset = 0;

                for (leaf_idx, leaf) in bhtree.leafs.iter().enumerate() {
                    let r_helio = [
                        leaf.pos[0] - params.pos_c[0],
                        leaf.pos[1] - params.pos_c[1],
                        leaf.pos[2] - params.pos_c[2],
                    ];
                    let dist = (r_helio[0] * r_helio[0]
                        + r_helio[1] * r_helio[1]
                        + r_helio[2] * r_helio[2])
                        .sqrt();
                    let h = leaf.l * 1.732;
                    let rc = dist - h;

                    if rc >= params.r_max || !survey.patch.check_sphere(&r_helio, h) {
                        continue;
                    }

                    active_leafs_count += 1;

                    let rc_xy = (leaf.pos[0] * leaf.pos[0] + leaf.pos[1] * leaf.pos[1]).sqrt();
                    let mut feh_temp = pop.feh();
                    if pop.id() < 7 {
                        feh_temp -= 0.07 * (rc_xy - RSUN);
                    }

                    let max_v = params.abs_mag_limits[1].min(params.app_mag_limits[1] - leaf.dmod);

                    let mut leaf_rng =
                        Philox::from_stream(params.seed as u64, 0, i as u32, leaf_idx as u32, 0, 0);

                    let active_window = self.isochrones.set_isochrones1(
                        &mut isochrone_state,
                        pop.age(),
                        pop.dage(),
                        feh_temp,
                        pop.dfeh(),
                        max_v,
                        leaf.mass * params.f_sample,
                        params.star_type,
                        &mut imf,
                        &mut leaf_rng,
                    );

                    let mut n_tot = active_window.n_tot;
                    if n_tot == 1 && leaf.mass < 1e-2 && (leaf.rho_max / leaf.rho_min > 250.0) {
                        n_tot = 0;
                    }

                    if n_tot == 0 {
                        continue;
                    }

                    active_leaves.push(ActiveLeaf {
                        orig_leaf_idx: leaf_idx,
                        pos: leaf.pos,
                        l: leaf.l,
                        lage: leaf.lage,
                        rho_max: leaf.rho_max,
                        rho_min: leaf.rho_min,
                        dmod: leaf.dmod,
                        mass: leaf.mass,
                        box_mmin: active_window.box_mmin,
                        box_mmax: active_window.box_mmax,
                        cpd_min: imf.cpd_min,
                        cpd_max: imf.cpd_max,
                        n_tot,
                        star_offset: cumulative_star_offset,
                    });
                    cumulative_star_offset += n_tot;
                }

                let accepted_particles = self.backend.spawn_stars(
                    i,
                    pop.as_ref(),
                    params,
                    &active_leaves,
                    &imf,
                    &self.isochrones,
                    &self.vcirc,
                )?;

                for star in accepted_particles {
                    let tgm = self
                        .isochrones
                        .interpolate_tgm(star.age, star.feh, star.smass);
                    let pos_xyz = [
                        star.pos[0] as f32 as f64,
                        star.pos[1] as f32 as f64,
                        star.pos[2] as f32 as f64,
                    ];
                    let ext = self.extinction.compute(&pos_xyz);
                    spawned_stars.push(SpawnedStar { star, tgm, ext });
                    pop_stars_count += 1;
                }
            }

            stats.spawn_duration += t_spawn.elapsed();
            stats.pop_stars[i] = pop_stars_count;
            stats.active_leafs[i] = active_leafs_count;

            if !quiet {
                println!(
                    "Pop {:<2} ({:<10}): spawned {:>6} stars from {:>5} active leaves (total {:>7} leaves)",
                    i,
                    pop_name(i),
                    pop_stars_count,
                    active_leafs_count,
                    stats.total_leafs[i]
                );
            }
        }

        if !quiet {
            println!("--------------------------------------------------------");
        }
        let raw_stars_count = spawned_stars.len();
        if let Some(ceiling) = tmag_ceiling {
            spawned_stars.retain(|s| {
                let tmag = extincted_tmag(
                    s.star.rad,
                    s.tgm.mags[1],
                    s.tgm.mags[3],
                    s.ext.exbv_solar as f64,
                );
                tmag.is_finite() && tmag <= ceiling
            });
            if !quiet {
                println!(
                    "Total stars spawned = {} ({} retained after Tmag <= {:.1} ceiling)",
                    raw_stars_count,
                    spawned_stars.len(),
                    ceiling
                );
            }
        } else if !quiet {
            println!("Total stars spawned = {}", raw_stars_count);
        }
        stats.total_stars = spawned_stars.len();

        // Serialization step
        let t_write = Instant::now();
        write_output_ebf(params, &spawned_stars)?;
        stats.write_duration = t_write.elapsed();
        if !quiet {
            println!("File written: {}", stats.output_path);

            stats.total_duration = total_start.elapsed();
            println!(
                "Execution completed in {:.2}s (tree: {:.2}s, spawn: {:.2}s, write: {:.2}s)",
                stats.total_duration.as_secs_f64(),
                stats.tree_load_duration.as_secs_f64(),
                stats.spawn_duration.as_secs_f64(),
                stats.write_duration.as_secs_f64()
            );
        } else {
            stats.total_duration = total_start.elapsed();
        }

        Ok((stats, raw_stars_count))
    }
}

/// Standalone convenience function to run generation from `Params` with specified backend.
pub fn run_model(params: &Params, backend_kind: BackendKind) -> anyhow::Result<GenerationStats> {
    let generator = Generator::new(&params.code_data_dir, backend_kind)?;
    generator.run_model(params)
}

fn pop_name(id: usize) -> &'static str {
    match id {
        0 => "ThinDisk0",
        1 => "ThinDisk1",
        2 => "ThinDisk2",
        3 => "ThinDisk3",
        4 => "ThinDisk4",
        5 => "ThinDisk5",
        6 => "ThinDisk6",
        7 => "ThickDisk",
        8 => "Spheroid",
        9 => "Bulge",
        _ => "Unknown",
    }
}

fn write_output_ebf(params: &Params, stars: &[SpawnedStar]) -> anyhow::Result<()> {
    let out_path = params.output_ebf_path();
    if let Some(parent) = Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Atomic write via temp file
    let tmp_path = format!("{}.tmp.{}", out_path, std::process::id());

    {
        let mut writer = EbfWriter::create(&tmp_path).map_err(|e| {
            anyhow::anyhow!("Failed to create temporary output file {}: {}", tmp_path, e)
        })?;

        let n = stars.len();

        let mut px = Vec::with_capacity(n);
        let mut py = Vec::with_capacity(n);
        let mut pz = Vec::with_capacity(n);
        let mut vx = Vec::with_capacity(n);
        let mut vy = Vec::with_capacity(n);
        let mut vz = Vec::with_capacity(n);
        let mut feh = Vec::with_capacity(n);
        let mut alpha = Vec::with_capacity(n);
        let mut smass = Vec::with_capacity(n);
        let mut age = Vec::with_capacity(n);
        let mut rad = Vec::with_capacity(n);
        let mut mag0 = Vec::with_capacity(n);
        let mut mag1 = Vec::with_capacity(n);
        let mut mag2 = Vec::with_capacity(n);
        let mut popid = Vec::with_capacity(n);
        let mut satid = Vec::with_capacity(n);
        let mut fieldid = Vec::with_capacity(n);
        let mut partid = Vec::with_capacity(n);

        let mut lum = Vec::with_capacity(n);
        let mut teff = Vec::with_capacity(n);
        let mut grav = Vec::with_capacity(n);
        let mut mact = Vec::with_capacity(n);
        let mut mtip = Vec::with_capacity(n);
        let mut dcmc_i = Vec::with_capacity(n);
        let mut dcmc_j = Vec::with_capacity(n);
        let mut dcmc_h = Vec::with_capacity(n);
        let mut dcmc_ks = Vec::with_capacity(n);

        let mut exbv_schlegel = Vec::with_capacity(n);
        let mut exbv_solar = Vec::with_capacity(n);
        let mut exbv_schlegel_inf = Vec::with_capacity(n);
        let mut glon = Vec::with_capacity(n);
        let mut glat = Vec::with_capacity(n);

        for s in stars {
            px.push(s.star.pos[0] as f32);
            py.push(s.star.pos[1] as f32);
            pz.push(s.star.pos[2] as f32);
            vx.push(s.star.vel[0] as f32);
            vy.push(s.star.vel[1] as f32);
            vz.push(s.star.vel[2] as f32);
            feh.push(s.star.feh as f32);
            alpha.push(s.star.alpha as f32);
            smass.push(s.star.smass as f32);
            age.push(s.star.age as f32);
            rad.push(s.star.rad as f32);
            mag0.push(s.star.mags[0] as f32);
            mag1.push(s.star.mags[1] as f32);
            mag2.push(s.star.mags[2] as f32);
            popid.push(s.star.popid);
            satid.push(s.star.satid);
            fieldid.push(0i32);
            partid.push(s.star.partid);

            lum.push(s.tgm.lum as f32);
            teff.push(s.tgm.teff as f32);
            grav.push(s.tgm.grav as f32);
            mact.push(s.tgm.mact as f32);
            mtip.push(s.tgm.mtip as f32);
            dcmc_i.push(s.tgm.mags[0] as f32);
            dcmc_j.push(s.tgm.mags[1] as f32);
            dcmc_h.push(s.tgm.mags[2] as f32);
            dcmc_ks.push(s.tgm.mags[3] as f32);

            exbv_schlegel.push(s.ext.exbv_schlegel);
            exbv_solar.push(s.ext.exbv_solar);
            exbv_schlegel_inf.push(s.ext.exbv_schlegel_inf);
            glon.push(s.ext.glon);
            glat.push(s.ext.glat);
        }

        // 1. /log with full provenance metadata
        let ceiling_str = match params.tmag_ceiling {
            Some(c) => format!("{:.6}", c),
            None => "none".to_string(),
        };
        let param_file_content =
            if !params.parameter_file.is_empty() && Path::new(&params.parameter_file).exists() {
                std::fs::read_to_string(&params.parameter_file).unwrap_or_default()
            } else {
                String::new()
            };
        let log_text = format!(
            "# File generated by galaxia-rs-{} (git {})\n\
             # backend: cpu-fp64\n\
             # tmag_ceiling: {}\n\
             # seed: {}\n\
             # coordinates: l={:.6} b={:.6} center=[{:.6}, {:.6}, {:.6}]\n\
             # survey_cuts: app_mag_limits=[{:.1}, {:.1}] abs_mag_limits=[{:.1}, {:.1}] color_limits=[{:.1}, {:.1}] r_max={:.1}\n\
             # <parameterfile>\n{}# </parameterfile>\n",
            env!("CARGO_PKG_VERSION"),
            env!("GIT_COMMIT"),
            ceiling_str,
            params.seed,
            params.longitude, params.latitude,
            params.pos_c[0], params.pos_c[1], params.pos_c[2],
            params.app_mag_limits[0], params.app_mag_limits[1],
            params.abs_mag_limits[0], params.abs_mag_limits[1],
            params.color_limits[0], params.color_limits[1],
            params.r_max,
            param_file_content
        );
        writer.write_char("/log", "", &[log_text.len()], log_text.as_bytes())?;

        // 2. Primary Astrometry and Kinematics
        writer.write_f32("/px", "1000 parsec", &[n], &px)?;
        writer.write_f32("/py", "1000 parsec", &[n], &py)?;
        writer.write_f32("/pz", "1000 parsec", &[n], &pz)?;
        writer.write_f32("/vx", "1000 meter/second", &[n], &vx)?;
        writer.write_f32("/vy", "1000 meter/second", &[n], &vy)?;
        writer.write_f32("/vz", "1000 meter/second", &[n], &vz)?;
        writer.write_f32("/feh", "", &[n], &feh)?;
        writer.write_f32("/alpha", "", &[n], &alpha)?;
        writer.write_f32("/smass", "solar_mass", &[n], &smass)?;
        writer.write_f32("/age", "log year", &[n], &age)?;
        writer.write_f32("/rad", "1000 parsec", &[n], &rad)?;
        writer.write_f32("/mag0", "", &[n], &mag0)?;
        writer.write_f32("/mag1", "", &[n], &mag1)?;
        writer.write_f32("/mag2", "", &[n], &mag2)?;
        writer.write_i32("/popid", "", &[n], &popid)?;
        writer.write_i32("/satid", "", &[n], &satid)?;
        writer.write_i32("/fieldid", "", &[n], &fieldid)?;
        writer.write_i32("/partid", "", &[n], &partid)?;
        writer.write_f64("/center", "1000 parsec", &[6], &params.pos_c)?;

        // 3. Physical Parameters and Photometric Filters
        writer.write_f32("/lum", "log solar_luminosity", &[n], &lum)?;
        writer.write_f32("/teff", "log kelvin", &[n], &teff)?;
        writer.write_f32("/grav", "log 0.01 meter/second/second", &[n], &grav)?;
        writer.write_f32("/mact", "solar_mass", &[n], &mact)?;
        writer.write_f32("/mtip", "solar_mass", &[n], &mtip)?;
        writer.write_f32("/dcmc_i", "magnitude", &[n], &dcmc_i)?;
        writer.write_f32("/dcmc_j", "magnitude", &[n], &dcmc_j)?;
        writer.write_f32("/dcmc_h", "magnitude", &[n], &dcmc_h)?;
        writer.write_f32("/dcmc_ks", "magnitude", &[n], &dcmc_ks)?;

        // 4. Extinction Maps and Galactic Coordinates
        writer.write_f32("/exbv_schlegel", "", &[n], &exbv_schlegel)?;
        writer.write_f32("/exbv_solar", "", &[n], &exbv_solar)?;
        writer.write_f32("/exbv_schlegel_inf", "", &[n], &exbv_schlegel_inf)?;
        writer.write_f32("/glon", "degree", &[n], &glon)?;
        writer.write_f32("/glat", "degree", &[n], &glat)?;
    }

    // Atomically rename temporary file to destination
    if let Err(e) = std::fs::rename(&tmp_path, &out_path) {
        let _ = std::fs::remove_file(&tmp_path);
        anyhow::bail!(
            "Failed to atomically rename {} to {}: {}",
            tmp_path,
            out_path,
            e
        );
    }

    Ok(())
}
