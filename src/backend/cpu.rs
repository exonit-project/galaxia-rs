// SPDX-License-Identifier: AGPL-3.0-or-later
// Original C++ code Copyright (c) 2012 Sanjib Sharma and Joss Bland-Hawthorn
// Ported from StellarPopulation.cpp
//! CPU accelerator backend for star generation.

use crate::backend::{AcceleratorBackend, ActiveLeaf};
use crate::bhtree::BHNode;
use crate::imf::Sampler;
use crate::interp::VCirc;
use crate::isochrone::{IsochroneDB, IsochroneFieldState};
use crate::params::Params;
use crate::population::{Population, StarParticle};
use crate::rng::Philox;
use crate::survey::SurveyDesign;

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AcceleratorBackend for CpuBackend {
    fn name(&self) -> &str {
        "CPU Backend"
    }

    fn platform_tag(&self) -> String {
        format!(
            "galaxia-rs-{}-cpu-fp64 (git {})",
            env!("CARGO_PKG_VERSION"),
            env!("GIT_COMMIT")
        )
    }

    fn spawn_stars(
        &self,
        pop_id: usize,
        pop: &dyn Population,
        params: &Params,
        active_leaves: &[ActiveLeaf],
        imf: &Sampler,
        isochrones: &IsochroneDB,
        vcirc: &VCirc,
    ) -> anyhow::Result<Vec<StarParticle>> {
        let survey = SurveyDesign::from_params(params);
        let mut spawned = Vec::new();
        let option_e = pop.option_e();
        let mut isochrone_state = IsochroneFieldState::new(isochrones);

        for leaf in active_leaves {
            if leaf.n_tot == 0 {
                continue;
            }

            // Sync field state thresholds for this leaf
            let max_v = params.abs_mag_limits[1].min(params.app_mag_limits[1] - leaf.dmod);
            if (isochrone_state.max_v - max_v).abs() > isochrone_state.delta_max_v {
                isochrone_state.max_v = max_v;
            }

            let mut leaf_imf = imf.clone();
            leaf_imf.cpd_min = leaf.cpd_min;
            leaf_imf.cpd_max = leaf.cpd_max;

            for star_idx in 0..leaf.n_tot {
                // Per-star deterministic stream matching CUDA thread stream
                let mut star_rng = Philox::from_stream(
                    params.seed as u64,
                    0,
                    pop_id as u32,
                    leaf.orig_leaf_idx as u32,
                    star_idx as u32,
                    1,
                );

                let mut star = StarParticle {
                    popid: pop_id as i32,
                    satid: 0,
                    partid: 0,
                    ..Default::default()
                };

                let bh_node = BHNode {
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
                isochrones.interpolate_star(&mut star, &isochrone_state);
                star.partid = 0;

                if star.abs_mags[0] < 1e4 {
                    if option_e == 0 {
                        let pos_age = pop.generate_pos(&bh_node, &mut star_rng);
                        star.pos = [pos_age[0], pos_age[1], pos_age[2]];
                    }

                    pop.set_vel(&mut star, vcirc, &mut star_rng);

                    // Galactocentric to Heliocentric frame shift
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
                        spawned.push(star);
                    }
                }
            }
        }

        Ok(spawned)
    }
}
