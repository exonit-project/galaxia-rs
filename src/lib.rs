// SPDX-License-Identifier: AGPL-3.0-or-later
//! galaxia-rs: Rust port of Galaxia 0.7.2.

// The C++ original uses truncated literals (pi = 3.14159, 1/sqrt(2) = 0.7071067811865475...) that are
// reproduced on purpose for output compatibility; do not "fix" them to std::f64::consts.
#![allow(clippy::approx_constant)]

pub mod backend;
pub mod batch;
pub mod bhtree;
pub mod coords;
pub mod ebf;
pub mod extinction;
pub mod generator;
pub mod imf;
pub mod interp;
pub mod isochrone;
pub mod params;
pub mod population;
pub mod rng;
pub mod survey;
#[doc(hidden)]
pub mod testutil;
