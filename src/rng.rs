// SPDX-License-Identifier: AGPL-3.0-or-later
//! Counter-based Philox4x32-10 pseudo-random number generator.
//!
//! Philox4x32-10 is a counter-based random number generator from the Random123
//! family (Salmon, Moraes, Dror, Shaw, SC 2011).
//!
//! # CUDA Parity
//! This implementation uses exact 32-bit integer arithmetic (multiplication, XOR,
//! and addition modulo 2^32) identical to the Philox4x32-10 implementation in CUDA C.
//!
//! # Stream Mapping
//! Given a simulation stream tuple `(seed: u64, field: u32, pop: u32, leaf: u32, star: u32, purpose: u32)`:
//! - Key `[u32; 2]`:
//!   - `key[0] = (seed as u32) ^ fmix32(purpose)`
//!   - `key[1] = ((seed >> 32) as u32) ^ fmix32(pop)`
//! - Counter `[u32; 4]`:
//!   - `ctr[0] = 0` (block index, incremented every 4 u32 words generated)
//!   - `ctr[1] = star` (star particle index within leaf)
//!   - `ctr[2] = leaf` (BHTree leaf index)
//!   - `ctr[3] = field` (survey field index)
//!
//! This deterministic mapping guarantees collision-free streams across seeds, fields,
//! populations, tree leaves, stars, and calculation purposes (e.g. IMF mass sampling,
//! positional jitter, velocity dispersions).

use std::f64::consts::PI;

pub const PHILOX_M4X32_0: u32 = 0xD2511F53;
pub const PHILOX_M4X32_1: u32 = 0xCD9E8D57;
pub const PHILOX_W32_0: u32 = 0x9E3779B9;
pub const PHILOX_W32_1: u32 = 0xBB67AE85;

/// MurmurHash3 32-bit finalizer (fmix32) for fast integer mixing.
#[inline(always)]
pub fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

/// Single round of Philox4x32.
#[inline(always)]
pub fn philox4x32_round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let prod0 = (ctr[0] as u64) * (PHILOX_M4X32_0 as u64);
    let hi0 = (prod0 >> 32) as u32;
    let lo0 = prod0 as u32;

    let prod1 = (ctr[2] as u64) * (PHILOX_M4X32_1 as u64);
    let hi1 = (prod1 >> 32) as u32;
    let lo1 = prod1 as u32;

    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// Key bump step in Philox4x32 (Weyl constants added).
#[inline(always)]
pub fn philox4x32_bumpkey(mut key: [u32; 2]) -> [u32; 2] {
    key[0] = key[0].wrapping_add(PHILOX_W32_0);
    key[1] = key[1].wrapping_add(PHILOX_W32_1);
    key
}

/// Philox4x32-10 core permutation: 10 rounds of Philox4x32.
pub fn philox4x32_10(mut ctr: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    ctr = philox4x32_round(ctr, key);
    for _ in 1..10 {
        key = philox4x32_bumpkey(key);
        ctr = philox4x32_round(ctr, key);
    }
    ctr
}

/// State of a Philox4x32-10 random number generator.
#[derive(Debug, Clone, PartialEq)]
pub struct Philox {
    key: [u32; 2],
    ctr: [u32; 4],
    buffer: [u32; 4],
    buffer_idx: usize,
    normal_cache: Option<f64>,
}

impl Philox {
    /// Create a new Philox generator with the given 64-bit key and counter starting at 0.
    pub fn new(key: [u32; 2]) -> Self {
        let ctr = [0u32; 4];
        let buffer = philox4x32_10(ctr, key);
        Self {
            key,
            ctr,
            buffer,
            buffer_idx: 0,
            normal_cache: None,
        }
    }

    /// Create a new Philox generator with explicit key and initial counter.
    pub fn with_counter(key: [u32; 2], ctr: [u32; 4]) -> Self {
        let buffer = philox4x32_10(ctr, key);
        Self {
            key,
            ctr,
            buffer,
            buffer_idx: 0,
            normal_cache: None,
        }
    }

    /// Construct a deterministic stream from the simulation coordinates:
    /// `(seed, field, pop, leaf, star, purpose)`.
    ///
    /// See module-level documentation for details on the mapping.
    pub fn from_stream(
        seed: u64,
        field: u32,
        pop: u32,
        leaf: u32,
        star: u32,
        purpose: u32,
    ) -> Self {
        let key = [
            (seed as u32) ^ fmix32(purpose),
            ((seed >> 32) as u32) ^ fmix32(pop),
        ];
        let ctr = [0u32, star, leaf, field];
        Self::with_counter(key, ctr)
    }

    /// Construct a stream from a 6-tuple `(seed, field, pop, leaf, star, purpose)`.
    pub fn from_tuple(stream: (u64, u32, u32, u32, u32, u32)) -> Self {
        Self::from_stream(stream.0, stream.1, stream.2, stream.3, stream.4, stream.5)
    }

    /// Step the counter and refill the 4-word output buffer.
    fn refill(&mut self) {
        let (c0, carry) = self.ctr[0].overflowing_add(1);
        self.ctr[0] = c0;
        if carry {
            let (c1, carry) = self.ctr[1].overflowing_add(1);
            self.ctr[1] = c1;
            if carry {
                let (c2, carry) = self.ctr[2].overflowing_add(1);
                self.ctr[2] = c2;
                if carry {
                    self.ctr[3] = self.ctr[3].wrapping_add(1);
                }
            }
        }
        self.buffer = philox4x32_10(self.ctr, self.key);
        self.buffer_idx = 0;
    }

    /// Generate the next 32-bit unsigned random integer.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.buffer_idx >= 4 {
            self.refill();
        }
        let val = self.buffer[self.buffer_idx];
        self.buffer_idx += 1;
        val
    }

    /// Generate a 53-bit uniform random floating-point number in the open interval `(0, 1)`.
    ///
    /// Consumes two 32-bit random integers. The returned value is strictly greater than 0.0
    /// and strictly less than 1.0.
    pub fn uniform_f64(&mut self) -> f64 {
        let w0 = self.next_u32();
        let w1 = self.next_u32();
        let u53 = ((w0 as u64) << 21) | ((w1 as u64) >> 11);
        // Map 0..=2^53-1 to open interval (0, 1):
        // min: 0.5 / 2^53 ~= 5.55e-17 > 0
        // max: (2^53 - 0.5) / 2^53 = 1 - 5.55e-17 < 1
        (u53 as f64 + 0.5) * (1.0 / 9007199254740992.0)
    }

    /// Generate a 24-bit uniform random floating-point number in the open interval `(0, 1)`.
    ///
    /// Consumes one 32-bit random integer. The returned value is strictly greater than 0.0
    /// and strictly less than 1.0.
    pub fn uniform_f32(&mut self) -> f32 {
        let w0 = self.next_u32();
        let u24 = w0 >> 8;
        // Map 0..=2^24-1 to open interval (0, 1):
        (u24 as f32 + 0.5f32) * (1.0f32 / 16777216.0f32)
    }

    /// Generate a standard normal random variate (mean 0, variance 1).
    ///
    /// Implements the Box-Muller polar-free direct trigonometric transform:
    /// Given independent `u1, u2` uniform in `(0, 1)`:
    /// `r = sqrt(-2 ln u1)`, `theta = 2 pi u2`
    /// `z0 = r cos(theta)`, `z1 = r sin(theta)`
    /// Returns `z0` and caches `z1` for the subsequent call.
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(cached) = self.normal_cache.take() {
            return cached;
        }
        let u1 = self.uniform_f64();
        let u2 = self.uniform_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        self.normal_cache = Some(z1);
        z0
    }

    /// Returns the current key.
    pub fn key(&self) -> [u32; 2] {
        self.key
    }

    /// Returns the current counter.
    pub fn counter(&self) -> [u32; 4] {
        self.ctr
    }
}

impl From<(u64, u32, u32, u32, u32, u32)> for Philox {
    fn from(t: (u64, u32, u32, u32, u32, u32)) -> Self {
        Self::from_tuple(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_philox4x32_10_kat_zero() {
        // KAT vector 1: counter = 0, key = 0
        let ctr = [0u32, 0, 0, 0];
        let key = [0u32, 0];
        let res = philox4x32_10(ctr, key);
        assert_eq!(
            res,
            [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
            "Mismatch for ctr=0, key=0"
        );
    }

    #[test]
    fn test_philox4x32_10_kat_ones() {
        // KAT vector 2: counter = 0xffffffff x 4, key = 0xffffffff x 2
        let ctr = [0xFFFFFFFFu32; 4];
        let key = [0xFFFFFFFFu32; 2];
        let res = philox4x32_10(ctr, key);
        assert_eq!(
            res,
            [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
            "Mismatch for ctr=ffffffff x 4, key=ffffffff x 2"
        );
    }

    #[test]
    fn test_stream_constructor_determinism() {
        let stream1 = Philox::from_stream(123456789, 1, 2, 100, 50, 0);
        let stream2 = Philox::from_stream(123456789, 1, 2, 100, 50, 0);
        assert_eq!(stream1, stream2);

        let stream3 = Philox::from_stream(123456789, 1, 2, 100, 51, 0);
        assert_ne!(stream1, stream3);
    }

    #[test]
    fn test_uniform_f64_open_interval() {
        let mut rng = Philox::new([123, 456]);
        for _ in 0..10_000 {
            let u = rng.uniform_f64();
            assert!(u > 0.0 && u < 1.0, "uniform_f64 out of bounds: {}", u);
        }
    }

    #[test]
    fn test_uniform_f32_open_interval() {
        let mut rng = Philox::new([999, 888]);
        for _ in 0..10_000 {
            let u = rng.uniform_f32();
            assert!(u > 0.0 && u < 1.0, "uniform_f32 out of bounds: {}", u);
        }
    }

    #[test]
    fn test_standard_normal_stats() {
        let mut rng = Philox::new([42, 84]);
        let n = 20_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let z = rng.standard_normal();
            sum += z;
            sum_sq += z * z;
        }
        let mean = sum / (n as f64);
        let variance = (sum_sq / (n as f64)) - mean * mean;
        assert!(mean.abs() < 0.05, "mean not close to 0: {}", mean);
        assert!(
            (variance - 1.0).abs() < 0.05,
            "variance not close to 1: {}",
            variance
        );
    }
}
