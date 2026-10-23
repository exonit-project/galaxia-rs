<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Validation summary

This document summarises the evidence gathered for the **CPU fp64** build of galaxia-rs
(version 0.2.2). It is a summary of internal checks, not an independent publication. Numbers
are reported as measured; where a check formally failed, that is stated. GPU backends were
not validated (see README) and are not covered here.

Supporting files in this repository: `tests/` (test suite), `tests/cpp/check_cpp_boundaries.cpp`
and `scripts/run_cpp_parity_tests.sh` (C++-linked harness), `validation/` (comparison scripts and the
230-field set), `docs/BENCHMARK_RESULTS.md`, `docs/PORTING_SPEC.md` (algorithm description).

## 1. What "equivalent to Galaxia 0.7.2" means here

* Galaxia (C++) draws random numbers with `drand48`; galaxia-rs uses a counter-based Philox4x32-10
  generator whose stream depends only on `(seed, field id, population id, BHTree leaf index)`.
  Individual stars therefore do **not** coincide with C++ output; the populations are expected to be
  **statistically** equivalent.
* Everything deterministic (grids, interpolation, thresholds, caches, accept/reject boundaries,
  densities) is intended to be bit-for-bit equivalent to the C++ code, and is tested as such below.

## 2. C++-linked parity harness

`tests/cpp/check_cpp_boundaries.cpp` is compiled against the **original, compiled Galaxia 0.7.2
objects** (`IsochroneDB.o`, `Isochrone.o`, `IsochroneBase.o`, `TableInterpolator.o`, `Sampler.o`, ...;
see `scripts/run_cpp_parity_tests.sh`). It generates `tests/fixtures/cpp_boundaries.json`
(34,216 lines of reference values), which `tests/test_cpp_parity.rs` compares against the Rust
implementation. It checks:

* witnesses for the leaf-specific mass cutoff and `maxV` cache behaviour (a case found by an
  independent code review in which an earlier Rust version accepted a star that C++ rejects);
* witnesses for float-to-double quantisation at age/metallicity grid transitions;
* the full Padova age grid (within 1e-7) and half-grid points, and the [Fe/H] grid (within 1e-7);
* hysteresis of the `Vmon` table (within 0.005 and 0.02);
* order and reset of leaves across two consecutive fields;
* the thin-disc density with option E = 0.

Result: **7/7 parity tests pass**; **10/10 robustness tests pass** (`tests/test_parity_fixes.rs`:
forced batch failures, missing EBF output, rejection of non-numeric flags, refusal of non-CPU
backends, CSV validation, atomic output). The complete `cargo test --release` suite at the tagged
release contained **79 tests, all passing** (67/67 at the earlier gate). The suite in this repository
is slightly smaller because the GPU-backend tests are only built with the experimental features.

Issues found by an independent code review and fixed before the 0.2.2 release (technical summary):

* the Padova age grid is rebuilt exactly as `Age[i] = icv[0].age + i*(icv[1].age - icv[0].age)`
  (`IsochroneBase.cpp:307-310`) instead of using the values in the file;
* values stored as `float` are promoted to `double` before comparison with the `double` grid
  (`IsochroneDB.cpp:306-316, 344-350`), and the Chabrier / bolometric-correction tables are parsed as
  `float` promoted to `double` (`TableInterpolator.cpp:159-161`);
* the per-track leaf minimum-mass rejection and `maxV` cache are reproduced as in C++;
* batch mode exits non-zero when any field fails, when `succeeded + failed != total`, or when a
  field reported as successful has no valid output file;
* build provenance no longer hard-codes a fallback commit hash and marks uncommitted builds `-dirty`;
* a non-numeric `--tmag-ceiling` is rejected both in the pre-scan and at run time.

## 3. Determinism

Validation set: `validation/validation_230_fields.csv` (230 fields spread over the sky, stratified in
galactic latitude and including the dense bulge/plane).

| Check | Result |
|---|---|
| 1 thread vs 7 threads (Apple M4) | 230 / 230 EBF files with identical SHA-256 (5,392,577 retained stars in both runs) |
| Linux x86_64 vs macOS arm64 vs Windows 11 x86_64 (CPU fp64, 230 fields) | 230 / 230 EBF files with identical SHA-256 (5,392,577 retained stars; 23,875,462 generated before cuts, on each OS) |
| Earlier multi-threaded implementation: 1 thread vs reference single-thread commit, and 1 vs 7 threads | 230 / 230 identical in both comparisons |

The random stream of each field depends only on `(seed, field id, population id, leaf index)`,
not on the thread that processes it or on queue order, which is why any thread count gives the
same bytes.

## 4. Size of the change introduced by the parity fixes

Comparison of the build before the parity fixes with the 0.2.2 build over the same 230 fields:

* total retained stars 5,392,532 -> 5,392,577 (+45, +0.00083%);
* fields whose count changed by more than 1%: **0 / 230**;
* median relative difference 0.0057% (57 ppm); mean 0.0119%; maximum 0.1919% (one star in a
  521-star field); 132 / 230 fields changed at all (57.4%), consistent with fixes that only affect
  stars at grid and cutoff boundaries;
* throughput at 7 threads decreased by 0.79% (requirement was <= 10%); peak RSS unchanged.

## 5. Statistical comparison against C++ populations

**Noise floor.** On 10 reference fields the C++ code was run twice with different seeds
(264,146 stars per run in total). Total-count ratio 1.0004 (264,252 / 264,146); median
per-field ratio 1.0033 (sigma 0.0516, consistent with Poisson noise in sparse fields);
Kolmogorov-Smirnov on T <= 21: D = 0.00561, p = 0.4695; pooled chi-square p-values between 0.14 and
0.99 for the observable families. This C++-vs-C++ spread defines the acceptable band.

**Pre-registered acceptance criteria** (fixed before the comparison was computed), for a run of at
least 200 fields: (1) all pooled observable families (T and J magnitude, J-Ks, Teff, log g, [Fe/H],
mass, population fractions) with chi-square p > 0.01 and a two-sample KS test on T <= 21 with
p >= 0.05; (2) per-field p-values compatible with a uniform distribution, and median
Rust/C++ count ratio within 0.99-1.01; (3) no latitude stratum with |z| >= 2.5; (4) manual audit
of the five worst fields; (5) no silent failures.

**Result on the 230 fields** (reference C++ populations vs galaxia-rs; this comparison was run on a
build made before the parity fixes of section 4, which changed the totals by +0.00083%):

* total stars 5,390,545 (C++) vs 5,392,532 (Rust): ratio 1.00037; median per-field ratio 1.0001;
  standard deviation of the ratio 0.0311 (inside the C++ noise floor, 0.0516);
* pooled chi-square: T magnitude chi2 = 10.45 (17 dof, p = 0.8838); J magnitude 11.07 (16 dof,
  p = 0.8054); class 0.62 (2 dof, p = 0.7350); Teff 4.37 (6 dof, p = 0.6274); T x class 25.35
  (33 dof, p = 0.8268, max |z| = 2.89);
* two-sample KS on T <= 21 (4,601,096 stars): D = 0.00076, p = 0.5252;
* count ratio per |b| stratum: [0,5] deg 1.0005 (92 fields); (5,15] 0.9995 (61); (15,30] 1.0021 (32);
  (30,60] 1.0063 (33); (60,90] 0.9812 (12 fields, 6,276 stars; typical Poisson noise ~1.3%);
* no field exceeded |z| = 4.5 in the audit of the worst fields; 230 / 230 fields completed, no silent failures.

Overall verdict recorded for this comparison: pass.

## 6. Downstream check: false-positive-probability calculation on 40 known TESS objects of interest

As an end-to-end test, a downstream false-positive-probability (FPP) calculation that uses a
synthetic background-star population was run for 40 known TESS objects of interest with three
populations: the original C++ output (A), a second independent C++ run with a different seed (B,
giving the noise floor), and galaxia-rs (R). Five criteria were registered before the run:

1. median deviation of R from A must not exceed 1.5 x the median deviation of B from A;
2. a paired Wilcoxon test between the deviations must show no significant difference (p > 0.05);
3. no object may deviate by more than a bound derived from the B-vs-A deviation, its standard error, and 0.01;
4. the number of decision-threshold flips between A and R must not exceed the A-vs-B flips + 1;
5. the number of verdict differences between A and R must not exceed the A-vs-B differences + 1.

Results with 50,000 draws and 3 Monte-Carlo seeds: criterion 1: median deviation 0.00018 vs
0.00054 (ratio 0.500), pass; criterion 2: W = 308.0, p = 0.5117, pass; **criterion 3: one violation
(deviation 0.03387 against a bound of 0.01426), FAIL as registered**; criterion 4: 0 flips (limit 1),
pass; criterion 5: 1 verdict difference (limit 1), pass. The initial overall result was therefore
FAIL, which triggered a targeted re-test of the violating object (and the verdict-differing one),
pre-registered in advance: 500,000 draws and 10 seeds. At that resolution the mean FPP values of A,
B and R agree to 1e-6 (R - A deviation 0.000000, B - A deviation 0.000002, bound 0.010003); the
distribution over seeds is unimodal with no low mode, and the second object also agrees exactly.
Criterion 3 therefore passes at the higher resolution, and the large initial deviation is
attributed to Monte-Carlo noise at 50,000 draws, which is the same behaviour the independent
C++ run B shows. This check is evidence of agreement on a downstream quantity, not a substitute
for the checks above.

## 7. Known limitations of the validation

* Statistical, not bit-wise, equivalence to C++ (different RNG).
* The statistical comparison (section 5) was run on the build before the parity fixes; section 4
  bounds the difference to the 0.2.2 build, but the comparison was not repeated in full.
* Only the DCMC photometric system and the generation path (`-r` and batch mode) were exercised;
  other photometric systems and the other Galaxia run options were not validated.
* The GPU backends were not validated and use single or unvalidated precision.
