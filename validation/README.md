<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# validation/

Scripts used for the statistical comparison against the original C++ Galaxia (see `../VALIDATION.md`).
They need `numpy` and `scipy`; the C++ ones also need a built Galaxia 0.7.2 executable
(`GALAXIA_BIN`) and `GALAXIA_DATA_DIR`.

* `validation_230_fields.csv` - the 230-field validation set (`tic` = integer field id, ICRS RA/Dec,
  galactic l/b, group id). `benchmark_300_fields.csv` - 300 dense fields used for throughput benchmarks.
* `generate_cpp_seed2_batch.py`, `measure_cpp_seed_noise.py` - run the original C++ code on the fields with
  different seeds (seed = `(id % 100000) + 1`, `+100003` for the second run) to measure the C++-vs-C++ noise floor.
* `convert_rust_batch_to_npz.py` - convert batch-mode EBF output to the NPZ layout used by the comparisons.
* `compare_stats.py` (+ `test_compare_stats.py`), `per_field_pvalues_ks.py`, `audit_batch_results.py`,
  `compare_cpp_and_rust_baselines.py` - pooled chi-square / KS comparisons, per-field p-values, stratified audit.
* `verify_cross_os_parity.py`, `verify_gate5.py` - SHA-256 / star-count comparison of two output directories
  (cross-OS and thread-count determinism; size of a change between two builds).

The C++ reference populations used for the published numbers were produced with the original code
on the same 230 fields; regenerate equivalent ones with the scripts above (the NPZ layout is
described in `convert_rust_batch_to_npz.py` and `generate_cpp_seed2_batch.py`).
