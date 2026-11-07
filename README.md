<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# galaxia-rs

A Rust port of **Galaxia 0.7.2**, the code that generates synthetic stellar surveys of the Milky Way
from the Besancon-like analytic model of Sharma et al. (2011). Given a sky field (a cone or a
rectangle), a photometric system and magnitude/colour limits, it produces a catalogue of stars with
positions, velocities, ages, metallicities, masses, stellar parameters, magnitudes and extinction,
written as EBF files (the same format as the original).

Galaxia was written by **Sanjib Sharma and Joss Bland-Hawthorn**; this port is a derivative work.
**This is an unofficial port**: it is not maintained, reviewed or endorsed by the original authors, and any
problem in it is ours, not theirs. See [NOTICE](NOTICE) and [LICENSE](LICENSE). **If you use this software, please cite Sharma, S.,
Bland-Hawthorn, J., Johnston, K. V., Binney, J. 2011, ApJ, 730, 3 first.**

## Relation to Galaxia 0.7.2

* Original: <http://galaxia.sourceforge.net> (C++, AGPL-3.0-or-later).
* galaxia-rs reads the **same `GalaxiaData` files** and implements the same model (BHTree
  octrees, Padova/DCMC isochrones, 3-D extinction maps, the same IMFs and density laws).
* Deterministic parts (grids, interpolation, accept/reject boundaries, caches) are meant to match the
  C++ code exactly and are tested against the original compiled objects (see
  [VALIDATION.md](VALIDATION.md)).
* The random-number generator is **different** (counter-based Philox4x32-10 instead of `drand48`), so
  individual stars differ from C++ output: populations are **statistically** equivalent, **not
  bitwise** identical, to the C++ ones.
* Additions: a multi-threaded batch mode that loads the data once, deterministic per-field random
  streams, output provenance metadata, stricter validation and error exits.
* Not ported: the BHTree setup option `-s` (use the original binary to generate BHTree files, or use
  the ones shipped in `GalaxiaData`), and anything not needed for the supported generation path
  (see Limitations).

## Getting the data

The data are **not** included here and are not redistributed. Download the Galaxia 0.7.2 distribution
and its `GalaxiaData` directory from <http://galaxia.sourceforge.net> (a few GB). Point
galaxia-rs at it with `--data-dir` (batch mode), the `GALAXIA_DATA_DIR` environment variable, or the
`codeDataDir` tag of a parameter file. It must contain `BHTree-2.2/`, `Isochrones/`, `Extinction/`,
`Model/`, ...

## Build

```sh
cargo build --release          # CPU-only, fp64 (the default and only supported build)
```

Requires a recent stable Rust toolchain. The default build has no GPU code and no non-Rust dependencies.

## Run

Single field, from a Galaxia parameter file:

```sh
export GALAXIA_DATA_DIR=/path/to/GalaxiaData
target/release/galaxia-rs -r tests/fixtures/t.par          # writes <outputDir>/<outputFile>.ebf
target/release/galaxia-rs -r field.par --tmag-ceiling 22.5
target/release/galaxia-rs --version                        # version, git commit, backend, ceiling
```

Batch mode: many fields from a CSV, data loaded **once**, processed by N threads:

```sh
target/release/galaxia-rs batch --csv fields.csv --out-dir out/ \
    --data-dir /path/to/GalaxiaData --threads 8 --tmag-ceiling 22.5
```

* `fields.csv` has a header with at least `tic,ra,dec` (extra columns are ignored). `tic` is an
  integer field identifier (originally a TESS Input Catalog id; any positive integer works and
  determines the random seed, `(id % 100000) + 1`); `ra`, `dec` are ICRS degrees.
* Each field is a 0.1 deg^2 cone, DCMC photometric system (I, J, H, Ks), all ten populations,
  intrinsic J <= 22, warp and flare on. One output per field: `out/gx_<id>.ebf`.
* `--tmag-ceiling X` keeps only stars whose extincted TESS magnitude (Stassun et al. 2018
  relation from J and Ks, plus extinction) is <= X.
* A failing field is reported on stderr and the process exits non-zero at the end; the other fields
  are still processed. Output files are written atomically (temporary file + rename).
* Output files embed a `/log` record with version, git commit and options.

## What is guaranteed

For the **CPU fp64 build**, output is **bit-identical** (same SHA-256 of the EBF files)

* across **Linux x86_64, macOS arm64 and Windows x86_64**, and
* for **any number of threads** (`--threads 1` ... `N`),

as verified on a 230-field set (see [VALIDATION.md](VALIDATION.md)). This holds for the same
source version and the same `GalaxiaData`. Nothing is guaranteed for other configurations.

## Performance

Batch mode amortises the 12-20 s that the original (and single-field mode) spends loading the data.
Measured figures (details and machines in [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md)):

* Single field, one thread: 14.9 s vs 15.9 s for C++ (mid latitude, 32,911 stars); 23.4 s vs 38.1 s for
  C++ on a dense bulge/plane field (7.19 million stars).
* Batch mode, Apple M4 (10 cores), 300 dense fields: 1,083.8 fields/h at 1 thread, 5,703.6 fields/h at
  7 threads (5.26x); 6,094.8 fields/h on the 230-field validation set at 7 threads.
* Batch mode, Linux x86_64 (16 threads), 300 dense fields: 564.6 fields/h at 1 thread, 4,225.7 at 12.
* Peak memory is ~6.2-6.7 GiB regardless of the number of threads (data are shared).

## Tests

```sh
cargo test --release                       # data-free tests only
GALAXIA_DATA_DIR=/path/to/GalaxiaData cargo test --release   # everything
```

* **Without `GALAXIA_DATA_DIR`** the tests that need `GalaxiaData` are *not selected*: each returns
  immediately after printing `NOT SELECTED: ...` (visible with `cargo test -- --nocapture`). This is an
  explicit opt-out, not a verification.
* **With `GALAXIA_DATA_DIR` set** those tests are selected and **fail** if the directory or any other
  required input is missing or wrong. They include generation of whole fields and 3-field batch runs
  (roughly a minute or two in release mode).
* Two further selectors: `GALAXIA_CPP_REF_DIR` (directory with `gx_test_cpp.ebf`, the output of the
  original `galaxia -r tests/fixtures/t.par` with `outputFile gx_test_cpp`; defaults to
  `$TMPDIR/gxt`) for the EBF/extinction comparison against C++ output, and `GALAXIA_PYTHON` (a Python
  with the `ebf` module) for the EBF-interop tests; without `GALAXIA_PYTHON` that test is not selected.

### C++ parity harness

`tests/cpp/check_cpp_boundaries.cpp` is linked against the **compiled original Galaxia 0.7.2 objects**
and regenerates the reference fixtures in `tests/fixtures/cpp_boundaries.json` (a committed copy is
included, so `cargo test` does not need the C++ build). To rebuild them and rerun the comparison:

```sh
GALAXIA_CPP_SRC=/path/to/galaxia-0.7.2/src \      # compiled source tree (contains the *.o files)
GALAXIA_DATA_DIR=/path/to/GalaxiaData \
scripts/run_cpp_parity_tests.sh
# or: scripts/run_cpp_parity_tests.sh /path/to/galaxia-0.7.2/src /path/to/GalaxiaData
```

Statistical comparison tooling is in `validation/` (see [VALIDATION.md](VALIDATION.md)).

## Limitations

* **GPU backends (CUDA, OpenCL) are EXPERIMENTAL, NOT VALIDATED and NOT FOR SCIENCE.** They run in
  single precision on some devices and have known open issues (behaviour of the rejection-loop cap,
  PTX handling, resource leaks on error paths, thread-safety assumptions of the CUDA context). They
  are compiled only with the non-default cargo features `cuda` / `opencl`, and even then the binary
  refuses them unless `--allow-fp32-test` is given. Do not use their output for science. No claim in
  this repository about validation or reproducibility applies to them.
* Only the CPU fp64 build is supported. Only the generation path (`-r`, batch mode) is validated; the
  setup option `-s` is not implemented.
* Output is statistically, not bitwise, equivalent to Galaxia C++ (different RNG). Totals differ
  from C++ by Poisson noise (e.g. +0.04% over 230 fields).
* Bit-identity across platforms was verified on the 230-field set for the tested platforms and
  compiler versions; it relies on strict IEEE-754 double arithmetic and is not guaranteed for other
  targets, toolchains or build flags (e.g. fast-math, other CPU architectures).
* Memory: ~6-7 GiB are needed to hold the model data.

## Citation

Please cite **Sharma, S., Bland-Hawthorn, J., Johnston, K. V., & Binney, J. 2011, ApJ, 730, 3**
("Galaxia: A Code to Generate a Synthetic Survey of the Milky Way") for any use of this software or
its output, and, if relevant, this repository (version and commit, available from
`galaxia-rs --version`).

## Licence

GNU Affero General Public License, version 3 or (at your option) any later version
(AGPL-3.0-or-later). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
