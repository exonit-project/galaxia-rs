<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Benchmark results

All figures are wall-clock measurements of the **CPU fp64** build (the only supported configuration).
They were measured on a small number of machines and are indicative, not guarantees. GPU
backends are experimental and not benchmarked here.

Test machines:

* **Linux x86_64**: 8-core / 16-thread AMD Ryzen 7 class CPU, 92 GB RAM.
* **macOS arm64**: Apple M4, 10 cores, 16 GB unified memory.

The reference for comparison is the original Galaxia 0.7.2 C++ binary (single process, single thread).

## 1. Single-field mode (`galaxia-rs -r field.par`, one thread)

Field: 0.1 deg^2 cone, DCMC photometric system, J <= 22.

| Field | Position (l, b) | Stars (C++ / Rust) | Galaxia C++ | galaxia-rs (CPU, 1 thread) |
|---|---|---:|---:|---:|
| High latitude | l=0, b=60 | 850 / 883 | 13.56 s | 14.35 s |
| Mid latitude | l=60, b=10 | 32,911 / 33,056 | 15.93 s | 14.87 s |
| Bulge / plane | l=1, b=-3 | 7,190,796 / 7,194,940 | 38.06 s | 23.44 s |

(The star counts differ slightly because the random-number generators differ; see VALIDATION.md.)

Where single-field time goes: loading the ten BHTree files takes 10.6-18.8 s depending on disk speed,
the Padova isochrones ~0.6 s, the 3-D extinction maps ~0.4 s; the actual star generation for a
mid-latitude field takes ~5.1 s on the CPU (C++: ~9.0 s); writing the EBF file takes < 0.01 s.
Single-field mode therefore pays the data-loading cost on every run.

## 2. Batch mode (`galaxia-rs batch --csv fields.csv --out-dir out/`)

Batch mode loads the model data (10 BHTrees, isochrones, extinction maps) **once** into memory,
shared between worker threads, and processes the fields from a dynamic queue. A failure in one
field is reported on stderr and makes the process exit non-zero at the end; it does not abort the
other fields.

* Throughput measured on a 230-field all-sky run, single thread: ~1,804 fields/hour on the CPU
  (this depends strongly on the field mix; the dense-field benchmarks below are slower per field).
* Peak resident memory is independent of the number of threads (data is shared): ~6.2-6.7 GiB
  in the dense-field runs below (~5.4 GiB on the 230-field set).

### 2.1 Thread scaling, Apple M4 (10 cores, 16 GB), 300 dense fields, `--tmag-ceiling 22.5`

| Threads | Batch wall (s) | Throughput (fields/h) | Speed-up | Peak RSS (GiB) |
|:---:|---:|---:|---:|---:|
| 1 | 996.50 | 1,083.8 | 1.00x | 6.63 |
| 2 | 506.67 | 2,131.6 | 1.97x | 6.42 |
| 4 | 264.07 | 4,089.9 | 3.77x | 6.51 |
| 7 | 189.35 | 5,703.6 | 5.26x | 6.69 |

After the later correctness fixes (see VALIDATION.md) the same benchmark at 7 threads gave
190.86 s (5,658.7 fields/h, 0.79% lower) with a peak RSS of 6.68 GiB.

### 2.2 Thread scaling, Linux x86_64 (16 threads, 92 GB), 300 dense fields

| Threads | Batch wall (s) | Throughput (fields/h) | Speed-up | Peak RSS (GiB) | Notes |
|:---:|---:|---:|---:|---:|:---|
| 1 | 1,912.85 | 564.6 | 1.00x | 6.17 | one pinned core |
| 6 | 385.93 | 2,798.4 | 4.96x | 6.17 | six pinned cores |
| 12 | 255.58 | 4,225.7 | 7.48x | 6.17 | shared host, other load present |

### 2.3 230-field validation set on the Apple M4 at 7 threads

Wall-clock 135.85 s, 6,094.8 fields/hour, resident memory flat at ~5.4 GiB.
The 230-field set is `validation/validation_230_fields.csv`; the same set run with 1 thread took
692.41 s and with 7 threads 135.82 s, producing byte-identical output (see VALIDATION.md).

Benchmark driver: `scripts/run_bench.py`.
