#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch-mode throughput / peak-RSS benchmark for several thread counts (Linux: uses /usr/bin/time -v).

Example:
    scripts/run_bench.py --bin target/release/galaxia-rs --csv validation/benchmark_300_fields.csv \
        --data-dir /path/to/GalaxiaData --threads 1 4 7
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="target/release/galaxia-rs")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data-dir", default=os.environ.get("GALAXIA_DATA_DIR"))
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--tmag-ceiling", default="22.5")
    ap.add_argument("--json-out", default=None, help="optional path for the JSON results")
    args = ap.parse_args()
    if not args.data_dir:
        ap.error("--data-dir or GALAXIA_DATA_DIR is required")

    results = []
    for n in args.threads:
        out_dir = tempfile.mkdtemp(prefix=f"gx_bench_n{n}_")
        cmd = ["/usr/bin/time", "-v", args.bin, "batch", "--csv", args.csv, "--out-dir", out_dir,
               "--data-dir", args.data_dir, "--backend", "cpu", "--threads", str(n),
               "--tmag-ceiling", args.tmag_ceiling]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t0
        if p.returncode != 0:
            raise SystemExit(f"batch run failed (N={n}):\n{p.stderr[-2000:]}")
        rss = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", p.stderr)
        fph = re.search(r"Throughput:\s+([\d.]+)\s+fields/hour", p.stdout)
        bw = re.search(r"Batch wall-clock:\s+([\d.]+)s", p.stdout)
        files = [f for f in os.listdir(out_dir) if f.endswith(".ebf")]
        rss_kb = int(rss.group(1)) if rss else 0
        results.append({
            "threads": n,
            "fields": len(files),
            "wall_clock_s": float(bw.group(1)) if bw else wall,
            "throughput_fields_per_h": float(fph.group(1)) if fph else 0.0,
            "peak_rss_gib": rss_kb / (1024.0 * 1024.0),
        })
        shutil.rmtree(out_dir)

    print(f"{'Threads':<8} | {'Fields':<7} | {'Wall (s)':<10} | {'Fields/h':<10} | {'Peak RSS (GiB)':<14}")
    for r in results:
        print(f"{r['threads']:<8} | {r['fields']:<7} | {r['wall_clock_s']:<10.2f} | "
              f"{r['throughput_fields_per_h']:<10.1f} | {r['peak_rss_gib']:<14.2f}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
