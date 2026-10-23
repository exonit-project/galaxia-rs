#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Measures the baseline stochastic variation of C++ Galaxia running on the same fields
with two different random seeds (seed1 = seed_for_tic, seed2 = seed1 + 100003).

This establishes the natural variance band of Monte Carlo sampling for Galaxia.
"""

import os
import sys
import csv
import subprocess
import time
import shutil
import numpy as np

sys.path.insert(0, "docs/spec_scripts")
from read_ebf import read_ebf_item

GALAXIA_BIN = os.environ.get("GALAXIA_BIN", "galaxia")  # original C++ Galaxia 0.7.2 executable
GALAXIA_DATA = os.environ["GALAXIA_DATA_DIR"]
CATALOG_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_230_fields.csv")
WORK_DIR = "/tmp/cpp_seed_noise"

REQUIRED_KEYS = ["rad", "mact", "grav", "teff", "feh", "dcmc_j", "dcmc_h", "dcmc_ks", "exbv_solar", "popid"]

def convert_ebf_to_npz(ebf_path, out_npz):
    d = {}
    for k in REQUIRED_KEYS:
        try:
            arr = read_ebf_item(ebf_path, "/" + k)
            if k == "exbv_solar":
                d["ebv"] = arr
            else:
                d[k] = arr
        except Exception as e:
            print(f"Error reading {k} from {ebf_path}: {e}")
    np.savez(out_npz, **d)

def run_field(tic, l, b, seed, out_dir, out_name):
    par = f"""outputFile                          {out_name}
outputDir                           {out_dir}/
photoSys                            DCMC
magcolorNames                       J,J-Ks
appMagLimits[0]                     -100
appMagLimits[1]                     22
absMagLimits[0]                     -1000
absMagLimits[1]                     1000
colorLimits[0]                      -1000
colorLimits[1]                      1000
geometryOption                      1
longitude                           {l:.6f}
latitude                            {b:.6f}
surveyArea                          0.1
fSample                             1.0
popID                               -1
warpFlareOn                         1
seed                                {seed}
r_max                               1000
starType                            0
photoError                          0
"""
    par_file = os.path.join(out_dir, f"{out_name}.par")
    with open(par_file, "w") as fp:
        fp.write(par)

    env = os.environ.copy()
    env["GALAXIA_DATA"] = GALAXIA_DATA
    cmd = [GALAXIA_BIN, "-r", f"{out_name}.par"]
    res = subprocess.run(cmd, cwd=out_dir, env=env, capture_output=True, text=True)
    ebf_out = os.path.join(out_dir, f"{out_name}.ebf")
    if not os.path.exists(ebf_out):
        raise RuntimeError(f"C++ failed for {out_name}: {res.stderr}\n{res.stdout}")
    return ebf_out

def main():
    if not os.path.exists(GALAXIA_BIN):
        print(f"C++ binary not found at {GALAXIA_BIN}")
        return

    os.makedirs(WORK_DIR, exist_ok=True)
    dir_seed1 = os.path.join(WORK_DIR, "seed1")
    dir_seed2 = os.path.join(WORK_DIR, "seed2")
    os.makedirs(dir_seed1, exist_ok=True)
    os.makedirs(dir_seed2, exist_ok=True)

    # Pick 10 representative fields from chunk 0 (high, mid, low latitudes)
    fields = []
    with open(CATALOG_CSV, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i in [0, 1, 2, 3, 4, 10, 20, 50, 80, 100]:
                fields.append(row)

    print(f"Running C++ baseline variance on {len(fields)} fields with 2 different seeds...")
    count_ratios = []

    for item in fields:
        tic = item["tic"]
        l = float(item["gal_l"])
        b = float(item["gal_b"])
        seed1 = int(tic) % 100000 + 1
        seed2 = seed1 + 100003

        print(f"Processing TIC {tic} (l={l:.2f}, b={b:.2f})...")
        t0 = time.time()
        ebf1 = run_field(tic, l, b, seed1, WORK_DIR, f"s1_{tic}")
        ebf2 = run_field(tic, l, b, seed2, WORK_DIR, f"s2_{tic}")

        npz1 = os.path.join(dir_seed1, f"field_{tic}.npz")
        npz2 = os.path.join(dir_seed2, f"field_{tic}.npz")
        convert_ebf_to_npz(ebf1, npz1)
        convert_ebf_to_npz(ebf2, npz2)

        d1 = np.load(npz1)
        d2 = np.load(npz2)
        n1 = len(d1["rad"])
        n2 = len(d2["rad"])
        ratio = n2 / n1 if n1 > 0 else 1.0
        count_ratios.append(ratio)
        print(f"  TIC {tic}: N(seed1)={n1}, N(seed2)={n2}, ratio={ratio:.4f} in {time.time() - t0:.2f}s")

    print(f"\n--- C++ Seed Variation Summary ---")
    print(f"Median Count Ratio N(seed2)/N(seed1): {np.median(count_ratios):.4f}")
    print(f"Min / Max Count Ratio: {np.min(count_ratios):.4f} / {np.max(count_ratios):.4f}")
    print(f"Std dev of Ratio: {np.std(count_ratios):.4f}")

    # Now run compare_stats on the two C++ runs
    print("\nRunning compare_stats.py on C++ (seed1) vs C++ (seed2)...")
    cmd = [
        sys.executable,
        "validation/compare_stats.py",
        dir_seed1,
        dir_seed2
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
