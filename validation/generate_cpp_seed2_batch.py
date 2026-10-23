#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Generates C++ Galaxia reference fields for seed2 = seed_for_tic(tic) + 100003
across all 230 validation fields in parallel (8 workers on cores 4-11, nice -n 10).
Converts to NPZ with Tmag <= 22.5 cut, matching /tmp/val230_ref format exactly.
"""

import os
import sys
import csv
import time
import shutil
import subprocess
import multiprocessing as mp
import numpy as np

sys.path.insert(0, "docs/spec_scripts")
from read_ebf import read_ebf_item

GALAXIA_BIN = os.environ.get("GALAXIA_BIN", "galaxia")  # original C++ Galaxia 0.7.2 executable
GALAXIA_DATA = os.environ["GALAXIA_DATA_DIR"]
CATALOG_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_230_fields.csv")
WORK_DIR = "/tmp/val230_cpp_seed2_work"
OUT_DIR = "/tmp/val230_cpp_seed2"

REQUIRED_KEYS = ["rad", "mact", "grav", "teff", "feh", "dcmc_j", "dcmc_h", "dcmc_ks", "exbv_solar"]

def stassun_tmag(J, Ks):
    c = J - Ks
    T = np.full_like(J, np.nan, dtype=float)
    m1 = (-0.1 <= c) & (c <= 0.70)
    T[m1] = J[m1] + 1.22163 * c[m1]**3 - 1.74299 * c[m1]**2 + 1.89115 * c[m1] + 0.0563
    m2 = (0.7 < c) & (c <= 1.0)
    T[m2] = J[m2] - 269.372 * c[m2]**3 + 668.453 * c[m2]**2 - 545.64 * c[m2] + 147.811
    m3 = c < -0.1
    T[m3] = J[m3] + 0.5
    m4 = c > 1.0
    T[m4] = J[m4] + 1.75
    return T

def process_field(item):
    tic = item["tic"]
    l = float(item["gal_l"])
    b = float(item["gal_b"])
    seed = (int(tic) % 100000 + 1) + 100003

    field_work = os.path.join(WORK_DIR, f"tic_{tic}")
    os.makedirs(field_work, exist_ok=True)

    out_name = f"s2_{tic}"
    ebf_path = os.path.join(field_work, f"{out_name}.ebf")
    npz_path = os.path.join(OUT_DIR, f"field_{tic}.npz")

    if os.path.exists(npz_path):
        return tic, True

    par_content = f"""outputFile                          {out_name}
outputDir                           {field_work}/
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
    par_file = os.path.join(field_work, f"{out_name}.par")
    with open(par_file, "w") as f:
        f.write(par_content)

    env = os.environ.copy()
    env["GALAXIA_DATA"] = GALAXIA_DATA
    cmd = [GALAXIA_BIN, "-r", par_file]
    r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # C++ returns code 1 on success due to exit(1) in main
    if not os.path.exists(ebf_path):
        return tic, False

    # Convert EBF to NPZ with Tmag <= 22.5 cut
    d = {}
    for k in REQUIRED_KEYS:
        arr = read_ebf_item(ebf_path, "/" + k)
        if k == "exbv_solar":
            d["ebv"] = arr
        else:
            d[k] = arr

    rad = d["rad"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dm = 5.0 * np.log10(rad * 1000.0) - 5.0
        J = d["dcmc_j"] + dm + 0.282 * 3.1 * d["ebv"]
        Ks = d["dcmc_ks"] + dm + 0.112 * 3.1 * d["ebv"]
    T = stassun_tmag(J, Ks)
    keep = np.isfinite(T) & (T <= 22.5)
    filtered = {k: v[keep] for k, v in d.items()}

    np.savez(npz_path, **filtered)
    shutil.rmtree(field_work, ignore_errors=True)
    return tic, True

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    fields = []
    with open(CATALOG_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fields.append(row)

    print(f"Generating C++ seed2 for {len(fields)} fields using 8 parallel workers...")
    t0 = time.time()

    # Use 8 workers
    with mp.Pool(processes=8) as pool:
        done = 0
        for tic, ok in pool.imap_unordered(process_field, fields):
            done += 1
            if not ok:
                print(f"WARNING: Field TIC {tic} failed!")
            if done % 25 == 0 or done == len(fields):
                elapsed = time.time() - t0
                rate = done / elapsed * 3600.0
                print(f"  [{done:>3}/{len(fields)}] finished in {elapsed:.1f}s ({rate:.1f} fields/h)...")

    shutil.rmtree(WORK_DIR, ignore_errors=True)
    print(f"\nCompleted C++ seed2 generation for {len(fields)} fields in {time.time() - t0:.2f}s.")

if __name__ == "__main__":
    main()
