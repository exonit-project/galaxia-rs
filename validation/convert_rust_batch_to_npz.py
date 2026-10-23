#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Converts galaxia-rs batch EBF files into NPZ files matching the format
expected by compare_stats.py and the reference C++ shards.
"""

import os
import sys
import glob
import time
import numpy as np

# Ensure read_ebf is importable from docs/spec_scripts or spec_scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_SCRIPTS = os.path.join(os.path.dirname(SCRIPT_DIR), "docs", "spec_scripts")
if os.path.exists(SPEC_SCRIPTS):
    sys.path.insert(0, SPEC_SCRIPTS)
else:
    sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "docs", "spec_scripts"))

from read_ebf import read_ebf_item

KEY_MAPPING = {
    "/rad": "rad",
    "/mact": "mact",
    "/grav": "grav",
    "/teff": "teff",
    "/feh": "feh",
    "/dcmc_j": "dcmc_j",
    "/dcmc_h": "dcmc_h",
    "/dcmc_ks": "dcmc_ks",
    "/exbv_solar": "ebv",
    "/popid": "popid",
}

def convert_file(ebf_path: str, out_npz: str):
    data = {}
    for ebf_key, npz_key in KEY_MAPPING.items():
        try:
            arr = read_ebf_item(ebf_path, ebf_key)
            data[npz_key] = arr
        except Exception:
            # Fallback if popid is missing or named differently
            pass
    np.savez(out_npz, **data)

def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val230_rust_ebf"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/val230_rust_npz"
    os.makedirs(out_dir, exist_ok=True)

    ebf_files = sorted(glob.glob(os.path.join(in_dir, "gx_*.ebf")))
    print(f"Converting {len(ebf_files)} EBF files from {in_dir} to {out_dir}...")
    t0 = time.time()

    converted = 0
    total_stars = 0
    for ebf_path in ebf_files:
        base = os.path.basename(ebf_path)
        # gx_<tic>.ebf -> field_<tic>.npz
        tic = base.replace("gx_", "").replace(".ebf", "")
        out_npz = os.path.join(out_dir, f"field_{tic}.npz")
        convert_file(ebf_path, out_npz)
        converted += 1
        if converted % 50 == 0 or converted == len(ebf_files):
            print(f"  [{converted}/{len(ebf_files)}] converted...")

    print(f"Successfully converted {converted} files in {time.time() - t0:.2f}s.")

if __name__ == "__main__":
    main()
