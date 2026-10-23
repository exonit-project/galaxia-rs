#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Verify Gate 5: Size of change against the pre-fix baseline.
Requirements:
- Compare total retained star count against pre-fix baseline (baseline: 5,392,532 stars).
- Verify 0 fields with >1% change.
- Report median difference (in ppm or %), max difference, and delta stars.
"""
import sys
import os
import glob
from pathlib import Path
import numpy as np

# Use spec_scripts read_ebf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "spec_scripts"))
import read_ebf

def main():
    if len(sys.argv) < 3:
        print("Usage: verify_gate5.py <dir_baseline> <dir_new>")
        sys.exit(1)

    dir_base = Path(sys.argv[1])
    dir_new = Path(sys.argv[2])

    files_base = sorted(glob.glob(f"{dir_base}/gx_*.ebf"))
    files_new = sorted(glob.glob(f"{dir_new}/gx_*.ebf"))

    print(f"Baseline dir ({dir_base}): {len(files_base)} files")
    print(f"New dir ({dir_new}): {len(files_new)} files")

    common_names = sorted(set(os.path.basename(f) for f in files_base).intersection(
        set(os.path.basename(f) for f in files_new)
    ))
    print(f"Common fields to compare: {len(common_names)}")

    tot_base = 0
    tot_new = 0
    rel_diffs = []
    diff_gt_1pct = []
    max_rel_diff = 0.0
    max_field = ""

    for fname in common_names:
        fb = dir_base / fname
        fn = dir_new / fname

        nb = len(read_ebf.read_ebf_item(str(fb), "/rad"))
        nn = len(read_ebf.read_ebf_item(str(fn), "/rad"))

        tot_base += nb
        tot_new += nn

        rel = abs(nn - nb) / nb if nb > 0 else 0.0
        rel_diffs.append(rel)
        if rel > max_rel_diff:
            max_rel_diff = rel
            max_field = f"{fname} ({nb} -> {nn}, delta={nn-nb})"

        if rel > 0.01:
            diff_gt_1pct.append((fname, nb, nn, rel))

    median_diff = float(np.median(rel_diffs))
    delta_stars = tot_new - tot_base
    pct_change = (delta_stars / tot_base * 100.0) if tot_base > 0 else 0.0

    print("\n========================================================")
    print("                GATE 5 EVALUATION RESULTS               ")
    print("========================================================")
    print(f"Total fields evaluated:     {len(common_names)} / 230")
    print(f"Baseline stars (pre-fix baseline):   {tot_base}")
    print(f"New stars:                  {tot_new}")
    print(f"Delta stars:                {delta_stars:+d} ({pct_change:+.6f}%)")
    print(f"Median relative difference: {median_diff*1e6:.1f} ppm ({median_diff*100:.5f}%)")
    print(f"Max relative difference:    {max_rel_diff*100:.4f}% on {max_field}")
    print(f"Fields with > 1% change:    {len(diff_gt_1pct)}")
    print(f"Gate 5 status:              {'PASS' if len(diff_gt_1pct) == 0 and abs(pct_change) < 0.1 else 'FAIL'}")
    print("========================================================\n")

if __name__ == "__main__":
    main()
