#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Compares C++ seed1 vs C++ seed2 baseline on the exact 230 validation fields,
and places it side-by-side with Rust vs C++ seed1.
Outputs:
1. Median ratio, min/max, and sigma for both pairs.
2. Complete per-stratum comparison table.
"""

import os
import sys
import csv
import json
import numpy as np

DIR_CPP1 = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val230_ref"
DIR_CPP2 = sys.argv[2] if len(sys.argv) > 2 else "/tmp/val230_cpp_seed2"
DIR_RUST = sys.argv[3] if len(sys.argv) > 3 else "/tmp/val230_rust_npz"
CATALOG_CSV = sys.argv[4] if len(sys.argv) > 4 else "validation/validation_230_fields.csv"

def get_star_count(npz_path):
    if not os.path.exists(npz_path):
        return None
    d = np.load(npz_path)
    return len(d["rad"])

def main():
    fields = []
    with open(CATALOG_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fields.append({
                "tic": row["tic"],
                "l": float(row["gal_l"]),
                "b": float(row["gal_b"]),
                "chunk": row["chunk"]
            })

    cpp_cpp_ratios = []
    rust_cpp_ratios = []

    # Stratified accumulators
    strata = {
        "b_0_5":   {"n_fields": 0, "n_cpp1": 0, "n_cpp2": 0, "n_rust": 0},
        "b_5_15":  {"n_fields": 0, "n_cpp1": 0, "n_cpp2": 0, "n_rust": 0},
        "b_15_30": {"n_fields": 0, "n_cpp1": 0, "n_cpp2": 0, "n_rust": 0},
        "b_30_60": {"n_fields": 0, "n_cpp1": 0, "n_cpp2": 0, "n_rust": 0},
        "b_60_90": {"n_fields": 0, "n_cpp1": 0, "n_cpp2": 0, "n_rust": 0},
    }

    per_field_data = []

    for item in fields:
        tic = item["tic"]
        f_cpp1 = os.path.join(DIR_CPP1, f"field_{tic}.npz")
        f_cpp2 = os.path.join(DIR_CPP2, f"field_{tic}.npz")
        f_rust = os.path.join(DIR_RUST, f"field_{tic}.npz")

        n1 = get_star_count(f_cpp1)
        n2 = get_star_count(f_cpp2)
        nr = get_star_count(f_rust)

        if n1 is None or n2 is None or nr is None:
            continue

        r_cpp_cpp = n2 / n1 if n1 > 0 else 1.0
        r_rust_cpp = nr / n1 if n1 > 0 else 1.0

        cpp_cpp_ratios.append(r_cpp_cpp)
        rust_cpp_ratios.append(r_rust_cpp)

        abs_b = abs(item["b"])
        if abs_b <= 5.0:
            s_key = "b_0_5"
        elif abs_b <= 15.0:
            s_key = "b_5_15"
        elif abs_b <= 30.0:
            s_key = "b_15_30"
        elif abs_b <= 60.0:
            s_key = "b_30_60"
        else:
            s_key = "b_60_90"

        strata[s_key]["n_fields"] += 1
        strata[s_key]["n_cpp1"] += n1
        strata[s_key]["n_cpp2"] += n2
        strata[s_key]["n_rust"] += nr

        per_field_data.append({
            "tic": tic, "l": item["l"], "b": item["b"],
            "n1": n1, "n2": n2, "nr": nr,
            "r_cpp_cpp": r_cpp_cpp, "r_rust_cpp": r_rust_cpp
        })

    cpp_cpp_ratios = np.array(cpp_cpp_ratios)
    rust_cpp_ratios = np.array(rust_cpp_ratios)

    print("================================================================================")
    print("      SIDE-BY-SIDE NOISE BASELINE COMPARISON ON THE EXACT SAME 230 FIELDS")
    print("================================================================================")
    print(f"Total fields evaluated: {len(cpp_cpp_ratios)}")
    print(f"{'Metric':<30} {'C++ vs C++ (seed1 vs seed2)':<28} {'Rust vs C++ (seed1 vs seed1)':<28}")
    print("-" * 86)
    print(f"{'Median Count Ratio':<30} {np.median(cpp_cpp_ratios):<28.4f} {np.median(rust_cpp_ratios):<28.4f}")
    print(f"{'Min / Max Count Ratio':<30} {f'{np.min(cpp_cpp_ratios):.4f} / {np.max(cpp_cpp_ratios):.4f}':<28} {f'{np.min(rust_cpp_ratios):.4f} / {np.max(rust_cpp_ratios):.4f}':<28}")
    print(f"{'Std Dev of Ratio (sigma)':<30} {np.std(cpp_cpp_ratios):<28.4f} {np.std(rust_cpp_ratios):<28.4f}")
    tot_cpp1 = np.sum([s["n_cpp1"] for s in strata.values()])
    tot_cpp2 = np.sum([s["n_cpp2"] for s in strata.values()])
    tot_rust = np.sum([s["n_rust"] for s in strata.values()])
    print(f"{'Pooled Total Stars':<30} {f'{tot_cpp2:,} / {tot_cpp1:,}':<28} {f'{tot_rust:,} / {tot_cpp1:,}':<28}")
    pooled_cpp = tot_cpp2 / tot_cpp1
    pooled_rust = tot_rust / tot_cpp1
    print(f"{'Pooled Overall Ratio':<30} {f'{pooled_cpp:.5f} ({pooled_cpp-1.0:+.3%})':<28} {f'{pooled_rust:.5f} ({pooled_rust-1.0:+.3%})':<28}")
    print("=" * 86)

    print("\nGalactic Latitude Strata Breakdown (|b|):")
    print("-" * 92)
    print(f"{'Stratum':<10} {'Fields':>6} {'N_C++(s1)':>12} {'N_C++(s2)':>12} {'N_Rust':>12} {'Rust/C++':>12} {'C++2/C++1':>12}")
    print("-" * 92)
    for s_key, s_data in strata.items():
        n_c1 = s_data["n_cpp1"]
        n_c2 = s_data["n_cpp2"]
        n_r = s_data["n_rust"]
        r_rc = n_r / n_c1 if n_c1 > 0 else 1.0
        r_cc = n_c2 / n_c1 if n_c1 > 0 else 1.0
        print(f"{s_key:<10} {s_data['n_fields']:>6d} {n_c1:>12,d} {n_c2:>12,d} {n_r:>12,d} {r_rc:>12.4f} {r_cc:>12.4f}")
    print("-" * 92)

if __name__ == "__main__":
    main()
