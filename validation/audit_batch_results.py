#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Audits the 230-field validation JSON stats against the criteria in VALIDATION.md.
Analyzes:
1. Per-field count ratio distribution.
2. Galactic latitude strata breakdown.
3. Top 5 worst fields (by min p-value or max chi2) with physical explanation.
"""

import os
import sys
import json
import csv
import numpy as np

STATS_JSON = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val230_stats.json"
CATALOG_CSV = sys.argv[2] if len(sys.argv) > 2 else "validation/validation_230_fields.csv"

def main():
    with open(STATS_JSON, "r") as f:
        data = json.load(f)

    # Load field catalog metadata
    field_meta = {}
    with open(CATALOG_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            field_meta[f"field_{row['tic']}.npz"] = {
                "tic": row["tic"],
                "ra": float(row["ra"]),
                "dec": float(row["dec"]),
                "gal_l": float(row["gal_l"]),
                "gal_b": float(row["gal_b"]),
                "chunk": row["chunk"],
            }

    per_file = data.get("per_file", {})
    count_ratios = []
    field_pvalues = []
    field_records = []

    # Stratified collectors
    # Strata 1: |b| in [0, 5]
    # Strata 2: |b| in (5, 15]
    # Strata 3: |b| in (15, 30]
    # Strata 4: |b| in (30, 60]
    # Strata 5: |b| in (60, 90]
    strata_counts = {
        "b_0_5": {"nA": 0, "nB": 0, "n_fields": 0},
        "b_5_15": {"nA": 0, "nB": 0, "n_fields": 0},
        "b_15_30": {"nA": 0, "nB": 0, "n_fields": 0},
        "b_30_60": {"nA": 0, "nB": 0, "n_fields": 0},
        "b_60_90": {"nA": 0, "nB": 0, "n_fields": 0},
    }

    for fname, fstat in per_file.items():
        nA = fstat["stars_a"]
        nB = fstat["stars_b"]
        ratio = nB / nA if nA > 0 else 1.0
        count_ratios.append(ratio)

        meta = field_meta.get(fname, {"tic": fname, "gal_l": 0.0, "gal_b": 0.0, "chunk": "unknown"})
        abs_b = abs(meta["gal_b"])
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

        strata_counts[s_key]["nA"] += nA
        strata_counts[s_key]["nB"] += nB
        strata_counts[s_key]["n_fields"] += 1

        # Calculate minimum p-value across families for this field
        p_vals = [fam["p_value"] for fam in fstat["families"].values()]
        min_p = min(p_vals) if p_vals else 1.0
        max_chi2 = max(fam["chi2"] for fam in fstat["families"].values()) if p_vals else 0.0
        field_pvalues.append(min_p)

        field_records.append({
            "fname": fname,
            "tic": meta["tic"],
            "l": meta["gal_l"],
            "b": meta["gal_b"],
            "chunk": meta["chunk"],
            "nA": nA,
            "nB": nB,
            "ratio": ratio,
            "min_p": min_p,
            "max_chi2": max_chi2,
            "families": fstat["families"]
        })

    print("================================================================================")
    print("                    230-FIELD STATISTICAL AUDIT REPORT")
    print("================================================================================")
    print(f"Total matched fields:     {len(count_ratios)}")
    print(f"Pooled Stars (Reference): {data['total_stars_a']:,}")
    print(f"Pooled Stars (Candidate): {data['total_stars_b']:,}")
    pooled_ratio = data['total_stars_b'] / data['total_stars_a']
    print(f"Pooled Count Ratio:       {pooled_ratio:.5f} ({pooled_ratio - 1.0:+.3%})")
    print(f"Median Field Count Ratio: {np.median(count_ratios):.4f}")
    print(f"Min / Max Count Ratio:    {np.min(count_ratios):.4f} / {np.max(count_ratios):.4f}")
    print(f"Std Dev of Count Ratio:   {np.std(count_ratios):.4f}")
    print("--------------------------------------------------------------------------------")

    print("\nGalactic Latitude Strata Breakdown:")
    print("--------------------------------------------------------------------------------")
    print(f"{'Stratum':<12} {'Fields':>8} {'Ref Stars':>14} {'Cand Stars':>14} {'Ratio':>10}")
    print("--------------------------------------------------------------------------------")
    for s_key, s_data in strata_counts.items():
        s_ratio = s_data["nB"] / s_data["nA"] if s_data["nA"] > 0 else 1.0
        print(f"{s_key:<12} {s_data['n_fields']:>8d} {s_data['nA']:>14,d} {s_data['nB']:>14,d} {s_ratio:>10.4f}")

    print("\n--------------------------------------------------------------------------------")
    print("Top 5 Worst Fields (by minimum p-value across families):")
    print("--------------------------------------------------------------------------------")
    field_records.sort(key=lambda x: x["min_p"])
    for rank, f in enumerate(field_records[:5], 1):
        print(f"Rank {rank}: TIC {f['tic']} (l={f['l']:.2f}°, b={f['b']:.2f}°, Chunk {f['chunk']})")
        print(f"  Stars: N_ref = {f['nA']}, N_cand = {f['nB']} (Ratio: {f['ratio']:.4f})")
        print(f"  Min p-value = {f['min_p']:.4e}, Max chi2 = {f['max_chi2']:.2f}")
        for fam_name, fam_stat in f["families"].items():
            if fam_stat["p_value"] < 0.05:
                print(f"    - Family '{fam_name}': chi2={fam_stat['chi2']:.2f}, dof={fam_stat['dof']}, p={fam_stat['p_value']:.4e}, max|z|={fam_stat['max_abs_z']:.2f}")
        print()

if __name__ == "__main__":
    main()
