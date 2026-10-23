#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Computes per-field p-value distributions (tmag and KS on T <= 21.0) across all 230 fields
and evaluates a Kolmogorov-Smirnov test of those p-values against Uniform(0, 1).
"""

import os
import sys
import glob
import json
import numpy as np
from scipy.stats import ks_2samp, kstest

def compute_tmag(data):
    rad = np.asarray(data["rad"], dtype=float)
    dcmc_j = np.asarray(data["dcmc_j"], dtype=float)
    dcmc_ks = np.asarray(data["dcmc_ks"], dtype=float)
    ebv = np.asarray(data["ebv"], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        dm = 5.0 * np.log10(rad * 1000.0) - 5.0
        J = dcmc_j + dm + 0.282 * 3.1 * ebv
        Ks = dcmc_ks + dm + 0.112 * 3.1 * ebv
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

        valid = np.isfinite(T) & (T <= 21.0)
        return T[valid]

def main():
    dir_ref = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val230_ref"
    dir_cand = sys.argv[2] if len(sys.argv) > 2 else "/tmp/val230_rust_npz"
    stats_json_path = sys.argv[3] if len(sys.argv) > 3 else "/tmp/val230_stats.json"

    with open(stats_json_path, "r") as f:
        stats_data = json.load(f)

    per_file = stats_data["per_file"]
    matched_files = sorted(per_file.keys())

    tmag_pvalues = []
    ks_pvalues = []
    ks_d_stats = []

    print(f"Evaluating per-field statistics for {len(matched_files)} fields...")

    for fname in matched_files:
        # tmag chi2 p-value from compare_stats
        tmag_p = per_file[fname]["families"]["tmag"]["p_value"]
        tmag_pvalues.append(tmag_p)

        # compute per-field two-sample KS on T <= 21.0
        path_a = os.path.join(dir_ref, fname)
        path_b = os.path.join(dir_cand, fname)

        data_a = np.load(path_a)
        data_b = np.load(path_b)

        T_a = compute_tmag(data_a)
        T_b = compute_tmag(data_b)

        if len(T_a) >= 5 and len(T_b) >= 5:
            res = ks_2samp(T_a, T_b)
            ks_d_stats.append(res.statistic)
            ks_pvalues.append(res.pvalue)
        else:
            ks_pvalues.append(1.0)
            ks_d_stats.append(0.0)

    tmag_pvalues = np.array(tmag_pvalues)
    ks_pvalues = np.array(ks_pvalues)

    # Test against uniform U(0, 1)
    res_tmag_unif = kstest(tmag_pvalues, "uniform")
    res_ks_unif = kstest(ks_pvalues, "uniform")

    print("\n================================================================================")
    print("                    PER-FIELD P-VALUE DISTRIBUTIONS & UNIFORMITY")
    print("================================================================================")
    print(f"Total fields evaluated: {len(matched_files)}")

    print("\n1. Per-Field Tmag Chi-Squared P-Values:")
    print(f"  - Mean / Median:   {np.mean(tmag_pvalues):.4f} / {np.median(tmag_pvalues):.4f}")
    print(f"  - Min / Max:       {np.min(tmag_pvalues):.4e} / {np.max(tmag_pvalues):.4f}")
    print(f"  - Percentiles (10%, 25%, 50%, 75%, 90%):")
    pcts = [10, 25, 50, 75, 90]
    p_vals_pct = np.percentile(tmag_pvalues, pcts)
    for p, val in zip(pcts, p_vals_pct):
        print(f"      {p}%: {val:.4f}")
    print(f"  - KS Test against Uniform(0, 1): D = {res_tmag_unif.statistic:.4f}, p-value = {res_tmag_unif.pvalue:.4f}")

    print("\n2. Per-Field Two-Sample Kolmogorov-Smirnov P-Values (T <= 21.0):")
    print(f"  - Mean / Median:   {np.mean(ks_pvalues):.4f} / {np.median(ks_pvalues):.4f}")
    print(f"  - Min / Max:       {np.min(ks_pvalues):.4e} / {np.max(ks_pvalues):.4f}")
    print(f"  - Percentiles (10%, 25%, 50%, 75%, 90%):")
    ks_vals_pct = np.percentile(ks_pvalues, pcts)
    for p, val in zip(pcts, ks_vals_pct):
        print(f"      {p}%: {val:.4f}")
    print(f"  - KS Test against Uniform(0, 1): D = {res_ks_unif.statistic:.4f}, p-value = {res_ks_unif.pvalue:.4f}")

    # Output json report
    out_dict = {
        "n_fields": len(matched_files),
        "tmag_uniform_ks": {"d_stat": float(res_tmag_unif.statistic), "p_value": float(res_tmag_unif.pvalue)},
        "ks_uniform_ks": {"d_stat": float(res_ks_unif.statistic), "p_value": float(res_ks_unif.pvalue)},
        "tmag_pvalues": tmag_pvalues.tolist(),
        "ks_pvalues": ks_pvalues.tolist(),
    }
    with open("/tmp/val230_pvalues_uniform.json", "w") as f:
        json.dump(out_dict, f, indent=2)

if __name__ == "__main__":
    main()
