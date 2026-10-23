#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Statistical comparison between two star-catalogue generators (.npz format).

Compares reference directory DIR_A with candidate directory DIR_B on matched
sky fields. Computes binned statistics (chi2, z-scores) for multiple families
and a two-sample Kolmogorov-Smirnov test on pooled T-magnitude (T <= 21.0).

Exit codes:
  0: PASS (all criteria satisfied)
  1: FAIL (statistical deviation detected)
  2: Usage error or no matched files
"""

import argparse
import json
import math
import os
import sys
import numpy as np

# Try importing scipy for exact chi2 survival function; fallback to Wilson-Hilferty.
try:
    from scipy.stats import chi2 as scipy_chi2
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Binnings / families configuration
TMAG_EDGES = np.arange(6.0, 24.0, 1.0)
JMAG_EDGES = np.arange(6.0, 24.0, 1.0)
TEFF_EDGES = np.array([-np.inf, 3.5, 3.6, 3.7, 3.8, 4.0, np.inf])
REQUIRED_KEYS = ["rad", "mact", "grav", "teff", "feh", "dcmc_j", "dcmc_h", "dcmc_ks", "ebv"]


def chi2_pvalue(chi2: float, dof: int) -> float:
    """
    Compute upper-tail p-value for chi2 with dof degrees of freedom.
    Uses scipy.stats.chi2.sf if scipy is available, else Wilson-Hilferty normal approximation.
    """
    if dof <= 0:
        return 1.0
    if chi2 <= 0.0:
        return 1.0
    if HAS_SCIPY:
        return float(scipy_chi2.sf(chi2, dof))

    s = 2.0 / (9.0 * dof)
    z_wh = ((chi2 / dof) ** (1.0 / 3.0) - (1.0 - s)) / math.sqrt(s)
    p = 0.5 * math.erfc(z_wh / math.sqrt(2.0))
    return float(max(0.0, min(1.0, p)))


def compute_ks_2samp(data1: np.ndarray, data2: np.ndarray) -> tuple[float, float, bool]:
    """
    Two-sample Kolmogorov-Smirnov statistic D and asymptotic p-value.
    ne = (nA * nB) / (nA + nB)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * D
    p_ks = sum_{k=1..100} 2.0 * (-1.0)**(k - 1) * math.exp(-2.0 * k * k * lam * lam)
    """
    n1 = len(data1)
    n2 = len(data2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0, True

    s1 = np.sort(data1)
    s2 = np.sort(data2)
    data_all = np.concatenate([s1, s2])

    cdf1 = np.searchsorted(s1, data_all, side="right") / n1
    cdf2 = np.searchsorted(s2, data_all, side="right") / n2
    d = float(np.max(np.abs(cdf1 - cdf2)))

    if d <= 0.0:
        return 0.0, 1.0, True

    ne = (n1 * n2) / (n1 + n2)
    sqrt_ne = math.sqrt(ne)
    lam = (sqrt_ne + 0.12 + 0.11 / sqrt_ne) * d

    if lam * lam * 2.0 > 700.0:
        p_ks = 0.0
    else:
        p_ks = 0.0
        for k in range(1, 101):
            term = 2.0 * ((-1.0) ** (k - 1)) * math.exp(-2.0 * k * k * lam * lam)
            p_ks += term
            if abs(term) < 1e-15:
                break
        p_ks = max(0.0, min(1.0, p_ks))

    passed = (p_ks >= 0.001)
    return d, float(p_ks), passed


def load_catalog_file(filepath: str) -> dict:
    """
    Load a catalogue .npz file, derive J, Ks, color c, and Tmag (Stassun 2018),
    and filter down to stars where np.isfinite(T).
    """
    data = np.load(filepath)
    for k in REQUIRED_KEYS:
        if k not in data:
            raise KeyError(f"Missing required key '{k}' in {filepath}")

    rad = np.asarray(data["rad"], dtype=float)
    grav = np.asarray(data["grav"], dtype=float)
    teff = np.asarray(data["teff"], dtype=float)
    dcmc_j = np.asarray(data["dcmc_j"], dtype=float)
    dcmc_ks = np.asarray(data["dcmc_ks"], dtype=float)
    ebv = np.asarray(data["ebv"], dtype=float)
    popid = np.asarray(data["popid"]) if "popid" in data else None

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

        valid = np.isfinite(T)

    return {
        "T": T[valid],
        "J": J[valid],
        "grav": grav[valid],
        "teff": teff[valid],
        "popid": popid[valid] if popid is not None else None,
        "n_stars": int(np.sum(valid)),
    }


def get_bin_labels(family_name: str) -> list[str]:
    """Return descriptive labels for bins of a given family."""
    if family_name == "tmag":
        return [f"[{e:.0f},{e+1:.0f})" if e < 22.0 else f"[{e:.0f},{e+1:.0f}]" for e in TMAG_EDGES[:-1]]
    elif family_name == "jmag":
        return [f"[{e:.0f},{e+1:.0f})" if e < 22.0 else f"[{e:.0f},{e+1:.0f}]" for e in JMAG_EDGES[:-1]]
    elif family_name == "class":
        return ["dwarf(grav>=3.5)", "giant(grav<3.5)"]
    elif family_name == "teff":
        labels = []
        for i in range(len(TEFF_EDGES) - 1):
            low = "-inf" if np.isneginf(TEFF_EDGES[i]) else f"{TEFF_EDGES[i]:.1f}"
            high = "+inf" if np.isposinf(TEFF_EDGES[i+1]) else f"{TEFF_EDGES[i+1]:.1f}"
            labels.append(f"[{low},{high})")
        return labels
    elif family_name == "pop":
        return [f"pop_{k}" for k in range(10)]
    elif family_name == "tmag_x_class":
        dwarf_labels = [f"dwarf_[{e:.0f},{e+1:.0f})" for e in TMAG_EDGES[:-1]]
        giant_labels = [f"giant_[{e:.0f},{e+1:.0f})" for e in TMAG_EDGES[:-1]]
        return dwarf_labels + giant_labels
    return []


def compute_family_counts(cat: dict, include_pop: bool = True) -> dict[str, np.ndarray]:
    """Compute histogram counts for each family from processed catalog data."""
    T = cat["T"]
    J = cat["J"]
    grav = cat["grav"]
    teff = cat["teff"]
    popid = cat.get("popid")

    counts = {}
    counts["tmag"], _ = np.histogram(T, bins=TMAG_EDGES)
    counts["jmag"], _ = np.histogram(J, bins=JMAG_EDGES)
    counts["class"] = np.array([np.sum(grav >= 3.5), np.sum(grav < 3.5)], dtype=int)
    counts["teff"], _ = np.histogram(teff, bins=TEFF_EDGES)

    if include_pop and popid is not None:
        counts["pop"] = np.array([np.sum(popid == k) for k in range(10)], dtype=int)

    dwarf_mask = grav >= 3.5
    giant_mask = grav < 3.5
    c_dwarf, _ = np.histogram(T[dwarf_mask], bins=TMAG_EDGES)
    c_giant, _ = np.histogram(T[giant_mask], bins=TMAG_EDGES)
    counts["tmag_x_class"] = np.concatenate([c_dwarf, c_giant])

    return counts


def compute_family_stats(nA: np.ndarray, nB: np.ndarray, bin_labels: list[str] | None = None) -> dict:
    """
    Compute chi2, dof, p-value, and z-scores for a bin count comparison.
    z = (nB - nA) / sqrt(nA + nB)
    chi2 = sum(z**2) over bins with nA + nB >= 10
    """
    nA = np.asarray(nA, dtype=float)
    nB = np.asarray(nB, dtype=float)
    tot = nA + nB

    z = np.zeros_like(nA, dtype=float)
    nz = tot > 0
    z[nz] = (nB[nz] - nA[nz]) / np.sqrt(tot[nz])

    valid = tot >= 10
    dof = int(np.sum(valid))
    if dof > 0:
        chi2 = float(np.sum(z[valid] ** 2))
        max_abs_z = float(np.max(np.abs(z[valid])))
    else:
        chi2 = 0.0
        max_abs_z = 0.0

    p = chi2_pvalue(chi2, dof)
    passed = bool((p >= 0.001) and (max_abs_z <= 4.5))

    bins_detail = []
    if bin_labels is not None:
        for idx in range(len(nA)):
            bins_detail.append({
                "label": bin_labels[idx] if idx < len(bin_labels) else str(idx),
                "nA": int(nA[idx]),
                "nB": int(nB[idx]),
                "z": float(z[idx]),
                "included": bool(valid[idx]),
            })

    return {
        "chi2": chi2,
        "dof": dof,
        "p_value": p,
        "max_abs_z": max_abs_z,
        "passed": passed,
        "bins": bins_detail,
    }


def compare_directories(dir_a: str, dir_b: str, compute_per_file: bool = False) -> dict:
    """
    Perform statistical comparison between directory A and directory B.
    Returns dictionary with all metrics, statistics, and pass/fail statuses.
    """
    files_a = {f for f in os.listdir(dir_a) if f.endswith(".npz") and os.path.isfile(os.path.join(dir_a, f))}
    files_b = {f for f in os.listdir(dir_b) if f.endswith(".npz") and os.path.isfile(os.path.join(dir_b, f))}
    matched_files = sorted(files_a & files_b)

    if not matched_files:
        raise ValueError("No matched .npz files found between directories.")

    # Check whether popid is present in both catalogues for all files
    all_have_pop = True
    for f in matched_files:
        data_a = np.load(os.path.join(dir_a, f))
        data_b = np.load(os.path.join(dir_b, f))
        if "popid" not in data_a or "popid" not in data_b:
            all_have_pop = False
            break

    pooled_counts_a: dict[str, np.ndarray] = {}
    pooled_counts_b: dict[str, np.ndarray] = {}
    pooled_T_a = []
    pooled_T_b = []
    total_stars_a = 0
    total_stars_b = 0

    per_file_results = {}

    for fname in matched_files:
        path_a = os.path.join(dir_a, fname)
        path_b = os.path.join(dir_b, fname)

        cat_a = load_catalog_file(path_a)
        cat_b = load_catalog_file(path_b)

        total_stars_a += cat_a["n_stars"]
        total_stars_b += cat_b["n_stars"]

        # Keep pooled T <= 21.0 for KS test
        t_a_filt = cat_a["T"][cat_a["T"] <= 21.0]
        t_b_filt = cat_b["T"][cat_b["T"] <= 21.0]
        pooled_T_a.append(t_a_filt)
        pooled_T_b.append(t_b_filt)

        counts_a = compute_family_counts(cat_a, include_pop=all_have_pop)
        counts_b = compute_family_counts(cat_b, include_pop=all_have_pop)

        for fam in counts_a:
            if fam not in pooled_counts_a:
                pooled_counts_a[fam] = counts_a[fam].copy()
                pooled_counts_b[fam] = counts_b[fam].copy()
            else:
                pooled_counts_a[fam] += counts_a[fam]
                pooled_counts_b[fam] += counts_b[fam]

        if compute_per_file:
            file_fams = {}
            for fam in counts_a:
                labels = get_bin_labels(fam)
                stat = compute_family_stats(counts_a[fam], counts_b[fam], bin_labels=labels)
                file_fams[fam] = stat
            per_file_results[fname] = {
                "stars_a": cat_a["n_stars"],
                "stars_b": cat_b["n_stars"],
                "families": file_fams,
            }

    # Compute pooled family statistics
    pooled_family_stats = {}
    for fam in pooled_counts_a:
        labels = get_bin_labels(fam)
        stat = compute_family_stats(pooled_counts_a[fam], pooled_counts_b[fam], bin_labels=labels)
        pooled_family_stats[fam] = stat

    # Compute Kolmogorov-Smirnov test on pooled T <= 21.0
    all_T_a_ks = np.concatenate(pooled_T_a) if pooled_T_a else np.array([], dtype=float)
    all_T_b_ks = np.concatenate(pooled_T_b) if pooled_T_b else np.array([], dtype=float)
    ks_d, ks_p, ks_passed = compute_ks_2samp(all_T_a_ks, all_T_b_ks)

    # PASS criterion: every family has p >= 0.001 AND no bin with nA+nB >= 10 has |z| > 4.5 AND KS p >= 0.001
    families_passed = all(stat["passed"] for stat in pooled_family_stats.values())
    overall_passed = families_passed and ks_passed
    exit_code = 0 if overall_passed else 1

    return {
        "overall_passed": overall_passed,
        "exit_code": exit_code,
        "dir_a": dir_a,
        "dir_b": dir_b,
        "matched_files": matched_files,
        "total_stars_a": total_stars_a,
        "total_stars_b": total_stars_b,
        "pooled": {
            "families": pooled_family_stats,
            "ks_test": {
                "nA": len(all_T_a_ks),
                "nB": len(all_T_b_ks),
                "d_stat": ks_d,
                "p_value": ks_p,
                "passed": ks_passed,
            },
        },
        "per_file": per_file_results if compute_per_file else None,
    }


def format_summary_table(results: dict) -> str:
    """Format a clean textual summary table of the comparison."""
    lines = []
    lines.append("=" * 80)
    lines.append("                    STAR-CATALOGUE STATISTICAL COMPARISON")
    lines.append("=" * 80)
    lines.append(f"Reference Dir (A): {results['dir_a']}")
    lines.append(f"Candidate Dir (B): {results['dir_b']}")
    lines.append(f"Matched Files:     {len(results['matched_files'])}")
    lines.append(f"Pooled Stars:       nA = {results['total_stars_a']}, nB = {results['total_stars_b']}")
    lines.append("-" * 80)
    lines.append(f"{'Family':<16} {'chi2':>9} {'dof':>6} {'p-value':>12} {'max|z|':>10}   {'Status':<6}")
    lines.append("-" * 80)

    for fam, stat in results["pooled"]["families"].items():
        st = "PASS" if stat["passed"] else "FAIL"
        lines.append(
            f"{fam:<16} {stat['chi2']:>9.2f} {stat['dof']:>6d} {stat['p_value']:>12.4e} {stat['max_abs_z']:>10.2f}   {st:<6}"
        )

    lines.append("-" * 80)
    ks = results["pooled"]["ks_test"]
    ks_st = "PASS" if ks["passed"] else "FAIL"
    lines.append(
        f"Two-Sample Kolmogorov-Smirnov Test (pooled T <= 21.0):\n"
        f"  nA = {ks['nA']}, nB = {ks['nB']}, D = {ks['d_stat']:.5f}, p-value = {ks['p_value']:.4e} -> {ks_st}"
    )

    if results.get("per_file"):
        lines.append("-" * 80)
        lines.append("Per-File Statistics:")
        for fname, f_res in results["per_file"].items():
            lines.append(f"  File: {fname} (nA={f_res['stars_a']}, nB={f_res['stars_b']})")
            for fam, stat in f_res["families"].items():
                st = "PASS" if stat["passed"] else "FAIL"
                lines.append(
                    f"    {fam:<14} chi2={stat['chi2']:>7.2f} dof={stat['dof']:>3d} "
                    f"p={stat['p_value']:>10.4e} max|z|={stat['max_abs_z']:>6.2f} [{st}]"
                )

    lines.append("=" * 80)
    overall_st = "PASS" if results["overall_passed"] else "FAIL"
    lines.append(f"OVERALL RESULT: {overall_st} (Exit code: {results['exit_code']})")
    lines.append("=" * 80)
    return "\n".join(lines)


def format_markdown_report(results: dict) -> str:
    """Format Markdown report with statistical tables."""
    lines = []
    overall_st = "PASS" if results["overall_passed"] else "FAIL"
    lines.append("# Star-Catalogue Statistical Comparison\n")
    lines.append(f"- **Reference (A)**: `{results['dir_a']}`")
    lines.append(f"- **Candidate (B)**: `{results['dir_b']}`")
    lines.append(f"- **Matched Files**: {len(results['matched_files'])}")
    lines.append(f"- **Pooled Stars**: $N_A = {results['total_stars_a']}$, $N_B = {results['total_stars_b']}$")
    lines.append(f"- **Overall Result**: **{overall_st}**\n")

    lines.append("## Pooled Family Statistics\n")
    lines.append("| Family | $\\chi^2$ | dof | p-value | max $|z|$ | Status |")
    lines.append("|---|---|---|---|---|---|")
    for fam, stat in results["pooled"]["families"].items():
        st = "**PASS**" if stat["passed"] else "**FAIL**"
        lines.append(f"| `{fam}` | {stat['chi2']:.2f} | {stat['dof']} | {stat['p_value']:.4e} | {stat['max_abs_z']:.2f} | {st} |")

    lines.append("\n## Kolmogorov-Smirnov Test (pooled $T \\le 21.0$)\n")
    ks = results["pooled"]["ks_test"]
    ks_st = "**PASS**" if ks["passed"] else "**FAIL**"
    lines.append(f"- Sample sizes: $N_A = {ks['nA']}$, $N_B = {ks['nB']}$")
    lines.append(f"- $D$ statistic: `{ks['d_stat']:.5f}`")
    lines.append(f"- p-value: `{ks['p_value']:.4e}`")
    lines.append(f"- Status: {ks_st}\n")

    if results.get("per_file"):
        lines.append("## Per-File Statistics\n")
        for fname, f_res in results["per_file"].items():
            lines.append(f"### `{fname}` ($N_A={f_res['stars_a']}$, $N_B={f_res['stars_b']}$)\n")
            lines.append("| Family | $\\chi^2$ | dof | p-value | max $|z|$ | Status |")
            lines.append("|---|---|---|---|---|---|")
            for fam, stat in f_res["families"].items():
                st = "**PASS**" if stat["passed"] else "**FAIL**"
                lines.append(f"| `{fam}` | {stat['chi2']:.2f} | {stat['dof']} | {stat['p_value']:.4e} | {stat['max_abs_z']:.2f} | {st} |")
            lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on PASS, 1 on FAIL, 2 on usage error / no matched files."""
    if argv is None:
        argv = sys.argv[1:]
    elif isinstance(argv, list) and len(argv) > 0 and argv[0].endswith(".py"):
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description="Statistically compare two star-catalogue generators (.npz files)."
    )
    parser.add_argument("dir_a", metavar="DIR_A", help="Reference directory containing .npz files")
    parser.add_argument("dir_b", metavar="DIR_B", help="Candidate directory containing .npz files")
    parser.add_argument("--json", dest="json_out", default=None, help="Output JSON file path")
    parser.add_argument("--md", dest="md_out", default=None, help="Output Markdown file path")
    parser.add_argument("--per-file", dest="per_file", action="store_true", help="Report per-file statistics")

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) and e.code != 0 else 2
    except Exception as e:
        print(f"Usage error: {e}", file=sys.stderr)
        return 2

    if not os.path.isdir(args.dir_a):
        print(f"Error: DIR_A '{args.dir_a}' is not a directory or does not exist.", file=sys.stderr)
        return 2
    if not os.path.isdir(args.dir_b):
        print(f"Error: DIR_B '{args.dir_b}' is not a directory or does not exist.", file=sys.stderr)
        return 2

    try:
        results = compare_directories(args.dir_a, args.dir_b, compute_per_file=args.per_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    print(format_summary_table(results))

    if args.json_out:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write JSON output to '{args.json_out}': {e}", file=sys.stderr)

    if args.md_out:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.md_out)), exist_ok=True)
            with open(args.md_out, "w", encoding="utf-8") as f:
                f.write(format_markdown_report(results))
        except Exception as e:
            print(f"Warning: Failed to write Markdown output to '{args.md_out}': {e}", file=sys.stderr)

    return results["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
