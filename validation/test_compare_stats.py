#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Unit and integration tests for validation/compare_stats.py.

Can be run directly via:
    python validation/test_compare_stats.py
or using pytest:
    pytest validation/test_compare_stats.py
"""

import json
import os
import sys
import tempfile
import numpy as np

# Ensure validation module is on sys.path regardless of execution location
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from compare_stats import main, compute_ks_2samp, compute_family_stats
except ImportError:
    from validation.compare_stats import main, compute_ks_2samp, compute_family_stats


def make_synthetic_file(
    filepath: str,
    n_stars: int,
    seed: int,
    shift_j: float = 0.0,
    include_pop: bool = True,
) -> None:
    """Generate a realistic synthetic catalogue .npz file."""
    rng = np.random.default_rng(seed)
    rad = rng.uniform(0.1, 5.0, n_stars)
    mact = rng.uniform(0.6, 2.5, n_stars)
    grav = rng.normal(3.8, 0.8, n_stars)
    teff = rng.normal(3.7, 0.15, n_stars)
    feh = rng.normal(-0.1, 0.2, n_stars)
    dcmc_j = rng.uniform(1.0, 7.0, n_stars) + shift_j
    dcmc_ks = dcmc_j - rng.uniform(0.1, 0.6, n_stars)
    dcmc_h = 0.5 * (dcmc_j + dcmc_ks)
    ebv = rng.uniform(0.01, 0.15, n_stars)

    data = {
        "rad": rad,
        "mact": mact,
        "grav": grav,
        "teff": teff,
        "feh": feh,
        "dcmc_j": dcmc_j,
        "dcmc_h": dcmc_h,
        "dcmc_ks": dcmc_ks,
        "ebv": ebv,
    }
    if include_pop:
        data["popid"] = rng.integers(0, 10, n_stars)

    np.savez(filepath, **data)


def test_identical_distributions():
    """Test 1: Identical distributions (drawn with different seeds, ~20000 stars each across 3 files) -> main() returns 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_a = os.path.join(tmpdir, "dir_a")
        dir_b = os.path.join(tmpdir, "dir_b")
        os.makedirs(dir_a)
        os.makedirs(dir_b)

        # 3 files with 20,000 stars each
        for i in range(1, 4):
            fname = f"field_{i:04d}_seed.npz"
            make_synthetic_file(os.path.join(dir_a, fname), n_stars=20000, seed=1000 + i)
            make_synthetic_file(os.path.join(dir_b, fname), n_stars=20000, seed=2000 + i)

        json_out = os.path.join(tmpdir, "out.json")
        md_out = os.path.join(tmpdir, "out.md")

        exit_code = main([dir_a, dir_b, "--json", json_out, "--md", md_out, "--per-file"])
        assert exit_code == 0, f"Expected return code 0 for identical distributions, got {exit_code}"

        # Verify output files exist and are valid
        assert os.path.isfile(json_out), f"JSON report file was not created: {json_out}"
        assert os.path.isfile(md_out), f"Markdown report file was not created: {md_out}"

        with open(json_out, "r", encoding="utf-8") as f:
            report = json.load(f)
            assert report["overall_passed"] is True
            assert report["exit_code"] == 0
            assert len(report["matched_files"]) == 3
            assert "tmag" in report["pooled"]["families"]
            assert "jmag" in report["pooled"]["families"]
            assert "class" in report["pooled"]["families"]
            assert "teff" in report["pooled"]["families"]
            assert "pop" in report["pooled"]["families"]
            assert "tmag_x_class" in report["pooled"]["families"]
            assert report["pooled"]["ks_test"]["passed"] is True
            assert report["per_file"] is not None
            assert len(report["per_file"]) == 3


def test_shifted_distribution():
    """Test 2: B shifted by +0.3 mag in J -> main() returns 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_a = os.path.join(tmpdir, "dir_a")
        dir_b = os.path.join(tmpdir, "dir_b")
        os.makedirs(dir_a)
        os.makedirs(dir_b)

        # 3 files with 20,000 stars each, B shifted by +0.3 mag in J
        for i in range(1, 4):
            fname = f"field_{i:04d}_seed.npz"
            make_synthetic_file(os.path.join(dir_a, fname), n_stars=20000, seed=1000 + i)
            make_synthetic_file(os.path.join(dir_b, fname), n_stars=20000, seed=2000 + i, shift_j=0.3)

        json_out = os.path.join(tmpdir, "out.json")
        exit_code = main([dir_a, dir_b, "--json", json_out])
        assert exit_code == 1, f"Expected return code 1 for shifted distribution, got {exit_code}"

        with open(json_out, "r", encoding="utf-8") as f:
            report = json.load(f)
            assert report["overall_passed"] is False
            assert report["exit_code"] == 1


def test_no_matched_files():
    """Test 3: No matched files -> main() returns 2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_a = os.path.join(tmpdir, "dir_a")
        dir_b = os.path.join(tmpdir, "dir_b")
        os.makedirs(dir_a)
        os.makedirs(dir_b)

        make_synthetic_file(os.path.join(dir_a, "field_0001_seed1.npz"), n_stars=100, seed=1)
        make_synthetic_file(os.path.join(dir_b, "field_9999_seed2.npz"), n_stars=100, seed=2)

        exit_code = main([dir_a, dir_b])
        assert exit_code == 2, f"Expected return code 2 for no matched files, got {exit_code}"


def test_usage_error_nonexistent_directory():
    """Test: Non-existent directories -> main() returns 2."""
    exit_code = main(["/nonexistent/directory/a", "/nonexistent/directory/b"])
    assert exit_code == 2, f"Expected return code 2 for non-existent directory, got {exit_code}"


def test_without_popid():
    """Test: When files omit popid, analysis succeeds with 5 families and passes for identical distributions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_a = os.path.join(tmpdir, "dir_a")
        dir_b = os.path.join(tmpdir, "dir_b")
        os.makedirs(dir_a)
        os.makedirs(dir_b)

        for i in range(1, 3):
            fname = f"field_{i:04d}.npz"
            make_synthetic_file(os.path.join(dir_a, fname), n_stars=10000, seed=100 + i, include_pop=False)
            make_synthetic_file(os.path.join(dir_b, fname), n_stars=10000, seed=200 + i, include_pop=False)

        json_out = os.path.join(tmpdir, "out_nopop.json")
        exit_code = main([dir_a, dir_b, "--json", json_out])
        assert exit_code == 0, f"Expected return code 0, got {exit_code}"

        with open(json_out, "r", encoding="utf-8") as f:
            report = json.load(f)
            assert report["overall_passed"] is True
            assert "pop" not in report["pooled"]["families"]
            assert "tmag" in report["pooled"]["families"]


if __name__ == "__main__":
    print("Running test_identical_distributions...")
    test_identical_distributions()
    print("PASS: test_identical_distributions")

    print("Running test_shifted_distribution...")
    test_shifted_distribution()
    print("PASS: test_shifted_distribution")

    print("Running test_no_matched_files...")
    test_no_matched_files()
    print("PASS: test_no_matched_files")

    print("Running test_usage_error_nonexistent_directory...")
    test_usage_error_nonexistent_directory()
    print("PASS: test_usage_error_nonexistent_directory")

    print("Running test_without_popid...")
    test_without_popid()
    print("PASS: test_without_popid")

    print("\nAll tests passed successfully!")
