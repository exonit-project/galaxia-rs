#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Build the C++-linked parity harness against the ORIGINAL Galaxia 0.7.2 objects, regenerate the
# reference fixtures (tests/fixtures/cpp_boundaries.json) and run the Rust parity tests against them.
#
# Usage:
#   GALAXIA_CPP_SRC=/path/to/galaxia-0.7.2/src \
#   GALAXIA_DATA_DIR=/path/to/GalaxiaData \
#   scripts/run_cpp_parity_tests.sh
#
# or:  scripts/run_cpp_parity_tests.sh <galaxia-0.7.2/src dir> <GalaxiaData dir>
#
# GALAXIA_CPP_SRC must point to the `src/` directory of a COMPILED Galaxia 0.7.2 source tree
# (i.e. the directory that contains Isochrone.h and the built *.o object files).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

CPP_SRC_DIR="${1:-${GALAXIA_CPP_SRC:-}}"
DATA_DIR="${2:-${GALAXIA_DATA_DIR:-}}"

if [[ -z "${CPP_SRC_DIR}" || -z "${DATA_DIR}" ]]; then
    echo "Usage: GALAXIA_CPP_SRC=<compiled galaxia-0.7.2/src> GALAXIA_DATA_DIR=<GalaxiaData> $0" >&2
    echo "   or: $0 <galaxia-0.7.2/src dir> <GalaxiaData dir>" >&2
    exit 2
fi
if [[ ! -d "${CPP_SRC_DIR}" ]]; then
    echo "ERROR: original C++ Galaxia sources not found at ${CPP_SRC_DIR}" >&2
    exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: GalaxiaData directory not found at ${DATA_DIR}" >&2
    exit 1
fi

mkdir -p target tests/fixtures

echo "=== [1/4] Compiling C++ parity test harness against original 0.7.2 objects ==="
g++ -std=c++11 -O2 -I"${CPP_SRC_DIR}" \
    tests/cpp/check_cpp_boundaries.cpp \
    "${CPP_SRC_DIR}/IsochroneDB.o" \
    "${CPP_SRC_DIR}/IsochroneBase.o" \
    "${CPP_SRC_DIR}/Isochrone.o" \
    "${CPP_SRC_DIR}/TableInterpolator.o" \
    "${CPP_SRC_DIR}/Interp.o" \
    "${CPP_SRC_DIR}/Sampler.o" \
    "${CPP_SRC_DIR}/Functions.o" \
    "${CPP_SRC_DIR}/sutils.o" \
    "${CPP_SRC_DIR}/ebf.o" \
    "${CPP_SRC_DIR}/ebftable.o" \
    "${CPP_SRC_DIR}/Population.o" \
    "${CPP_SRC_DIR}/GInterpolator.o" \
    "${CPP_SRC_DIR}/Cot.o" \
    -o target/check_cpp_boundaries

echo "=== [2/4] Generating C++ reference test vectors ==="
GALAXIA_DATA_DIR="${DATA_DIR}" target/check_cpp_boundaries tests/fixtures/cpp_boundaries.json

echo "=== [3/4] Running Rust C++ parity verification tests ==="
GALAXIA_DATA_DIR="${DATA_DIR}" cargo test --release --test test_cpp_parity -- --nocapture

echo "=== [4/4] Running parity-fix and failure-mode tests ==="
GALAXIA_DATA_DIR="${DATA_DIR}" cargo test --release --test test_parity_fixes -- --nocapture

echo "=== All C++ parity tests passed ==="
