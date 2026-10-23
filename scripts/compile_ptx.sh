#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
set -euo pipefail

# Compilation script for Galaxia CUDA kernels 
# EXPERIMENTAL CUDA backend only. Uses $NVCC if set, otherwise `nvcc` from PATH.
# $GALAXIA_CUDA_ARCH selects the target (default compute_89).

NVCC="${NVCC:-nvcc}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$ROOT_DIR/target/ptx"
PTX_OUT="$ROOT_DIR/target/ptx/spawn.ptx"
SRC_CU="$ROOT_DIR/src/backend/cuda/kernels/spawn.cu"

echo "Compiling $SRC_CU to $PTX_OUT using $NVCC..."
"$NVCC" -ptx --gpu-architecture="${GALAXIA_CUDA_ARCH:-compute_89}" -o "$PTX_OUT" "$SRC_CU"

# Patch version for driver compatibility with some CUDA 13.x drivers
if grep -q '\.version 9\.3' "$PTX_OUT"; then
    sed -i 's/\.version 9\.3/\.version 9\.2/g' "$PTX_OUT"
    echo "Patched PTX header to .version 9.2 for driver compatibility with some CUDA 13.x drivers."
fi

echo "PTX compilation complete: $(ls -lh "$PTX_OUT")"
