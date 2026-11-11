#!/usr/bin/env bash
# Builds qes_bench for the local arch (x86_64 on P7, aarch64 on GX1).
# Requires: CUDA nvcc, libibverbs-dev (infiniband/verbs.h), libcudart/libcuda.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ARCH="$(uname -m)"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
[[ -x "$NVCC" ]] || NVCC="$(command -v nvcc)"
LIBDIR="$([[ $ARCH == x86_64 ]] && echo /usr/lib/x86_64-linux-gnu || echo /usr/lib/aarch64-linux-gnu)"
unset LD_LIBRARY_PATH || true
export LIBRARY_PATH="${LIBDIR}${LIBRARY_PATH:+:$LIBRARY_PATH}"
OUT="$ROOT/bin"
mkdir -p "$OUT"
# native asks the driver for the GPU in this container; NVCC_ARCH=compute_120/sm_121 pins it when
# there is no device to ask (a build container on the other host) or when the box is busy serving.
NVARCH="${NVCC_ARCH:-native}"
"$NVCC" -O2 -std=c++17 -arch="$NVARCH" -x cu "$ROOT/qmesh_expert_seam.cu" "$ROOT/qes_bench.cu" \
  -o "$OUT/qes_bench.$ARCH" -libverbs -lcuda -lcudart -Xlinker -rpath="$LIBDIR"
ldd "$OUT/qes_bench.$ARCH" | grep -E 'ibverbs|cuda' || true
echo BUILD_OK "$OUT/qes_bench.$ARCH"
# Shared library for the Python binding (qes.py): same sources, no bench main.
"$NVCC" -O2 -std=c++17 -arch="$NVARCH" -Xcompiler -fPIC -shared -x cu "$ROOT/qmesh_expert_seam.cu" \
  -o "$OUT/libqmesh_expert_seam.$ARCH.so" -libverbs -lcuda -lcudart -Xlinker -rpath="$LIBDIR"
echo BUILD_OK "$OUT/libqmesh_expert_seam.$ARCH.so"
