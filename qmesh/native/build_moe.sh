#!/usr/bin/env bash
# Builds libqmesh_moe_cpp.<arch>.so: the C compute hook (qmesh_moe_cpp.cpp) plus the alignment and
# conversion kernels (qmesh_moe_kernels.cu).
#
# This one links libtorch, on purpose. The MoE math is the vendored cuda-exl3 kernels, which the
# Python path called as `cuda_exl3_C::exl3_moe_*`; linking and calling them through the dispatcher
# means the two paths share one implementation and one bug fix. No rpath is needed for the same
# reason: the loader only has to resolve these symbols once Python has imported torch, and
# qmesh/moe_cpp.py imports torch before dlopen. The rpath below is belt and braces for a bench
# that loads the .so from a bare python3.
#
# Requires: nvcc, g++, and a python3 that can `import torch` (the same one the server runs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ARCH="$(uname -m)"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
# `-arch=native` asks nvcc which GPU this host has; CUDA_ARCH names it instead, for compiling on a
# box with no GPU of that arch visible (a syntax/build check before the gate).
CUDA_ARCH="${CUDA_ARCH:-native}"
[[ -x "$NVCC" ]] || NVCC="$(command -v nvcc)"
# The host compiler needs the toolkit's headers too (qmesh_moe_cpp.h includes cuda_runtime.h for
# cudaStream_t), and the linker needs its stubs; derive both from whichever nvcc this is.
CUDA_HOME="$(cd "$(dirname "$NVCC")/.." && pwd)"
CUDA_LIB="$CUDA_HOME/lib64"; [[ -d "$CUDA_LIB" ]] || CUDA_LIB="$CUDA_HOME/lib"
CXX="${CXX:-g++}"
PY="${PYTHON:-python3}"
LIBDIR="$([[ $ARCH == x86_64 ]] && echo /usr/lib/x86_64-linux-gnu || echo /usr/lib/aarch64-linux-gnu)"
unset LD_LIBRARY_PATH || true
export LIBRARY_PATH="${LIBDIR}${LIBRARY_PATH:+:$LIBRARY_PATH}"
# MOE_OUT/MOE_OBJ let the pair build stage write into its own tmp dir and mv the .so into bin/,
# so a release tree never grows an obj/ directory and a failed build leaves nothing behind.
OUT="${MOE_OUT:-$ROOT/bin}"
OBJ="${MOE_OBJ:-$ROOT/obj}"
mkdir -p "$OUT" "$OBJ"

TORCH_DIR="$("$PY" -c 'import os, torch; print(os.path.dirname(torch.__file__))')"
ABI="$("$PY" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
echo "torch $TORCH_DIR abi=$ABI"

nice -n 15 "$NVCC" -O2 -std=c++17 -arch="$CUDA_ARCH" -Xcompiler -fPIC -c -x cu \
  "$ROOT/qmesh_moe_kernels.cu" -o "$OBJ/qmesh_moe_kernels.$ARCH.o"

nice -n 15 "$CXX" -O2 -std=c++20 -fPIC -pthread \
  -D_GLIBCXX_USE_CXX11_ABI="$ABI" -DTORCH_API_INCLUDE_EXTENSION_H=0 \
  -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -I"$CUDA_HOME/include" -I"$ROOT" \
  -c "$ROOT/qmesh_moe_cpp.cpp" -o "$OBJ/qmesh_moe_cpp.$ARCH.o"

nice -n 15 "$CXX" -shared -o "$OUT/libqmesh_moe_cpp.$ARCH.so" \
  "$OBJ/qmesh_moe_cpp.$ARCH.o" "$OBJ/qmesh_moe_kernels.$ARCH.o" \
  -L"$TORCH_DIR/lib" -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
  -L"$CUDA_LIB" -L"$LIBDIR" -lcudart -lcuda \
  -Xlinker -rpath="$TORCH_DIR/lib" -Xlinker -rpath="$LIBDIR"

echo BUILD_OK "$OUT/libqmesh_moe_cpp.$ARCH.so"
