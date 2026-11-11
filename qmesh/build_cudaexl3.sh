#!/bin/sh
# Build the vendored cuda-exl3 extension for the GX2 cold-expert path (GB10, sm_121).
#
# The extension is imported from vendor/cuda-exl3/src (qmesh/exl3_moe_cuda.py puts that on
# sys.path), so the .so must land in place: vendor/cuda-exl3/src/cuda_exl3/_C.<abi>.so.
# It is not tracked -- every checkout builds it once, in the container that runs the server.
#
#   sh qmesh/build_cudaexl3.sh            # ~100 s at MAX_JOBS=8
#
# Override TORCH_CUDA_ARCH_LIST to build for another part (12.0 is P7; same sources, no guard).
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here/vendor/cuda-exl3"

TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.1}
MAX_JOBS=${MAX_JOBS:-8}
export TORCH_CUDA_ARCH_LIST MAX_JOBS
nice -n 10 python3 setup.py build_ext --inplace

# Import check: the GX path needs exl3_moe_gemm with the shared-accumulator arguments, and the
# adapter's own probe reads that schema, so checking it here is checking the build. It also has to
# assert the *names*: on 2026-11-02 a built .so was older than the tree it served and lacked
# exl3_set_moe_acc_cap, so `SEAM_MOE_PRELOAD=1` fell back to the mixed-K module with only a
# SEAM_MOE_NOEXT line in the server log -- a release that passes its stamp and serves at a fraction of
# the rate. Every attribute qmesh/exl3_moe_cuda.py calls is required here, so the build fails instead.
CUDA_EXL3_SRC="$here/vendor/cuda-exl3/src" python3 - <<'PY'
import os, sys
sys.path.insert(0, os.environ["CUDA_EXL3_SRC"])
import cuda_exl3.ops  # noqa: F401  (loads the extension)
import torch
schema = str(torch.ops.cuda_exl3_C.exl3_moe_gemm.default._schema)
assert "accumulate" in schema, schema
need = ["exl3_moe_gemm", "exl3_moe_combine", "exl3_moe_had_in", "exl3_moe_glu_had_in",
        "exl3_set_moe_acc_cap"]
missing = [n for n in need if not hasattr(torch.ops.cuda_exl3_C, n)]
assert not missing, "cuda_exl3_C is missing %s (the .so is older than the tree it serves)" % missing
print("cuda_exl3 ok:", schema, "ops=" + ",".join(need))
PY
