"""Growable device ranges on CUDA virtual memory management (notes/GrowKV.md).

A VmmRange reserves address space for its maximum size once and maps physical memory in granules (2 MiB on the P7)
only for the prefix that is in use. The base address never moves, so kernels and captured CUDA graphs that bake it
stay valid across grow and shrink. Tensors are views of the full range; only the mapped prefix may be touched.

The native part (cache/native/vmm_pool.cpp, outside exllamav3_ext/ so the main extension does not compile it) is a
small JIT module of its own, built on first use.
"""
from __future__ import annotations

import os
import threading

import torch

_ext = None
_ext_lock = threading.Lock()


def ext():
    global _ext
    with _ext_lock:
        if _ext is None:
            from torch.utils.cpp_extension import load, CUDA_HOME
            src = os.path.join(os.path.dirname(__file__), "native", "vmm_pool.cpp")
            inc = [os.path.join(CUDA_HOME, "include")] if CUDA_HOME else []
            _ext = load("exllamav3_vmm", [src], extra_include_paths = inc,
                        extra_ldflags = ["-L/usr/lib/x86_64-linux-gnu", "-lcuda"], verbose = False)
        return _ext


class VmmRange:
    """One reserved device range of `max_bytes`, physically backed for [0, mapped)."""

    def __init__(self, device: torch.device, max_bytes: int):
        self.device = torch.device(device)
        self.handle = ext().vmm_create(self.device.index or 0, int(max_bytes))
        base, self.reserved, self.mapped, self.granule = ext().vmm_info(self.handle)
        self.base = base
        # torch.from_blob asks the runtime which device owns the base pointer, which fails for a range with nothing
        # mapped: the first granule stays mapped for the range's lifetime (the floor of trim)
        self.floor = self.granule
        self.mapped = ext().vmm_grow(self.handle, self.floor)
        ext().vmm_view(self.handle, self.floor).zero_()

    def view(self, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
        n = 1
        for s in shape:
            n *= s
        nbytes = n * torch.empty((), dtype = dtype).element_size()
        assert nbytes <= self.reserved, f"VmmRange view {nbytes} B > reserved {self.reserved} B"
        return ext().vmm_view(self.handle, nbytes).view(dtype).view(shape)

    def ensure(self, nbytes: int) -> int:
        """Map so that at least nbytes from the base are backed, zero-filled (cudaMalloc'd pools were torch.zeros);
        returns the bytes newly mapped. The fill is enqueued on the current stream, ahead of any use."""
        before = self.mapped
        if nbytes > self.mapped:
            self.mapped = ext().vmm_grow(self.handle, int(nbytes))
            ext().vmm_view(self.handle, self.mapped)[before:].zero_()
        return self.mapped - before

    def trim(self, nbytes: int) -> int:
        """Release physical memory above nbytes (rounded up to a granule); returns the bytes released. The caller
        guarantees no queued or running kernel reads the released tail."""
        before = self.mapped
        nbytes = max(int(nbytes), self.floor)
        if nbytes < self.mapped:
            self.mapped = ext().vmm_shrink(self.handle, nbytes)
        return before - self.mapped

    def round_up(self, nbytes: int) -> int:
        g = self.granule
        return (nbytes + g - 1) // g * g

    def free(self):
        if self.handle:
            ext().vmm_destroy(self.handle)
            self.handle = 0
            self.mapped = 0
