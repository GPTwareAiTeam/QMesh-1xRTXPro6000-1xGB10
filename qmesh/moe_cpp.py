"""Installs qmc_compute (libqmesh_moe_cpp) as the seam server's compute hook.

Everything here runs at load time, once:

    cpp = CppMoe(server).install(handle=srv._h)   # device pointers to C, then qmc_compute into
                                                   # the proxy's hook; srv.set_fn is NOT called

The hook is the C symbol: qes_server_set_fn gets qmc_entry() and this context, so a served request
for any of the 40 layers runs from the proxy thread into the kernels without a Python frame. The
layers this context holds no stacks for (38/39, the mixed-K pack) are handed from C to the server's
own compute through a ctypes trampoline that install() keeps alive - the same call the shipped
Server.set_fn made for every layer, now made only for those two. The signature is the shipped seam
one, so the refusal codes and the protocol are untouched: this is a new callback behind the existing
setter, not a new wire.

The C side needs one qmc_group per bitrate group per layer, pointing at the stacked tensors
qmesh/exl3_moe_cuda.CudaExl3Moe already holds (the layer's resident copy of the cold pack). Nothing
is copied here except each layer's remap table, which CudaExl3Moe builds as int64 and C indexes as
int32 (~1 KiB per layer).

Set SEAM_MOE_CPP=1 in the server to take this path (worker_seam_server installs the hook), or call
install() from a bench that wants both paths in one process.
"""
from __future__ import annotations

import ctypes
import os

import torch

from qmesh.native import qes

HIDDEN = qes.HIDDEN
TOPK = qes.TOPK
NL = qes.NL

MAX_GROUPS = 8

# qes_server_fn: void* ctx, then the proxy's six (layer, rows, x, ids, w, out). The same type qes.py
# uses for its trampoline; here it types only the fallback, because the hook itself is the C symbol.
_SERVER_FN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


class QmcGroup(ctypes.Structure):
    """Mirror of qmc_group (qmesh/native/qmesh_moe_cpp.h)."""
    _fields_ = [("n_experts", ctypes.c_int32),
                ("n_cold", ctypes.c_int32),
                ("words13", ctypes.c_int32),
                ("words2", ctypes.c_int32),
                ("inter", ctypes.c_int32),
                ("cb", ctypes.c_int32),
                ("trellis13", ctypes.c_void_p),
                ("suh13", ctypes.c_void_p),
                ("svh13", ctypes.c_void_p),
                ("trellis2", ctypes.c_void_p),
                ("suh2", ctypes.c_void_p),
                ("svh2", ctypes.c_void_p),
                ("remap", ctypes.c_void_p)]


def lib_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "native", "bin",
                        f"libqmesh_moe_cpp.{os.uname().machine}.so")


_lib = None


def lib():
    """The shared library, loaded after torch so libtorch's symbols are already in the process."""
    global _lib
    if _lib is not None:
        return _lib
    path = lib_path()
    if not os.path.exists(path):
        raise RuntimeError(f"{path} is missing: run qmesh/native/build_moe.sh")
    h = ctypes.CDLL(path)
    h.qmc_open.restype = ctypes.c_int
    h.qmc_open.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                           ctypes.POINTER(ctypes.c_void_p)]
    h.qmc_add_layer.restype = ctypes.c_int
    h.qmc_add_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                ctypes.POINTER(QmcGroup)]
    h.qmc_compute.restype = ctypes.c_int
    h.qmc_compute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    h.qmc_last_call_ns.restype = ctypes.c_int
    h.qmc_last_call_ns.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
                                   ctypes.POINTER(ctypes.c_int64)]
    h.qmc_stats.restype = ctypes.c_int
    h.qmc_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
                            ctypes.POINTER(ctypes.c_int64)]
    h.qmc_out_mode.restype = ctypes.c_int
    h.qmc_out_mode.argtypes = [ctypes.c_void_p]
    h.qmc_set_out_mode.restype = ctypes.c_int
    h.qmc_set_out_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    h.qmc_set_block_m.restype = ctypes.c_int
    h.qmc_set_block_m.argtypes = [ctypes.c_void_p, ctypes.c_int]
    h.qmc_dump_align.restype = ctypes.c_int
    h.qmc_dump_align.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    h.qmc_last_error.restype = ctypes.c_char_p
    h.qmc_last_error.argtypes = []
    h.qmc_entry.restype = ctypes.c_void_p
    h.qmc_entry.argtypes = []
    h.qmc_set_fallback.restype = ctypes.c_int
    h.qmc_set_fallback.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    h.qmc_close.restype = None
    h.qmc_close.argtypes = [ctypes.c_void_p]
    _lib = h
    return h


def last_error() -> str:
    return (lib().qmc_last_error() or b"").decode(errors="replace")


class CppMoe:
    """The C compute hook, one context for a loaded server."""

    def __init__(self, server, ws_rows=None):
        self.server = server
        # Piece ceiling for the C picker: None = the server's own --ws-rows, which is what a served
        # request must use (the ladder is duplicated in C, and the two must agree).
        self.ws_rows = None if ws_rows is None else int(ws_rows)
        self.lib = lib()
        self.ctx = None
        self._keep = []        # remap tables (int32) and the group arrays handed to qmc_add_layer
        self.layers = []       # layers the C context serves
        self._registered = set()
        self.fallback = []     # layers left to the server's path (mixed-K: no cuda-exl3 stacks)

    def install(self, layers=None, handle=None):
        """Register the given layers' stacks into the C context; returns self.

        Pointers only. Each group hands C the data_ptr() of the tensors CudaExl3Moe already holds in
        server._cx_layers, so a layer that SEAM_MOE_PRELOAD=1 restacked at load is not restacked
        again and no stacked bytes are duplicated: the only allocation here is each layer's remap
        table in int32 (~1 KiB). Those copies and the group arrays live in self._keep; the stacked
        tensors live in server._cx_layers, which is what makes them outlive this object.

        `layers` defaults to all 40 (a served request can name any of them); a bench that measures
        one layer passes just that one, so a run with SEAM_MOE_PRELOAD=0 restacks one layer.

        A layer whose restack is not there - the mixed-K module (38/39, or any layer the load
        reported as SEAM_MOE_NOFIT / SEAM_MOE_NOEXT) - is recorded in self.fallback and skipped:
        there are no cuda-exl3 stacks to point at, and refusing it would fail requests the
        server's own path still answers. dispatch() is what routes those.

        `handle` is the seam proxy (qes.Server._h). Given it, the proxy's hook becomes this
        library's own qmc_compute, installed through the shipped qes_server_set_fn, so a served
        request never re-enters the interpreter; the layers C cannot serve reach the server
        through a trampoline this object keeps alive (see _install_on_proxy). Without it nothing is
        installed and the caller routes with dispatch(), which is how the bench compares both
        paths in one process.
        """
        srv = self.server
        dev = torch.device(srv.device) if not isinstance(srv.device, torch.device) else srv.device
        ordinal = int(dev.index or 0)
        want = list(range(NL)) if layers is None else [int(L) for L in layers]

        ctx = ctypes.c_void_p()
        ws = int(getattr(srv, "ws_rows", 0)) if self.ws_rows is None else self.ws_rows
        rc = self.lib.qmc_open(ordinal, HIDDEN, TOPK, ws, ctypes.byref(ctx))
        if rc != 0:
            raise RuntimeError(f"CppMoe: qmc_open failed: {last_error()}")
        self.ctx = ctx

        for layer in want:
            ffn, _ = srv.layers[layer]
            cx = srv._cx_moe(layer, ffn)
            if cx is None:
                self.fallback.append(layer)
                continue
            groups = []
            for grp in cx.groups:
                if grp.n_experts == 0:
                    continue
                r32 = grp.remap.to(torch.int32).contiguous()
                self._keep.append(r32)
                g = QmcGroup()
                g.n_experts = int(grp.n_experts)
                g.n_cold = int(grp.remap.numel())
                g.words13 = int(grp.trellis13.shape[3])
                g.words2 = int(grp.trellis2.shape[3])
                g.inter = int(grp.trellis2.shape[1]) * 16
                g.cb = int(grp.cb)
                g.trellis13 = grp.trellis13.data_ptr()
                g.suh13 = grp.suh13.data_ptr()
                g.svh13 = grp.svh13.data_ptr()
                g.trellis2 = grp.trellis2.data_ptr()
                g.suh2 = grp.suh2.data_ptr()
                g.svh2 = grp.svh2.data_ptr()
                g.remap = r32.data_ptr()
                groups.append(g)
            if not groups:
                self.fallback.append(layer)
                continue
            arr = (QmcGroup * len(groups))(*groups)
            self._keep.append(arr)
            rc = self.lib.qmc_add_layer(self.ctx, layer, len(groups), arr)
            if rc != 0:
                raise RuntimeError(f"CppMoe: qmc_add_layer({layer}) failed: {last_error()}")
            self.layers.append(layer)
            self._registered.add(layer)

        print(f"SEAM_MOE_CPP_REGISTER layers={len(self.layers)} fallback={self.fallback} "
              f"ws_rows={srv.ws_rows} dev={ordinal}", flush=True)
        if not self.layers:
            raise RuntimeError(f"CppMoe: no layer has cuda-exl3 stacks to serve")
        if handle is not None:
            # The proxy reads the response at the stride the seam's wire caps laid the slot out with
            # (fp32, or bf16/fp16 under SEAM_QES_OUT): the slot is the seam's decision, made before
            # any session, so the hook follows it. A SEAM_MOE_CPP_OUT that contradicts it would
            # write the wrong width into the slot, so that is refused, not overridden.
            want = qes.out_mode()
            mode = self.out_mode()
            if want < 0 or (mode != 0 and mode != want):
                self.close()
                raise RuntimeError(f"CppMoe: out mode {mode} (SEAM_MOE_CPP_OUT) contradicts the "
                                   f"seam's reply format {want} (SEAM_QES_OUT); refusing to install")
            if mode != want:
                self.set_out_mode(want)
            print(f"SEAM_MOE_CPP_WIRE caps=0x{qes.caps():x} out={qes.OUT_NAMES[want]}", flush=True)
            self._install_on_proxy(handle)
        return self

    def _install_on_proxy(self, handle) -> None:
        """Make qmc_compute the proxy's hook, with the two unserved layers falling back to Python.

        qes_server_set_fn takes a plain function pointer, and qmc_compute's arguments are already a
        qes_server_fn, so the install is one cast on the C symbol - the request path from the proxy
        thread to the kernels stays in C for every layer this context registered. The fallback is
        a ctypes trampoline over the server's own compute, i.e. exactly what Server.set_fn used to
        install, kept here so 38/39 still answer; it must not raise, or ctypes swallows the
        exception and invents a return, so it refuses with the protocol's code 5 instead.

        The symbol is resolved on the seam library's own handle: qes.lib() binds qes_server_set_fn
        with the ctypes callback type as its second argument, and the C address is a plain void*.
        """
        qlib = ctypes.CDLL(qes.lib()._name)
        qlib.qes_server_set_fn.restype = None
        qlib.qes_server_set_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        def tramp(ctx, layer, rows, x, ids, w, out):
            try:
                return int(self.server.compute(layer, rows, x, ids, w, out) or 0)
            except BaseException as e:   # noqa: BLE001 - the proxy must hear "no answer"
                print(f"SEAM_COMPUTE_REFUSED cpp fallback raised {type(e).__name__}: {e}",
                      flush=True)
                return 5

        self._fallback_cb = _SERVER_FN(tramp)   # keep alive: C calls it from proxy threads
        h = handle if isinstance(handle, ctypes.c_void_p) else ctypes.c_void_p(int(handle))
        if self.lib.qmc_set_fallback(self.ctx, ctypes.cast(self._fallback_cb, ctypes.c_void_p),
                                     h) != 0:
            raise RuntimeError("CppMoe: qmc_set_fallback rejected")
        qlib.qes_server_set_fn(h, self.lib.qmc_entry(), self.ctx)
        print(f"SEAM_MOE_CPP_PROXY entry=qmc_compute fallback_layers={self.fallback}", flush=True)

    def dispatch(self, layer, rows, x, ids, w, out) -> int:
        """Route a request in Python: C where it can, server path where not.

        Only for callers that pass srv.set_fn(cpp.dispatch) instead of install(handle=...), and for
        the bench, which drives both paths from one process. On the proxy the hook is qmc_compute
        itself; this function's fallback half is what install() hands C as the trampoline.

        One set lookup per request; the per-expert work stays inside qmc_compute. A layer in
        self.fallback runs server.compute unchanged, which is what keeps the mixed-K layers (38/39)
        answering instead of refusing. A raise is left to the seam trampoline, which turns it into
        the protocol's own refusal code.
        """
        if layer in self._registered:
            return self.compute(layer, rows, x, ids, w, out)
        return self.server.compute(layer, rows, x, ids, w, out)

    # -- direct calls, for the bench (the server calls the same symbol through the hook) ----------
    def compute(self, layer, rows, x, ids, w, out) -> int:
        return int(self.lib.qmc_compute(self.ctx, int(layer), int(rows), int(x), int(ids),
                                        int(w), int(out)))

    def last_call_ns(self):
        wall = ctypes.c_int64()
        dev = ctypes.c_int64()
        if self.lib.qmc_last_call_ns(self.ctx, ctypes.byref(wall), ctypes.byref(dev)):
            return None
        return wall.value, dev.value

    def out_mode(self) -> int:
        """0 = fp32 slot, 1 = bf16, 2 = fp16: the stride the reader of the response must use."""
        return int(self.lib.qmc_out_mode(self.ctx))

    def set_out_mode(self, mode: int) -> int:
        """Change the response format; returns the old mode. The reader must follow immediately."""
        old = int(self.lib.qmc_set_out_mode(self.ctx, int(mode)))
        if old < 0:
            raise RuntimeError(f"CppMoe: set_out_mode({mode}) rejected")
        return old

    def set_block_m(self, block_m: int) -> int:
        """0 = pick the tier from the previous call's live count; >0 freezes it for capture."""
        old = int(self.lib.qmc_set_block_m(self.ctx, int(block_m)))
        if old < 0:
            raise RuntimeError(f"CppMoe: set_block_m({block_m}) rejected")
        return old

    def dump_align(self, layer: int, group: int):
        """The last piece's alignment for one group, as (sorted_ids, eids, nblk, total, block_m).

        Requires SEAM_MOE_CPP_DUMP=1 in the environment before qmc_open (the host mirror is
        allocated there); block_m/total/nblk are 0 when the group has nothing live.
        """
        cap = 1 << 22
        sorted_buf = ctypes.create_string_buffer(cap * 4)
        eids_buf = ctypes.create_string_buffer(cap * 4)
        nblk = ctypes.c_int()
        total = ctypes.c_int()
        block_m = ctypes.c_int()
        rc = int(self.lib.qmc_dump_align(self.ctx, int(layer), int(group), sorted_buf, eids_buf,
                                         ctypes.byref(nblk), ctypes.byref(total),
                                         ctypes.byref(block_m)))
        if rc:
            raise RuntimeError(f"CppMoe: dump_align failed: {last_error()}")
        import numpy as _np
        sorted_ids = _np.frombuffer(sorted_buf, dtype=_np.int32, count=max(total.value, 0)).copy()
        eids = _np.frombuffer(eids_buf, dtype=_np.int32, count=max(nblk.value, 0)).copy()
        return sorted_ids, eids, nblk.value, total.value, block_m.value

    def stats(self):
        calls = ctypes.c_int64()
        refused = ctypes.c_int64()
        self.lib.qmc_stats(self.ctx, ctypes.byref(calls), ctypes.byref(refused))
        return calls.value, refused.value

    def close(self):
        if self.ctx is not None:
            self.lib.qmc_close(self.ctx)
            self.ctx = None
        self._keep.clear()
        self.layers.clear()
        self._registered.clear()
