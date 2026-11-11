#!/usr/bin/env python3
"""GX1 expert server for the native P7<->GX1 seam (qmesh/native/qes.py).

One fused mixed-K BlockSparseMLP per layer holds that layer's cold experts (the
placement's `warm` list, in order). A seam request carries cold-local ids [rows,6]
(-1 = expert resident on P7) and routing weights [rows,6]. The compute hook runs on the
seam's proxy thread: it copies the request out of the pinned slab, runs the module with
`external_selected`/`external_weights` on a dedicated CUDA stream, and writes fp32 output
back into the pinned response region.

The experts are loaded once and the server then serves client sessions back to back: the
control port (--port) answers a versioned hello with the row buckets it can serve and the
data port of each, and a session lives as long as its control connection. Client exit
(clean, ctrl-c, kill -9) tears down that session's QPs only, so benchmark arms can run one
after another - or one per pair - without reloading the cold pack.

-1 picks are mapped to local expert 0 with weight 0, so they contribute exact zeros
(the module's own range check stays satisfied).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import ctypes
import gc
import json
import os
import sys
import threading
import time

import numpy as np
import torch

from qmesh.native import qes

# exllamav3 is imported inside SeamServer.__init__: its module import JIT-builds exllamav3_ext
# (minutes on a cold cache, and it blocks on the torch extension baton), which a --stub server
# must never pay for - the wire/protocol baseline has to come up on a box whose GPU is busy.

# libc handle for malloc_trim(0): the only way Python can hand glibc's freed arenas back to the
# kernel (see _trim_host). Absent on a musl box, where the trim is simply skipped.
try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:  # noqa: BLE001 - a missing libc handle costs the trim, not the server
    _LIBC = None

HIDDEN = qes.HIDDEN
TOPK = qes.TOPK
NL = qes.NL

# Per-call audit of the cold ids the client sent (see _check_ids). Logged for calls up to this many
# rows: small calls are where a bad id shows, and the log stays quiet for prefill-sized buckets.
_IDS_LOG_MAX = int(os.environ.get("SEAM_IDS_LOG_MAX_ROWS", "6"))

# Printed ids for a rejected request: the whole array for small calls (the values are the
# diagnosis), capped for big ones. 8b2674f9 dropped this line while adding MOE_KERNEL below and
# left its use in _check_ids behind, so every compute raised NameError and the server fail-stopped
# on its first request (2026-11-01 17:31 GX2). That is the worst shape of bug this file can carry:
# it lives on the rejection path, so only a healthy run reaches it, and taking the box down is the
# opposite of what the check is for.
_IDS_BAD_MAX = int(os.environ.get("SEAM_IDS_BAD_MAX_ROWS", "64"))

# Cold-expert kernel (see _forward_piece). "cudaexl3" runs a layer through cuda-exl3's MoE gemm
# (vendor/cuda-exl3), "mixedk" through exllamav3's mixed-K module. cuda-exl3 is the default: it is
# faster in every bucket measured on GX2 (layer 5, 268 cold experts) -- 16.9 / 30.2 / 54.9 / 103.6
# ms at 512 / 2048 / 4096 / 8192 rows against 30.3 / 41.3 / - / 145.3 for the module -- but it needs
# one restack per layer into per-bitrate trellis stacks on first use, which also drops the
# per-expert copies. A layer whose restack raises is served by the module from then on.
MOE_KERNEL = os.environ.get("SEAM_MOE_KERNEL", "cudaexl3")

# Read at import so a renamed or dropped constant stops the server before it serves anyone. The
# 17:31 NameError resolved only when a client's first forward reached the audit, which cost that
# client 30 s of timeouts and the box a full reload; the same mistake here costs one start.
assert _IDS_LOG_MAX >= 0 and _IDS_BAD_MAX >= 0 and MOE_KERNEL in ("cudaexl3", "mixedk"), \
    "cold-id audit caps and the kernel name must resolve at import"

# Reply format the seam's wire caps select (SEAM_QES_OUT, protocol v5): the slabs were laid out for
# it in qes_server_open before any session, so every Python writer of a response region must use
# it. Read at import, like the audit caps above: an unknown word stops the box before it serves.
OUT_MODE = qes.out_mode()
assert OUT_MODE in (0, 1, 2), "SEAM_QES_OUT must be f32|bf16|fp16"
OUT_DTYPE = (torch.float32, torch.bfloat16, torch.float16)[OUT_MODE]
OUT_ELEM = 4 if OUT_MODE == 0 else 2


def _out_view(out_ptr, rows):
    """The response region as a [rows, HIDDEN] tensor in the wire's reply dtype (pinned, in place)."""
    raw = torch.from_numpy(view(out_ptr, rows * HIDDEN * OUT_ELEM, np.uint8))
    return raw.view(OUT_DTYPE).reshape(rows, HIDDEN)


def _check_ids(layer, rows, ids, num_experts):
    """Audit one request's cold ids on the host, straight from the pinned slab (no device work).

    Contract with the client: 0 <= id < num_experts addresses this layer's cold expert `id`, and a
    negative id means "this pick is not cold on this layer" (the caller maps it to local expert 0
    with weight 0, so it contributes exact zeros). A positive id outside the range is a client bug,
    and computing with it reads past the module's expert tables inside a fused launch; the illegal
    access then surfaces layers later, at whatever call next synchronises, which is why the
    rows=6 death of 2026-11-01 has no explanation in its own log. So the range is enforced here,
    where the ids arrive, and every small call is logged in full so the same run can rule the ids
    in or out instead of guessing at the launch.
    """
    if rows <= 0 or ids.size == 0:
        return
    lo, hi = int(ids.min()), int(ids.max())
    neg = int((ids < 0).sum())
    bad = int((ids >= num_experts).sum())
    if rows <= _IDS_LOG_MAX:
        print(f"SEAM_IDS layer={layer} rows={rows} n={ids.size} min={lo} max={hi} "
              f"miss={neg} n_cold={num_experts}", flush=True)
    if bad:
        if ids.size <= _IDS_BAD_MAX * TOPK:
            print(f"SEAM_IDS_BAD layer={layer} rows={rows} n_cold={num_experts} "
                  f"ids={ids.reshape(-1).tolist()}", flush=True)
        raise RuntimeError(f"cold ids out of range: {bad} of {ids.size} picks in [{lo},{hi}] "
                           f"for {num_experts} cold experts on layer {layer}, largest bad id "
                           f"{int(ids[ids >= num_experts][:8].max())}")


# How much MemAvailable one forward needs before it is even launched. Per-request device scratch is
# priced per row by the picker (_pick_step), but that is only the module's row buffers: the exl3
# restack transient, the allocator's growth and the pinned slab are extra, and on a box that is
# already swapping the allocation itself is what fails - a 6-row forward died on 2026-11-01 with
# avail_min_mib=560 swap_max_mib=3762, and the launch failure it raised killed a process that held
# a 113 GiB pack. 0 = off, which is the shipped default because the floor has to be measured per
# box, not guessed: the dev box serves 2048-row pieces at avail_mib=507 (SEAM_WS_PICK ... need_mib=331),
# so a floor set from the failing number alone would refuse every request. Set SEAM_FWD_FLOOR_MIB
# from a run that logs avail_mib on both the served and the refused calls.
FWD_FLOOR_MIB = int(os.environ.get("SEAM_FWD_FLOOR_MIB", "0"))

# Refusal codes. They travel to the client in the slot's fail word (see qmesh_expert_seam.h) and are
# what the client's proxy prints instead of "the seam transport is dead".
REFUSE_MEM = 4
REFUSE_COMPUTE = 5


def _free_cached():
    """Return cached-but-unused device blocks to the driver before deciding there is no room."""
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - a context that cannot empty is a context that is gone
        pass


def _cuda_ok():
    """True while this process's CUDA context can still run a kernel.

    The difference between "this request does not fit" and "this process can never answer again",
    and the only thing allowed to kill the server: a launch failure that leaves the context unusable
    turns every later session into the same failure, so the box is better off reloaded than
    accepting. A live context means the request was refused, not the server.
    """
    try:
        torch.cuda.synchronize()
        torch.zeros(1, device = "cuda")
        torch.cuda.synchronize()
        return True
    except Exception:  # noqa: BLE001 - any raise here is the answer: unusable
        return False


def build_layer(base, cold_ids, device):
    """Subset the layer's routed-expert module to `cold_ids` (checkpoint expert ids, ordered).

    Places all cold experts in one module so the fused mixed-K path runs one launch per layer.

    The copy shares the one Config instead of duplicating it: a plain deepcopy clones
    `config.stc`, the safetensors index of the whole pack, into every layer module - 65 MiB per
    copy, 40 copies, all of it held by the modules and therefore unreclaimable by gc/malloc_trim
    after the load (measured: 2.6 GiB of the 3.4 GiB swap bill on GX2)."""
    ffn = copy.deepcopy(base, {id(base.config): base.config})
    ffn.gates = [ffn.gates[e] for e in cold_ids]
    ffn.ups = [ffn.ups[e] for e in cold_ids]
    ffn.downs = [ffn.downs[e] for e in cold_ids]
    keep = set(ffn.gates + ffn.ups + ffn.downs)
    ffn.modules = [m for m in ffn.modules if m in keep]
    ffn.num_experts = len(cold_ids)
    ffn.num_local_experts = len(cold_ids)
    ffn.routing_first = 0
    ffn.routing_last = len(cold_ids)
    ffn.routing_gate = None
    ffn.shared_experts = None
    ffn.cpu_split_first = None
    ffn._split_perm = None
    ffn.load_routing = lambda **kw: None
    ffn.load(device)
    torch.cuda.synchronize()
    return ffn, len(cold_ids)


class SeamServer:
    def __init__(self, model_dir, placement_path, device, stub = False, ws_rows = 0,
                 fail_after = 0):
        # fail_after: testing hook for the refusal path - raise on call N+1. 0 = never.
        self.fail_after = int(fail_after or 0)
        self.seen = 0
        # Refused requests since start (SEAM_COMPUTE_REFUSED). A refusal is not a failure of the
        # server: the session it belonged to ended, the listener did not.
        self.refusals = 0
        # One compute at a time. The C proxy calls this hook from a thread per open handle, and a
        # client with several row buckets keeps several handles (1/6/64...) with one request in
        # flight each, so two threads can be inside _forward at once: the pinned-slab pointers are
        # per handle, but the module (its _buf_get scratch, its fused row buffers) and the one CUDA
        # stream are not. The Python-side tensor bookkeeping of the two calls then interleaves while
        # the other thread sits in stream.synchronize() or .cpu() with the GIL released - the same
        # scratch buffer is rebuilt under a kernel that is still reading it. That is a CUDA illegal
        # access / launch failure inside the GX compute, i.e. exactly the "6-row" deaths of
        # 2026-11-01, and it cannot be seen by a single-handle benchmark (that is serialized).
        self.compute_lock = threading.Lock()
        # ws_rows: the largest piece a request may be served in (see _forward); the piece actually
        # used is picked per call from WS_LADDER as far as the box's MemAvailable pays for it, so
        # this is a ceiling, not a promise. 0 = no split at all (one piece, whatever it costs).
        placement = json.load(open(placement_path))
        # One line, before anything is loaded, naming the table this server's cold-local ids are
        # positions in. The Hello advertises its sha (main), so a client on another table is
        # refused; this line is what an operator reads to know *which* table that was.
        print(f"SEAM_PLACEMENT path={(placement_path)} ranked={int(bool(placement.get('ranked')))} "
              f"warm_4bpw={placement.get('warm_4bpw_experts', '-')} "
              f"source_sha={(placement.get('source_sha256') or '-')[:12]}", flush=True)
        self.device = device
        self.stub = stub
        self.stream = None
        self.layers = {}
        self.ws_rows = int(ws_rows or 0)
        self.ws_auto = WS_AUTO
        if stub:
            # Wire/protocol baseline: no experts are needed and nothing here touches CUDA or
            # exllamav3, so a stub server is up in seconds even while the GPU holds another pack.
            return
        from exllamav3 import Config, Model

        # Refuse to start a load this box cannot hold. The pack is 113.65 GiB of a 121.6 GiB
        # unified pool: a load started with less than pack + margin available does not fail, it
        # swaps (3.8 GiB seen 2026-11-01), and every expert call then page faults (4 s outliers in
        # SEAM_COMPUTE) or fails a launch outright. Better an explicit exit 3 than a slow poison.
        self.pack_paths = [os.path.join(model_dir, f"cold-L{layer:02d}.safetensors")
                           for layer in range(NL)]
        sizes = [os.path.getsize(p) if os.path.exists(p) else 0 for p in self.pack_paths]
        self.pack_mib = sum(sizes) / 2**20
        # 2048, not 1024: the load itself costs ~1100-1300 MiB above the pack (page cache, the
        # Config/Model parse, the expert tensors' transient copies), and compute needs >= ~1600 MiB
        # post-load. A 1024 margin let GX2 land at avail 1177 / swapped 3766 and die on its first
        # forward (2026-11-01, "unspecified launch failure" at rows=1). The starter should set this
        # explicitly per box (GX2 is the tighter one); this default only stops a bare start.
        margin = int(os.environ.get("SEAM_LOAD_MARGIN_MIB", "2048"))
        avail = _avail_mib()
        if not sizes[0]:
            print(f"SEAM_NO_PACK dir={model_dir} - expected cold-L00.safetensors", flush=True)
            sys.exit(3)
        if (os.environ.get("SEAM_FORCE_LOAD", "0") == "0" and avail >= 0
                and avail < self.pack_mib + margin):
            print(f"SEAM_NO_MEMORY need_mib={self.pack_mib:.0f}+{margin} avail_mib={avail} "
                  f"swapped_mib={_swap_used_mib()} - refusing to load: the pack does not fit in "
                  f"MemAvailable and loading it would swap the box. Free it, or set "
                  f"SEAM_FORCE_LOAD=1 to try anyway", flush=True)
            sys.exit(3)
        self.fadvise = os.environ.get("SEAM_FADVISE", "1") != "0"
        print(f"SEAM_PACK_LOAD need_mib={self.pack_mib:.0f} avail_mib={avail} margin_mib={margin} "
              f"fadvise={int(self.fadvise)} swapped_mib={_swap_used_mib()}", flush=True)

        self.stream = torch.cuda.Stream(device=device)
        # The kernel has to be importable before the load starts: vendor/cuda-exl3 is built per box
        # (qmesh/build_cudaexl3.sh) and a server started on a checkout without it would otherwise
        # die on the first request, deep inside compute()'s fail-stop.
        self.preload = PRELOAD_CX and MOE_KERNEL == "cudaexl3" and not stub
        self._cx_cls = None
        if self.preload:
            try:
                from qmesh.exl3_moe_cuda import CudaExl3Moe  # noqa: F401  (loads the extension)
                self._cx_cls = CudaExl3Moe
            except Exception as e:
                self.preload = False
                print(f"SEAM_MOE_NOEXT preload=off err={type(e).__name__}: {e} "
                      f"- every layer will stay on the mixed-K module", flush=True)
        self._cx_layers: dict = {}
        self.layer_n: dict = {}
        # Per layer, the file's own claim about its cold order: `warm_bpw` (the bitrate of each
        # cold position) and `warm_4bpw_n` (the 4-bit group boundary). None for an unranked file.
        self.layer_bpw_file: dict = {}
        self.layer_4n_file: dict = {}
        config = Config.from_directory(model_dir)
        model = Model.from_config(config)
        for layer in range(NL):
            layer_pl = placement["layers"][str(layer)]
            cold = layer_pl["warm"]
            self.layer_n[layer] = len(cold)
            # The ranked file puts every 4-bit cold expert before every 2-bit one and records that
            # boundary; the shard says, per expert, which bitrate it was cut at. Both must name the
            # same set and the same boundary, or the client's cold-local ids mean something else
            # here than they mean there -- a silent id shift, not an error anywhere else.
            self.layer_bpw_file[layer] = layer_pl.get("warm_bpw")
            self.layer_4n_file[layer] = layer_pl.get("warm_4bpw_n")
            hit = None
            if self.preload and cold and DIRECT_PACK:
                # Slabs first, then the shard's bytes into their slices: the layer's only copy of
                # its experts is the one the kernels read, so this costs the layer's cold bytes and
                # cannot be refused for want of a second copy's worth of free memory.
                t0 = time.perf_counter()
                avail0, swap0 = _avail_mib(), _swap_used_mib()
                try:
                    hit = self._cx_cls.from_pack(self.pack_paths[layer], layer, cold, device,
                                                 fadvise = self.fadvise)
                except Exception as e:  # noqa: BLE001 - a bad shard must not lose the other 39
                    hit = None
                    print(f"SEAM_MOE_DIRECT_FAIL layer={layer} n={len(cold)} "
                          f"err={type(e).__name__}: {e} - falling back to the per-expert module "
                          f"path for this layer", flush=True)
                if hit is not None:
                    self._cx_layers[layer] = hit
                    self.layers[layer] = (None, len(cold))
                    avail1 = _avail_mib()
                    prev = self.__dict__.get("_cx_avail_min", avail1)
                    self._cx_avail_min = min(prev, avail1)
                    print(f"SEAM_MOE_DIRECT layer={layer} n={len(cold)} groups={hit.describe()} "
                          f"slab_mib={hit.nbytes() / 2**20:.0f} "
                          f"read_ms={1e3 * (time.perf_counter() - t0):.0f} "
                          f"avail_mib={avail0}->{avail1} swap_mib={swap0}->{_swap_used_mib()} "
                          f"rss_peak_mib={_rss_peak_mib()} cached_mib={_cached_mib()}",
                          flush=True)
                    self._check_placement_groups(layer, cold, hit)
            if hit is None:
                base = next(m for m in model.modules if m.key == f"layers.{layer}").mlp
                self.layers[layer] = build_layer(base, cold, device)
                del base
                torch.cuda.empty_cache()
            # Canonical compact JSON hashes the cold-local-to-global registration order.
            import hashlib
            digest = hashlib.sha256(json.dumps(cold, separators=(",", ":")).encode()).hexdigest()[:8]
            print(f"SEAM_REG_ORDER layer={layer} sha8={digest}", flush=True)
            # Host side of the same bill: the loader's CPU buffers for this layer are dead now, and
            # on a unified-pool box anything the kernel is holding on their behalf is swap the next
            # layer's cudaMalloc cannot use (see _trim_host). Reported per layer so a load that
            # leaks is visible in the log rather than in somebody's swap census afterwards.
            released = _trim_host()
            host_rss, host_swap = _host_rss_swap_mib()
            dropped = _drop_page_cache(self.pack_paths[layer]) if self.fadvise else False
            print(f"layer {layer}: {len(cold)} cold experts, "
                  f"{torch.cuda.memory_allocated(device) / 2**30:.2f} GiB "
                  f"avail_mib={_avail_mib()} cached_mib={_cached_mib()} "
                  f"rss_peak_mib={_rss_peak_mib()} swap_mib={_swap_used_mib()} "
                  f"host_rss_mib={host_rss} host_swap_mib={host_swap} host_trim_mib={released} "
                  f"cached_dropped={int(dropped)} direct={int(hit is not None)}", flush=True)
            # Convert now, not on the first request for this layer: this is the only moment the
            # group transient fits (see PRELOAD_CX). _cx_moe prints SEAM_MOE_CX or SEAM_MOE_NOFIT
            # per layer, and a refusal leaves the layer on the mixed-K module, which is correct
            # because nothing of the module was released in that case.
            if self.preload and hit is None:
                self._cx_moe(layer, self.layers[layer][0])
        # The CPU model and its config were only the factory: every layer module is built now and
        # owns its device tensors, and nothing else reads them (build_layer shares `config` and
        # never touches `stc` again). Closing the collection hands back the pack's open descriptors
        # and the loader's pinned staging pool (stloader_close_file frees it), and the trim below
        # hands back the loader's arenas, so the pack's 113.5 GiB is the only thing a loaded server
        # reserves -- 3.44 GiB of VmSwap after a load was this staging plus the deepcopy'd index,
        # and the owner cannot run swap off while the server keeps asking for it.
        closed = ""
        try:
            config.stc.close()
            closed = "ok"
        except Exception as e:  # noqa: BLE001 - a live collection is not a reason to lose the load
            closed = f"{type(e).__name__}: {e}"
        del model, config
        released = _trim_host()
        host_rss, host_swap = _host_rss_swap_mib()
        host_anon = _proc_status_mib(("RssAnon",))[0]
        print(f"SEAM_HOST_MEM phase=load_done rss_mib={host_rss} swap_mib={host_swap} "
              f"anon_mib={host_anon} trim_mib={released} stc_close={closed} avail_mib={_avail_mib()} "
              f"box_swap_mib={_swap_used_mib()}", flush=True)
        # Which mappings still hold the host bill, biggest first: an empty list means the trim
        # returned everything and the pack is the only thing the box gave up.
        for size_mib, swap_mib, rss_mib in _anon_maps():
            print(f"SEAM_HOST_MAP size_mib={size_mib} swap_mib={swap_mib} "
                  f"rss_mib={rss_mib}", flush=True)
        cx_n = sum(1 for v in self._cx_layers.values() if v is not None)
        if self._cx_layers:
            print(f"SEAM_PACK_KERNEL cudaexl3_layers={cx_n} mixedk_layers={NL - cx_n} "
                  f"of={NL} avail_mib={_avail_mib()} avail_min_mib={getattr(self, '_cx_avail_min', '?')} "
                  f"swap_mib={_swap_used_mib()} rss_peak_mib={_rss_peak_mib()}", flush=True)

    def _check_placement_groups(self, layer, cold, hit):
        """Assert the ranked file's cold order against the bitrate the shard was cut at.

        `hit.groups` carries each expert's bitrate and its position in `cold`. The file promises
        `cold[:warm_4bpw_n]` are exactly the 4-bit experts; a file reordered against a pack cut
        from another table would still be self-consistent -- the client numbers its cold ids from
        the same file, so nothing downstream can tell -- which is why the check lives here, where
        the shard's own geometry is in hand.
        """
        bpw_file = self.layer_bpw_file.get(layer)
        n_file = self.layer_4n_file.get(layer)
        if not bpw_file or n_file is None:
            return
        got = [None] * len(cold)
        for grp in hit.groups:
            for pos in grp.source:
                got[int(pos)] = int(grp.bits)
        wrong = [i for i, b in enumerate(got) if b != bpw_file[i]]
        n4 = sum(1 for b in got if b == 4)
        if wrong or n4 != n_file:
            print(f"SEAM_PLACEMENT_GROUPS layer={layer} MISMATCH n={len(cold)} "
                  f"warm_4bpw_n={n_file} shard_4bpw={n4} "
                  f"file={bpw_file[:n_file] + bpw_file[n_file:]} "
                  f"shard={got} positions={wrong[:8]}", flush=True)
            raise SystemExit(
                f"placement/shard mismatch on layer {layer}: the file's warm order does not "
                f"describe this pack (warm_4bpw_n={n_file}, {n4} 4-bit experts in the shard, "
                f"{len(wrong)} cold positions at the wrong bitrate); refusing to load a pack "
                f"under a table that numbers its experts differently")
        print(f"SEAM_PLACEMENT_GROUPS layer={layer} n={len(cold)} warm_4bpw_n={n_file} "
              f"groups4={n4} groups2={len(cold) - n4} ok=1", flush=True)

    def compute(self, layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr):
        """Pinned-slab pointers in; the answer written into the pinned response region in the
        wire's reply dtype (fp32, or bf16/fp16 under SEAM_QES_OUT; see OUT_MODE).

        Returns 0 when the response region holds this request's answer, or a refusal code (see
        REFUSE_MEM / REFUSE_COMPUTE). The C proxy turns a nonzero return into "poison this slot,
        tell the client which of the two it was, end this one session" and goes back to accepting
        (qmesh_expert_seam.h). The pack stays resident: a request this box cannot serve costs the
        client one failed call, not a 150-220 s reload for everyone.

        Never raise. The C proxy posts the response only after this hook returns, so an exception
        allowed to escape is swallowed by ctypes (it prints "Exception ignored") and the *previous*
        call's bytes in the pinned response region get posted as this call's answer. That is exactly
        what happened on GX2 on 2026-11-01 08:57: a CUDA launch failure inside _forward_piece left
        the context permanently broken, 170k callbacks raised, the server kept accepting sessions for
        20 min and the client decoded garbage ("magnetic magnetic ...").

        So a failure is either "this request cannot be computed" - refused, process alive, client
        gets NaN and a named error - or "this process can never answer again", which is the only
        case that still exits: a dead context fails every later session the same way, and a server
        that cannot compute must not pretend to. _cuda_ok decides.
        """
        # Per-call device time, reported every minute as SEAM_COMPUTE (median/mean/max us per row
        # bucket). Paired with the client's round-trip timing this splits the seam into GX1 compute
        # and wire: wire = round trip - compute.
        t0 = time.perf_counter_ns()
        self.seen += 1
        avail = _avail_mib()
        if 0 <= avail < FWD_FLOOR_MIB:
            # Cheap, no device work: return the allocator's unused blocks, then look again. Only a
            # box that is still short after that is refused before a launch it would pay for in
            # swap storms and launch failures.
            if not self.stub:
                _free_cached()
            avail = _avail_mib()
        if 0 <= avail < FWD_FLOOR_MIB:
            self.refusals += 1
            print(f"SEAM_COMPUTE_REFUSED layer={layer} rows={rows} why=memory "
                  f"need_mib={FWD_FLOOR_MIB} avail_mib={avail} swapped_mib={_swap_used_mib()} "
                  f"dev_{_dev_mem_mib()} "
                  f"calls={sum(len(v) for v in _times.values())} refusals={self.refusals} "
                  f"- answering nothing, the session ends, the server keeps listening", flush=True)
            return REFUSE_MEM
        try:
            # Exactly call N+1, not every call after it: the refusal must leave a process that
            # still answers, and the only way to show that from one server is a later call that
            # is served normally.
            if self.fail_after and self.seen == self.fail_after + 1:
                raise RuntimeError(f"injected compute failure on call {self.seen} (--fail-after "
                                   f"{self.fail_after})")
            # The lock covers the whole forward, not just one piece: a request served in ws_rows
            # pieces still builds its scratch between pieces, and the pieces of two requests must
            # not interleave inside the module.
            with self.compute_lock:
                self._forward(layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
        except Exception as e:  # noqa: BLE001 - see the docstring: return, never raise
            # The message names the *calling* shape; only the traceback names the frame that raised.
            # A CUDA illegal access surfaces at the next host sync (a `.tolist()`, a copy to host),
            # so the frame is regularly several layers away from the kernel that faulted - without
            # this, "rows=6" is the only evidence and it can be the wrong call entirely.
            import traceback
            trace = [ln for ln in traceback.format_exc().rstrip().splitlines()]
            # A sticky launch failure is sticky: the probe itself raises and the context is gone.
            if self.stub:
                # A stub has no CUDA context that could be dead, and asking for one would create
                # it - up to half a GiB of the same headroom the resident pack on this box needs.
                usable = True
            else:
                _free_cached()
                usable = _cuda_ok()
            self.refusals += 1
            print(f"SEAM_COMPUTE_{'REFUSED' if usable else 'FAIL'} layer={layer} rows={rows} "
                  f"err={type(e).__name__}: {e} "
                  f"avail_mib={_avail_mib()} swapped_mib={_swap_used_mib()} "
                  f"dev_{_dev_mem_mib()} "
                  f"calls={sum(len(v) for v in _times.values())} refusals={self.refusals} "
                  f"cuda_ok={int(usable)} "
                  + ("- answering nothing, the session ends, the server keeps listening"
                     if usable else
                     "- exiting: the CUDA context cannot run a kernel any more, so no later "
                     "session could be served"), flush=True)
            for line in trace:
                print(f"SEAM_COMPUTE_TRACE {line}", flush=True)
            if not usable:
                sys.stderr.flush()
                os._exit(70)
            return REFUSE_COMPUTE
        with _lock:
            _times.setdefault(int(rows), []).append((time.perf_counter_ns() - t0) / 1e3)
        return 0

    def _forward(self, layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr):
        """Serve one request of `rows` rows, in internal pieces of at most ws_rows.

        The row bucket and the device workspace are separate bills and this is what decouples them.
        Measured on the dev pair (--ws-probe, pack resident): the cold-expert forward's scratch is
        ~150 KiB per row, so a bucket of N rows costs 150*N KiB of transient device memory however
        the slab is sized - 301 MiB at 2048 rows, 602 at 4096, 1202 at 8192, against 733 MiB of
        headroom on GX2 and 876 on GX1. The scratch scales with the rows handed to the module, not
        with the number of requests, so serving a big bucket as several pieces makes the peak
        depend on the piece size instead. Pieces run in order on the same CUDA stream and the
        single synchronise at the end still covers them; the pinned slab and the wire cost are
        unchanged, which is the point: a bigger bucket then buys fewer per-layer calls
        (remote_seam_host chunks a prefill at the largest granted bucket) without a bigger
        workspace.

        The piece size itself is a bandwidth/footprint trade (WS_LADDER): every piece re-reads the
        layer's whole cold-expert weight set, so small pieces multiply the weight traffic, and
        pieces large enough not to fit turn the box into a swap storm. ws_rows is the ceiling and
        _pick_step takes the largest rung under it that MemAvailable pays for right now.
        """
        if self.stub:
            # Known-answer stub: the same out[r][j] = float(x[r][j]) + (layer + 1) the C proxy
            # computes when no hook is installed. Zeros would make a dead response path
            # indistinguishable from a correct answer - on 2026-11-01 a zero-answering stub hid a
            # full day of "is the payload arriving at all?" behind `finite=True` - so a stub answer
            # must depend on both the request bytes and the layer index.
            xb = view(x_ptr, rows * HIDDEN * 2, np.uint16).reshape(rows, HIDDEN)
            xf = (xb.astype(np.uint32) << 16).view(np.float32)
            _out_view(out_ptr, rows).copy_(torch.from_numpy(xf + (layer + 1)))
            return
        ffn, _ = self.layers[layer]
        step = self._pick_step(rows)
        for r0 in range(0, rows, step):
            self._forward_piece(layer, ffn, min(step, rows - r0),
                                x_ptr + r0 * HIDDEN * 2, ids_ptr + r0 * TOPK * 4,
                                w_ptr + r0 * TOPK * 2, out_ptr + r0 * HIDDEN * OUT_ELEM)

    def _pick_step(self, rows):
        """Rows per piece for one request: the largest ladder rung under ws_rows the box can pay.

        ws_rows <= 0 means "never split", whatever the workspace costs - the escape hatch the
        ws-probe/ws-check modes and a deliberate one-piece A/B rely on. Above that the ceiling is
        honoured and the rung is otherwise chosen from MemAvailable, because the failure this
        guards against is asymmetric: picking one rung too big cost 4.3 s per call on GX2 (8192
        rows, 0 MiB free, 1.3 GiB of swap written during ten calls), while picking one too small
        costs a weight pass (~12 ms per 2048 rows).
        """
        if self.ws_rows <= 0:
            return rows
        if not self.ws_auto:
            return min(self.ws_rows, rows)
        avail = _avail_mib()
        # A box that is already swapping cannot be talked into a bigger piece by free memory: GX2,
        # 3.9 GiB swapped out, ran 4 x 2048-row pieces of layer 0 at p50 105 ms but mean 172 ms
        # (9 of 10 calls did swap I/O) - worse than the 153 ms mean of 8 x 1024 on the same box -
        # while the same code with MemAvailable back at 1225 ran those pieces at mean 115 ms. The
        # swap in use, not MemAvailable, is the discriminator, so a paging box stays on the floor
        # rung; MemAvailable still decides between the rungs above it.
        swapped = _swap_used_mib()
        paging = swapped > SWAP_WARN_MIB
        # Step down the instant the current rung stops fitting, but climb back only with
        # WS_HYST_MIB to spare: on GX1 a resident pack leaves MemAvailable wandering around the
        # 2048-row threshold (542 MiB available against 566 needed), and a picker that followed
        # that noise would alternate 2048 and 1024 pieces call after call for no gain. Being slow
        # on the way up is the safe direction - a rung too small only costs the weight pass the
        # split already paid.
        prev = self.__dict__.get("_ws_pick")
        pick = None
        if not paging:
            for n in WS_LADDER:
                if n > self.ws_rows:
                    continue
                slack = 0 if (prev is None or n <= prev) else WS_HYST_MIB
                if avail < 0 or ws_mib_for(rows, n) + WS_GUARD_MIB + slack <= avail:
                    pick = n
                    break
        if pick is None:
            # Nothing fits: take the smallest rung rather than hand the box a request it will
            # swap through. A caller that reaches this on every call wants a smaller bucket.
            pick = min(self.ws_rows, WS_LADDER[-1])
        if pick != self.__dict__.get("_ws_pick"):
            self._ws_pick = pick
            print(f"SEAM_WS_PICK rows={rows} ws={pick} avail_mib={avail} swapped_mib={swapped} "
                  f"need_mib={ws_mib_for(rows, pick) + WS_GUARD_MIB:.0f} "
                  f"pieces={(rows + pick - 1) // pick}", flush = True)
        return min(pick, rows)

    def _forward_piece(self, layer, ffn, rows, x_ptr, ids_ptr, w_ptr, out_ptr):
        """One mixed-K forward of `rows` rows straight out of, and back into, the pinned slab."""
        # Same argument on the way in: the C proxy owns these bytes until this hook returns, so the
        # H2D can read them in place instead of staging a rows*HIDDEN*2 copy (117 MiB at 8192 rows).
        x = torch.from_numpy(view(x_ptr, rows * HIDDEN * 2, np.uint16).reshape(rows, HIDDEN))
        x = x.view(torch.bfloat16).to(self.device).half()
        ids_np = view(ids_ptr, rows * TOPK * 4, np.int32).reshape(rows, TOPK)
        _check_ids(layer, rows, ids_np, self.layer_n[layer])
        ids = torch.from_numpy(ids_np.astype(np.int64))
        w = torch.from_numpy(view(w_ptr, rows * TOPK * 2, np.uint16).reshape(rows, TOPK).copy())
        w = w.view(torch.bfloat16).float()

        miss = ids < 0
        sel = ids.clamp_min(0).to(self.device)
        gw = torch.where(miss, torch.zeros_like(w), w).to(self.device).half()
        # Cold-expert kernel switch (MOE_KERNEL). cuda-exl3 wants a layer's experts of one bitrate
        # stacked into one trellis instead of one module per expert; a direct-loaded layer is
        # already in that layout (`SEAM_MOE_DIRECT`), and any other layer is restacked on its first
        # call, which drops the per-expert copies so the resident cost is one copy of the cold pack
        # and not two. A layer that cannot restack stays on the module.
        cx = self._cx_moe(layer, ffn) if MOE_KERNEL == "cudaexl3" else None
        if cx is None and ffn is None:
            # Unreachable by construction: a layer has either its module or its direct stack.
            # Serving the previous call's bytes instead would look like a correct answer (see
            # compute()), so name it and die.
            raise RuntimeError(f"layer {layer}: no cold-expert kernel and no module to serve it")
        with torch.cuda.stream(self.stream):
            if cx is not None:
                part = cx.forward(x, sel, gw)
            else:
                part = ffn.forward(x.view(1, rows, HIDDEN),
                                   {"external_selected": sel, "external_weights": gw})
        self.stream.synchronize()
        # Device -> the client's pinned slab, in place. The obvious form of this line --
        # `slab[:] = part.float().cpu().numpy()` -- allocates a fresh rows*HIDDEN*4 CPU tensor per
        # call (235 MiB at 8192 rows) and glibc keeps those mmaps in its arenas long after the
        # copy, so a prefill paid ~50 GiB of host churn on a box whose headroom is measured in
        # hundreds of MiB. Wrapping the slab with from_numpy makes it the destination itself. The
        # copy_ rounds to the wire's reply dtype (RN) when SEAM_QES_OUT selects a 16-bit reply.
        _out_view(out_ptr, rows).copy_(part.view(rows, HIDDEN))

    def _cx_moe(self, layer, ffn):
        """The layer's cuda-exl3 adapter, built on first use (see _forward_piece).

        None means "this layer stays on the mixed-K module", and it is cached as such: a restack
        that failed once (an unstackable bitrate/frac mix, a broken extension) would fail on every
        call, and the client must not pay that exception per request.
        """
        cx = self.__dict__.setdefault("_cx_layers", {})
        if layer in cx:
            return cx[layer]
        if ffn is None:
            # A direct-loaded layer is already converted and was cached before the load moved on,
            # so a miss here means its module is gone and nothing replaced it. There is nothing to
            # restack; name it rather than raising on `_cx_transient_mib(None)`.
            cx[layer] = None
            print(f"SEAM_MOE_NOLAYER layer={layer} - no module and no stacked copy", flush=True)
            return None
        try:
            from qmesh.exl3_moe_cuda import (CudaExl3Moe, RestackBroken, transient_mib as
                                             _cx_transient_mib, _live_mib as _cx_cold_mib,
                                             peak_mib as _cx_peak_mib)
        except Exception as e:
            # No vendored extension on this box: the layer keeps its module copy, and saying so once
            # is better than dying inside compute()'s fail-stop on the first request.
            cx[layer] = None
            print(f"SEAM_MOE_NOEXT layer={layer} err={type(e).__name__}: {e} "
                  f"- serving this layer from the mixed-K module", flush=True)
            return None
        # One group's stacked tensors are allocated before any expert of that group is copied into
        # them, and the group's sources are freed before the next group starts, so a conversion on
        # an already-resident layer needs exactly `transient_mib` of free memory, not the layer's
        # size plus that (the sources are resident and already excluded from MemAvailable). On GB10
        # the pack and the device share one 121 GiB pool, and a cudaMalloc that cannot be satisfied
        # there does not fail cleanly: it pages the pack out and wedges the host (GX1 2026-11-01
        # 16:44, needing 5.4 GiB against 1.32 GiB available). Refuse up front when the transient
        # plus the floor does not fit, so the layer stays on the mixed-K module and the box lives.
        need = _cx_transient_mib(ffn)
        avail = _avail_mib()
        # Lowest MemAvailable seen at any conversion, reported with the load summary: a refusal or a
        # near miss during the load is then evidence about the host (another process, page cache),
        # not about the layer, which always costs the same at the same point in the load.
        if avail >= 0:
            prev = self.__dict__.get("_cx_avail_min", avail)
            self._cx_avail_min = min(prev, avail)
        if avail >= 0 and need + CX_MARGIN_MIB > avail:
            cx[layer] = None
            print(f"SEAM_MOE_NOFIT layer={layer} need_mib={need:.0f} avail_mib={avail} "
                  f"margin_mib={CX_MARGIN_MIB} cold_mib={_cx_cold_mib(ffn):.0f} "
                  f"peak_mib={_cx_peak_mib(ffn):.0f} "
                  f"- restack would page the pack out; serving this layer from the mixed-K module",
                  flush=True)
            return None
        t0 = time.perf_counter()
        avail0, swap0 = _avail_mib(), _swap_used_mib()
        try:
            hit = CudaExl3Moe(ffn, self.device)
        except RestackBroken as e:
            # The streaming restack frees each expert's source as it lands in the stack, so a
            # failure after the first group leaves this layer with no per-expert copy to fall back
            # on. Serving it from the mixed-K module would answer with stale or partial bytes, which
            # is the failure mode this server is fail-stop about (see compute()): name it and die.
            raise RuntimeError(f"cuda-exl3 restack of layer {layer} failed after source tensors "
                               f"were released ({e}); the layer has no fallback copy") from e
        except Exception as e:
            cx[layer] = None
            print(f"SEAM_MOE_FALLBACK layer={layer} err={type(e).__name__}: {e} "
                  f"- serving this layer from the mixed-K module", flush=True)
            return None
        torch.cuda.empty_cache()
        cx[layer] = hit
        print(f"SEAM_MOE_CX layer={layer} groups={hit.describe()} "
              f"stacked_mib={hit.nbytes() / 2**20:.0f} build_ms={1e3 * (time.perf_counter() - t0):.0f} "
              f"avail_mib={avail0}->{_avail_mib()} swap_mib={swap0}->{_swap_used_mib()} "
              f"rss_peak_mib={_rss_peak_mib()} cached_mib={_cached_mib()} "
              f"alloc_gib={torch.cuda.memory_allocated(self.device) / 2**30:.2f}", flush=True)
        return hit


_times: dict = {}
_lock = threading.Lock()


def _reporter():
    while True:
        time.sleep(60)
        with _lock:
            if not _times:
                continue
            snap = {k: v for k, v in _times.items()}
            _times.clear()
        for rows in sorted(snap):
            v = sorted(snap[rows])
            n = len(v)
            # avail_mib is the box state those calls were served in, printed here because this is the
            # only place a *served* call reports it: setting SEAM_FWD_FLOOR_MIB needs the avail of
            # served calls next to the avail of a refusal, and taking it from the load line or from
            # an idle box describes a state no forward ever ran in.
            print(f"SEAM_COMPUTE rows={rows} n={n} p50_us={v[n // 2]:.1f} "
                  f"mean_us={sum(v) / n:.1f} max_us={v[-1]:.1f} "
                  f"avail_mib={_avail_mib()} swapped_mib={_swap_used_mib()}", flush=True)


def view(addr, nbytes, dtype):
    """numpy view over raw pinned-slab memory (no copy)."""
    import ctypes
    buf = (ctypes.c_byte * nbytes).from_address(addr)
    return np.frombuffer(buf, dtype=dtype)


def slab_mib(buckets, slots = 1) -> float:
    """Pinned slab this server will allocate for `buckets`, MiB, one side.

    Same geometry as geometry() in qmesh/native/qmesh_expert_seam.cu (protocol v5): one slab per
    row bucket holding `slots` payload slots, each
    [64 B F_REQ][64 B F_RESP][64 B request header] + align64(rows*(HIDDEN*2 + TOPK*6)) request +
    align64(rows*HIDDEN*OUT_ELEM) response bytes (4 B fp32, 2 B under the 16-bit wire caps). Slots
    are in-flight requests, not layers: the layer id rides in the request header, so the whole
    staging cost is ~30.2 KiB per row per slot with the fp32 reply and ~20.2 KiB with a 16-bit one."""
    al = lambda v: (v + 63) // 64 * 64  # noqa: E731
    total = 0
    for rows in buckets:
        req = rows * (HIDDEN * 2 + TOPK * 4 + TOPK * 2)
        resp = rows * HIDDEN * OUT_ELEM
        total += (3 * 64 + al(req) + al(resp)) * slots
    return total / 2**20


# Host memory one piece of `rows` needs on top of the pack, KiB per row. Two measurements, both on
# a box with the 4460 pack resident: --ws-probe reported 301 MiB of transient device memory at 2048
# rows (147 KiB/row), but the host's MemAvailable fell by ~467 MiB across a window of ten 2048-row
# calls on GX2, i.e. ~233 KiB/row once the staging the device side shares with the host is counted.
# MemAvailable is what the picker can read, so the host figure is the one that governs; 235 KiB/row
# still clears the 2048-row rung on GX2 with a full pack (566 MiB against ~1220 available) and
# leaves it unreachable on GX1 (461-653 available, the box is already 3 GiB into swap).
WS_KIB_PER_ROW = 235

# Piece-size ladder for _forward. A request served in N pieces makes N passes over the layer's
# resident cold experts, so the per-row cost falls with piece size until the transient workspace
# stops fitting. Measured on GX2 with the 4460 pack resident, one layer at a time, p50 per row:
#   8192 rows  layer 0 : 1024 -> 18.5 us, 2048 -> 12.8 us, 4096 -> 27.4 us (+3.2 s p99)
#   8192 rows  layer 20: 1024 -> 19.9 us, 2048 -> 13.8 us, 4096 -> 39.8 us (+4.8 s p99)
#   4096 rows  layer 0 : 1024 -> 28.1 us, 2048 -> 12.6 us
#   4096 rows  layer 20: 1024 -> 19.9 us, 2048 -> 13.5 us
# and a single 8192-row piece needs 4.3 s a call and wrote 1.3 GiB of swap in ten calls. So 2048 is
# both the default ceiling and the top rung, and 1024 is the floor: GX1 measured 220 ms per 8192-row
# call in 512-row pieces against 154-169 in 1024s, so a 512 rung only ever bought a slower box.
WS_LADDER = (2048, 1024)
# Headroom the picker insists on beyond the piece's own workspace: the piece's transient is not the
# only thing racing for the last GiB (slab, CUDA context, the page cache of a 113.65 GiB pack), and
# being one rung too big costs a 4x stall, not 10%.
WS_GUARD_MIB = int(os.environ.get("SEAM_WS_GUARD_MIB", "96"))
# Extra headroom demanded before a request moves *up* a rung, so a box sitting near a threshold
# holds its choice instead of alternating piece sizes call after call.
WS_HYST_MIB = int(os.environ.get("SEAM_WS_HYST_MIB", "128"))
# SEAM_WS_ROWS_AUTO=0 pins every request to exactly ws_rows: for A/B runs that need the old
# behaviour, and for --ws-check/--ws-probe/--rows-bench, which set ws_rows themselves.
WS_AUTO = os.environ.get("SEAM_WS_ROWS_AUTO", "1") != "0"

# Swap in use above this and the resident pack is paging; see _swap_used_mib and
# SEAM_SERVER_SWAP_WARN in main(). A few hundred MiB is normal noise on these boxes, a few GiB is
# a box that is already over its head before the first request. The piece-size picker refuses to
# leave the floor rung while this is exceeded, so a box whose swap is stale rather than active
# raises it (SEAM_SWAP_WARN_MIB=N) instead of pinning the ladder for the rest of the run.
SWAP_WARN_MIB = int(os.environ.get("SEAM_SWAP_WARN_MIB", "512"))

# Floor a *restack* has to leave on the box besides its own peak (see _cx_moe). 1.5 GiB is the
# headroom a loaded GX actually has left (slab + workspace + CUDA context); a conversion that eats
# into it swaps the 113.65 GiB pack out page by page, and on GB10 that stalls the whole host, not
# just this process (GX1, 2026-11-01 16:44). It is a per-layer refusal test, not a global policy:
# layer N is restacked when avail - need >= margin, so N refuses every layer whose need + N exceeds
# avail, and the count of refused layers moves with N (measured on GX2, 2026-11-02, 4460 pack:
# N=1536 leaves 2, N=2600 leaves 3; the layers are 37-39, the last ~2.7-3.1 GiB of a 113.65 GiB load).
# Since SEAM_MOE_DIRECT (see exl3_moe_cuda.from_pack) the normal load path does not restack at all --
# it reads each layer straight out of the pack, needs no transient, and never consults this floor --
# so the margin now only gates the per-layer fallback taken when from_pack fails. Set
# SEAM_CX_MARGIN_MIB=0 only to measure a restack on a box you are willing to lose.
CX_MARGIN_MIB = int(os.environ.get("SEAM_CX_MARGIN_MIB", "1536"))

# Convert each cold layer to the cuda-exl3 kernel as it loads, on by default when that kernel is
# the one in use. The conversion needs one layer's cold bytes *plus* its largest bitrate group
# transiently (see exl3_moe_cuda._stack_group); with the full pack resident a GX has ~1.0 GiB
# MemAvailable and a 2.74 GiB layer therefore cannot convert, while the same layer costs nothing
# during the load because MemAvailable is still 116 GiB there (GX2, 2026-11-01: avail_mib 120079 ->
# 1000 over 40 layers, 116223 right after layer 0). Deferring the restack to the first request is
# what wedged GX1. SEAM_MOE_PRELOAD=0 restores that behaviour for a box measured to have slack.
PRELOAD_CX = os.environ.get("SEAM_MOE_PRELOAD", "1") != "0"

# Read each cold layer straight from its pack shard into the stacked layout instead of loading one
# module per expert and restacking those (see CudaExl3Moe.from_pack). On, because the module path's
# conversion needs a transient the last layers of a 113.65 GiB pack on a 121.6 GiB box do not have:
# GX2 2026-11-02 refused layers 37-39 (`SEAM_MOE_NOFIT`) and landed on 37/3. SEAM_MOE_DIRECT=0 keeps
# the module path for an A/B on one box.
DIRECT_PACK = os.environ.get("SEAM_MOE_DIRECT", "1") != "0"


def ws_mib_for(rows: int, ws_rows: int = 0) -> float:
    """Transient device workspace the cold-expert forward needs for a request, MiB."""
    n = ws_rows if 0 < ws_rows < rows else rows
    return n * WS_KIB_PER_ROW / 1024


def _avail_mib() -> int:
    """Host MemAvailable, MiB: what is left on this box for the slabs (see seam_auto_buckets)."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def _rss_peak_mib() -> int:
    """VmHWM of this process, MiB: the peak the restack drove the resident pack to."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def _cached_mib() -> int:
    """Cached (page cache) on this box, MiB: what a conversion can borrow before the pack swaps."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("Cached:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def _swap_used_mib() -> int:
    """Swap currently in use, MiB. On a GX this is the early warning that matters: the pack is
    113.5 GiB of a 121.6 GiB unified pool, so a few GiB swapped out means every expert call takes
    page faults (seen as 4 s outliers in SEAM_COMPUTE) and a launch can fail outright."""
    total = free = 0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("SwapTotal:"):
                total = int(line.split()[1])
            elif line.startswith("SwapFree:"):
                free = int(line.split()[1])
    except OSError:
        return -1
    # The return was lost when release 1 was rebased onto proto-3: the function fell through to
    # None, SEAM_SERVER_LOADED/SLAB printed swapped_mib=None and main()'s `if swapped_mib >
    # SWAP_WARN_MIB` raised TypeError. On a stub that kills the server before SEAM_LISTEN; on a real
    # load it kills it *after* the 153 s pack load. Caught by the 2026-11-01 failstop slot.
    return (total - free) // 1024


def _proc_status_mib(fields) -> tuple:
    """`(VmRSS, VmSwap, ...)` of this process, MiB, -1 per unreadable field."""
    seen = {}
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                key = line.split(":", 1)[0]
                if key in fields:
                    seen[key] = int(line.split()[1]) // 1024
    except OSError:
        return tuple(-1 for _ in fields)
    return tuple(seen.get(key, -1) for key in fields)


def _host_rss_swap_mib() -> tuple:
    """(resident, swapped) host memory of this process, MiB.

    On a GX the pack is device memory of a unified pool, so these two numbers are the whole host
    bill the server has any business holding: the pinned client slabs plus a live interpreter. A
    load that ends with GiBs here has left Python/glibc staging behind (see _trim_host)."""
    return _proc_status_mib(("VmRSS", "VmSwap"))


def _trim_host() -> int:
    """Hand freed CPU staging back to the OS. Returns MiB released (0 = nothing was stale).

    The loader reads every tensor of a ~2.9 GiB safetensors file through a CPU buffer and copies it
    into the unified pool; Python and glibc free those buffers into their arenas, but glibc only
    returns an arena to the kernel when it is trimmed, so the process keeps GiBs of dead staging -
    which the kernel evicts to swap as soon as the pack fills the pool (GX1 2026-11-02:
    3.44 GiB VmSwap, 2.88 GiB of it one anonymous mapping, 0.76 GiB resident). gc.collect() drops
    the Python objects, malloc_trim(0) releases the arenas; together they are the difference
    between "the box has headroom" and "every expert call takes page faults"."""
    before = _host_rss_swap_mib()
    gc.collect()
    try:
        _LIBC.malloc_trim(0)
    except Exception:  # noqa: BLE001 - a libc without malloc_trim must not fail a load
        pass
    after = _host_rss_swap_mib()
    # A mapping's swap entries go with the mapping, so this delta is memory the box really has back.
    return max(0, (before[0] + before[1]) - (after[0] + after[1]))


def _anon_maps(min_mib: int = 64, top: int = 4) -> list:
    """Largest anonymous mappings of this process, [(size_mib, swap_mib, rss_mib), ...].

    This is the answer to "why is a loaded server in swap": the pack is device memory, so any big
    anonymous mapping here is host staging that survived the load. Printed right after _trim_host
    so a load that still keeps GiBs names the mapping instead of costing another reload to find."""
    maps = []
    size = rss = swap = None
    try:
        with open("/proc/self/smaps") as fh:
            for line in fh:
                if "-" in line.split(" ", 1)[0] and len(line.split()) >= 5:
                    if size is not None and size > min_mib * 1024 and (swap or rss):
                        maps.append((size // 1024, swap // 1024, (rss or 0) // 1024))
                    # Anonymous = no pathname (6th field) and not a device mapping (/dev/..., [heap]).
                    fields = line.split()
                    anon = len(fields) == 5
                    size = 0 if anon else None
                    rss = swap = 0
                elif size is not None:
                    key = line.split(":", 1)[0]
                    if key == "Size":
                        size = int(line.split()[1])
                    elif key == "Rss":
                        rss = int(line.split()[1])
                    elif key == "Swap":
                        swap = int(line.split()[1])
    except OSError:
        return []
    maps.sort(reverse=True)
    return maps[:top]


def _drop_page_cache(path: str) -> bool:
    """Evict one pack file's page cache after its experts are on the device.

    The loader mmaps each cold-L<NN>.safetensors and copies it into the arena; the mmapped pages
    stay in the page cache. On GB10 (121.6 GiB host and device memory are one pool) that cache
    competes with the experts themselves: with the pack resident, 2.9 GiB of stale cache per layer
    is the difference between headroom for a 4096-row workspace and a swap storm. posix_fadvise
    drops only this file's pages (no root, no global drop_caches, which this container cannot
    write: /proc/sys/vm is read-only here)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def _dev_mem_mib() -> str:
    """`free_mib=/total_mib=/allocated_mib=/reserved_mib=` for the unified pool, best effort.

    Host MemAvailable alone cannot say whether the *device* side has room: on GB10 device and host
    memory are one pool, so once the 113.65 GiB pack is resident a cold-expert forward can still
    fail a launch while MemAvailable shows ~1 GiB and several GiB are swapped out. Printing the
    allocator's own view at load and at any SEAM_COMPUTE_FAIL turns "unspecified launch failure"
    into a number: free_mib near zero = the pack does not fit this box, not a kernel bug."""
    try:
        free, total = torch.cuda.mem_get_info()
        return (f"free_mib={int(free) // 2**20} total_mib={int(total) // 2**20} "
                f"allocated_mib={torch.cuda.memory_allocated() // 2**20} "
                f"reserved_mib={torch.cuda.memory_reserved() // 2**20}")
    except Exception:  # noqa: BLE001 - diagnostics must never be the thing that fails
        return "free_mib=-1 total_mib=-1 allocated_mib=-1 reserved_mib=-1"


def _set_tier(server, limit):
    """Force the tier split of a measurement: every layer module's fused-kernel row capacity is
    also its dispatch threshold, so experts above it take the batched reconstruct tier
    (moe_batch_recon) and experts at or below it the fused kernel. The buffers were sized at load
    (FUSED_ROWS_WIDE / TEMP_ROWS_FUSED), so this changes routing only: the memory bill, the pack
    and the process stay as they are, which is what makes a back-to-back A/B on identical input
    possible. Returns the number of modules changed."""
    n = 0
    for ffn, _ncold in server.layers.values():
        if hasattr(ffn, "fused_rows"):
            ffn.fused_rows = int(limit)
            n += 1
    return n


def _probe_stage(rows, ncold):
    """Pinned request slab for a probe: finite 1.0 in bf16 x and weights and in-range expert ids,
    so the probe exercises the same routing/dispatch path a real chunk does (zeros would hide a
    piece that came back empty). Returns (stage, x_ptr, ids_ptr, w_ptr, out_ptr).

    The probes and their readers are written for the fp32 reply; under a 16-bit SEAM_QES_OUT they
    would read halves as floats, so they refuse rather than measure."""
    if OUT_MODE != 0:
        raise RuntimeError(f"probes assume the fp32 reply; run them without SEAM_QES_OUT "
                           f"(out={qes.OUT_NAMES[OUT_MODE]})")
    lane = rows * (HIDDEN * 2 + TOPK * 4 + TOPK * 2) + rows * HIDDEN * 4
    stage = torch.zeros(lane, dtype = torch.uint8, pin_memory = True)
    base = stage.data_ptr()
    x_ptr = base
    ids_ptr = base + rows * HIDDEN * 2
    w_ptr = ids_ptr + rows * TOPK * 4
    out_ptr = w_ptr + rows * TOPK * 2
    view(x_ptr, rows * HIDDEN * 2, np.uint16).reshape(rows, HIDDEN)[:] = 0x3c00
    view(ids_ptr, rows * TOPK * 4, np.int32).reshape(rows, TOPK)[:] = \
        np.arange(rows * TOPK, dtype = np.int32).reshape(rows, TOPK) % max(ncold, 1)
    view(w_ptr, rows * TOPK * 2, np.uint16).reshape(rows, TOPK)[:] = 0x3c00
    return stage, x_ptr, ids_ptr, w_ptr, out_ptr


def _probe_run(server, rows, ptrs, tag):
    """Warm once, then time one call. Returns us or None after reporting the failure."""
    x_ptr, ids_ptr, w_ptr, out_ptr = ptrs
    dev = server.device
    try:
        # Warm: the first call of a tier allocates/locates that tier's buffers and kernels, and
        # leaving it in the timed call is what made the 2048 rung read 472 us/row instead of 140.
        server._forward(0, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
        torch.cuda.synchronize(dev)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"SEAM_WS rows={rows} tier={tag} OOM warmup err={str(e).splitlines()[0][:200]}",
              flush = True)
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter_ns()
    try:
        server._forward(0, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
        torch.cuda.synchronize(dev)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"SEAM_WS rows={rows} tier={tag} OOM err={str(e).splitlines()[0][:200]}", flush = True)
        return None
    return (time.perf_counter_ns() - t0) / 1e3


# --rows-bench times a request as one piece up to this many rows by default: past it the transient
# workspace (147 KiB/row, 1202 MiB at 8192 rows) is bigger than the headroom a loaded GX has, and
# the call swaps instead of computing.
WHOLE_MAX = int(os.environ.get("SEAM_ROWS_BENCH_WHOLE_MAX", "4096"))

def _vmstat_swap():
    """(pswpin, pswpout) in pages, straight out of /proc/vmstat.

    A pack that has been pushed to swap still "works": the calls fault the pages back in and the
    only symptom is that every forward is slower than the same forward on an idle box. Measuring
    si/so around the timed calls is what separates "the kernel is slow here" from "the box is
    paging", and the headroom question hangs on that difference."""
    pin = pout = 0
    try:
        for line in open("/proc/vmstat"):
            if line.startswith("pswpin "):
                pin = int(line.split()[1])
            elif line.startswith("pswpout "):
                pout = int(line.split()[1])
    except OSError:
        return -1, -1
    return pin, pout


def rows_bench(server, spec, iters, layers = "0", warmup = 2, ws_list = ""):
    """Time the real cold-expert forward at several row counts on the resident pack, no client.

    The seam's SEAM_COMPUTE p50 is what this measures plus the wire; this mode removes the wire
    (and the P7, and the slab copies' variability) and runs the exact `_forward` the server runs,
    so a per-row number taken with the full pack resident can be compared directly with a
    one-layer bench. Each row count is timed as one piece and, when --ws-rows or --rows-bench-ws
    is set, in pieces of those sizes too - the split is a separate bill. Swap traffic and
    MemAvailable are reported around each timed window.
    """
    if server.stub:
        print("SEAM_ROWS_BENCH stub=1: nothing to measure without the cold experts", flush = True)
        return
    # The bench sets the piece size itself, rung by rung; the per-call picker would override it.
    server.ws_auto = False
    layer_list = [int(x) for x in str(layers).split(",") if x.strip()]
    for layer in layer_list:
        _rows_bench_layer(server, spec, iters, layer, warmup, ws_list)

def _rows_bench_layer(server, spec, iters, layer, warmup, ws_list = ""):
    dev = server.device
    ncold = server.layers[layer][1]
    ffn = server.layers[layer][0]
    tier = getattr(ffn, "fused_rows", None)
    print(f"SEAM_ROWS_BENCH_HEAD layer={layer} n_cold={ncold} tier={tier} kernel={MOE_KERNEL} "
          f"iters={iters} ws_rows={server.ws_rows} avail_mib={_avail_mib()} "
          f"swapped_mib={_swap_used_mib()}", flush = True)
    ws_caps = [int(x) for x in str(ws_list).split(",") if x.strip()]
    for rows in [int(x) for x in spec.split(",") if x.strip()]:
        stage, x_ptr, ids_ptr, w_ptr, out_ptr = _probe_stage(rows, ncold)
        # One timed shape per piece size: 0 = the whole request in one piece. Naming a
        # --rows-bench-ws list means exactly those sizes; without one the default is the whole
        # request plus the server's own ws_rows split, which is the old behaviour, except that a
        # whole request above WHOLE_MAX is skipped - one 8192-row piece thrashed GX2 for 4.3 s a
        # call and wrote 1.3 GiB of swap in ten calls, and that is not a measurement anyone wants
        # by accident. 0 in the list still forces it.
        if ws_caps:
            want = [w for w in ws_caps if w == 0 or w < rows]
        else:
            want = [0] if rows <= WHOLE_MAX else []
            if 0 < server.ws_rows < rows:
                want.append(server.ws_rows)
            if not want:
                want = [server.ws_rows or rows]
        for ws in want:
            pieces = 1 if ws <= 0 else (rows + ws - 1) // ws
            piece_rows = rows if ws <= 0 else min(ws, rows)
            keep = server.ws_rows
            server.ws_rows = ws
            try:
                for _ in range(warmup):
                    server._forward(layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
                torch.cuda.synchronize(dev)
                torch.cuda.empty_cache()
                avail0, swap0 = _avail_mib(), _swap_used_mib()
                pin0, pout0 = _vmstat_swap()
                times, io_hits = [], 0
                for _ in range(iters):
                    pin_a, pout_a = _vmstat_swap()
                    t0 = time.perf_counter_ns()
                    server._forward(layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
                    torch.cuda.synchronize(dev)
                    times.append((time.perf_counter_ns() - t0) / 1e3)
                    pin_b, pout_b = _vmstat_swap()
                    if pin_b != pin_a or pout_b != pout_a:
                        io_hits += 1
                pin1, pout1 = _vmstat_swap()
                times.sort()
                p50 = times[len(times) // 2]
                print(f"SEAM_ROWS_BENCH rows={rows} piece_rows={piece_rows} pieces={pieces} "
                      f"iters={iters} p50_us={p50:.1f} p99_us={times[min(len(times) - 1, int(len(times) * 0.99))]:.1f} "
                      f"min_us={times[0]:.1f} max_us={times[-1]:.1f} us_per_row={p50 / rows:.2f} "
                      f"avail_mib={avail0}->{_avail_mib()} swapped_mib={swap0}->{_swap_used_mib()} "
                      f"calls_with_swap_io={io_hits} swapin_pages={max(pin1 - pin0, 0)} "
                      f"swapout_pages={max(pout1 - pout0, 0)}", flush = True)
                # Same figures under the tag the wire-side reports grep for, so one log carries both
                # the wire-inclusive SEAM_COMPUTE p50 and the compute-alone p50.
                print(f"SEAM_COMPUTE rows={rows} n={iters} p50_us={p50:.1f} "
                      f"mean_us={sum(times) / len(times):.1f} max_us={times[-1]:.1f} "
                      f"bench=rows_bench layer={layer} pieces={pieces}", flush = True)
            finally:
                server.ws_rows = keep
        del stage
        torch.cuda.empty_cache()


def _report_recon(server, rows, tag):
    """Print the batched reconstruct tier's group plan as built by this load, and whether forcing
    the dispatch threshold can work at all. Experts above fused_rows go to that tier; if the
    module has no recon state (batch_recon False) they would be dropped instead of served, so a
    forced rung must be skipped rather than timed and reported as fast."""
    states = [(layer, getattr(ffn, "batch_recon", None)) for layer, (ffn, _nc) in server.layers.items()]
    have = [(layer, r) for layer, r in states if r not in (None, False)]
    caps = sorted({int(r.cap) for _l, r in have})
    max_rows = sorted({int(r.max_rows) for _l, r in have})
    print(f"SEAM_WS_RECON rows={rows} tier={tag} layers={len(have)}/{len(states)} "
          f"cap={caps} max_rows={max_rows} "
          f"fused_rows={sorted({int(getattr(f, 'fused_rows', -1)) for f, _nc in server.layers.values()})}",
          flush = True)
    return bool(have)


def ws_probe(server, spec, min_avail_mib, tiers = ()):
    """Measure the GX-side workspace and cost of the real cold-expert forward, one row count at a time.

    The slab (slab_mib) is only the pinned staging bill. What a bigger row bucket really costs on
    the GX is the device memory today's mixed-K forward allocates for that many rows, and that has
    to be measured on the box with the pack already resident, not predicted: this runs the same
    _forward the server runs, at each row count, and prints the peak and the MemAvailable around
    it. An OOM or a nearly empty box is reported as such and the ladder continues, so one run maps
    the whole 2048/4096/8192 decision.

    `tiers` sweeps the dispatch threshold (fused_rows) instead of only the row count: every rung
    is warmed and then timed, so one pack load gives the whole rows x tiers table. The load-time
    value is restored afterwards.
    """
    if server.stub:
        print("SEAM_WS stub=1: nothing to measure without the cold experts", flush = True)
        return
    dev = server.device
    server.ws_auto = False             # the probe measures the rung it was pointed at
    ncold = server.layers[0][1]
    # Always measure the load-time threshold first (it is the deployed behaviour); a tiers list
    # adds the forced rungs after it, so one run is a complete A/B against the baseline.
    original = next((ffn.fused_rows for ffn, _ in server.layers.values()
                     if hasattr(ffn, "fused_rows")), None)
    # Only downwards: the fused kernel's buffers were sized for the load-time capacity (and mtile_ok
    # False means it is TEMP_ROWS_FUSED, not FUSED_ROWS_WIDE), so a higher rung would run past them.
    ladder = [t for t in tiers if t != original and (original is None or int(t) < int(original))]
    ladder = ([original] if original is not None else []) + ladder or [None]
    over = [t for t in tiers if original is not None and int(t) > int(original)]
    if over:
        print(f"SEAM_WS_TIERS_ABOVE load={original} want={over}: skipped, the fused buffers were "
              f"sized at load (set EXL3_MOE_FUSED_ROWS_WIDE before the load to sweep them)",
              flush = True)
    recon_ok = None   # decided after the first rung has built the recon state
    for rows in [int(x) for x in spec.split(",") if x.strip()]:
        stage, x_ptr, ids_ptr, w_ptr, out_ptr = _probe_stage(rows, ncold)
        for limit in ladder:
            tag = "default" if limit == original or limit is None else str(int(limit))
            forced = limit is not None and limit != original
            if forced and recon_ok is False:
                print(f"SEAM_WS rows={rows} tier={tag} SKIP: no batched reconstruct tier on this "
                      f"module, so experts above tier={tag} would be dropped, not served",
                      flush = True)
                continue
            if limit is not None and not _set_tier(server, limit):
                print(f"SEAM_WS rows={rows} tier={tag} SKIP: no module exposes fused_rows",
                      flush = True)
                continue
            avail0 = _avail_mib()
            if 0 <= avail0 < min_avail_mib:
                print(f"SEAM_WS rows={rows} tier={tag} SKIP avail_mib={avail0} "
                      f"min_avail_mib={min_avail_mib}: the pack is resident and the box has no "
                      f"room left to measure in", flush = True)
                continue
            us = _probe_run(server, rows, (x_ptr, ids_ptr, w_ptr, out_ptr), tag)
            if recon_ok is None:
                recon_ok = _report_recon(server, rows, tag)
            if us is not None:
                print(f"SEAM_WS rows={rows} tier={tag} "
                      f"alloc_mib={torch.cuda.memory_allocated(dev) / 2**20:.1f} "
                      f"reserved_mib={torch.cuda.memory_reserved(dev) / 2**20:.1f} "
                      f"peak_alloc_mib={torch.cuda.max_memory_allocated(dev) / 2**20:.1f} "
                      f"peak_reserved_mib={torch.cuda.max_memory_reserved(dev) / 2**20:.1f} "
                      f"avail_before_mib={avail0} avail_after_mib={_avail_mib()} us={us:.0f} "
                      f"us_per_row={us / max(rows, 1):.2f} "
                      f"fin={bool(np.isfinite(view(out_ptr, rows * HIDDEN * 4, np.float32)).all())}",
                      flush = True)
        del stage
        torch.cuda.empty_cache()
    if original is not None:
        _set_tier(server, original)


def ws_check(server, rows):
    """Does serving a request in ws_rows pieces compute the same thing as one piece?

    Capping the workspace by chunking is only worth having if it is numerically harmless, and the
    answer has to be measured: the mixed-K path pads each expert's token run and the accumulation
    order inside a piece can differ from the order in a whole-bucket call. This runs the same
    deterministic request twice - one piece, then ws_rows pieces - and prints the largest absolute
    difference in the fp32 response (SEAM_WS_CHECK). `rows` must be a multiple of ws_rows.
    """
    if server.stub:
        print("SEAM_WS_CHECK stub=1: nothing to compare without the cold experts", flush = True)
        return
    step = server.ws_rows
    if not 0 < step < rows:
        print(f"SEAM_WS_CHECK rows={rows} skipped: needs 0 < --ws-rows < rows", flush = True)
        return
    ncold = next(iter(server.layers.values()))[1]
    stage, x_ptr, ids_ptr, w_ptr, out_ptr = _probe_stage(rows, ncold)
    # A deterministic ramp in [0,1), not zeros: identical zeros would hide a piece that came back
    # empty. bf16 is the top half of an fp32, so the flat uint16 view takes the high 16 bits.
    ramp = (np.arange(rows * HIDDEN, dtype = np.float32) % 97 / 97).view(np.uint32) >> 16
    view(x_ptr, rows * HIDDEN * 2, np.uint16)[:] = ramp.astype(np.uint16)
    view(w_ptr, rows * TOPK * 2, np.uint16).reshape(rows, TOPK)[:] = 0x3f00   # 0.5 in bf16

    def run():
        server._forward(0, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
        torch.cuda.synchronize(server.device)
        return view(out_ptr, rows * HIDDEN * 4, np.float32).reshape(rows, HIDDEN).copy()

    whole = run()
    server.ws_rows = 0
    piece = run()
    server.ws_rows = step
    scale = float(np.abs(whole).max()) or 1.0
    print(f"SEAM_WS_CHECK rows={rows} ws_rows={step} pieces={rows // step} "
          f"max_abs={float(np.abs(whole - piece).max()):.6g} rel={float(np.abs(whole - piece).max()) / scale:.3g} "
          f"ref_absmax={scale:.6g} identical={bool(np.array_equal(whole, piece))}", flush = True)
    del stage
    torch.cuda.empty_cache()


def ws_tier_check(server, rows, tiers):
    """Does forcing the batched reconstruct tier change the numbers?

    Same deterministic ramp request as ws_check, run once per dispatch threshold in `tiers` (the
    load-time threshold first when it is not in the list), and the largest absolute difference
    against the first run. The tiers accumulate in different orders (fused kernel slot tiles vs one
    batched GEMM over a padded slab), so identical bits are not expected here; ws_check proves bit
    identity *within* a tier, this bounds the difference *between* them.
    """
    if server.stub:
        print("SEAM_WS_TIER stub=1: nothing to compare without the cold experts", flush = True)
        return
    limits = [int(x) for x in tiers if str(x).strip()]
    if not limits:
        print("SEAM_WS_TIER skipped: needs a tier list", flush = True)
        return
    original = next((ffn.fused_rows for ffn, _ in server.layers.values()
                     if hasattr(ffn, "fused_rows")), None)
    if original is None:
        print("SEAM_WS_TIER skipped: no module exposes fused_rows", flush = True)
        return
    ncold = next(iter(server.layers.values()))[1]
    stage, x_ptr, ids_ptr, w_ptr, out_ptr = _probe_stage(rows, ncold)
    ramp = (np.arange(rows * HIDDEN, dtype = np.float32) % 97 / 97).view(np.uint32) >> 16
    view(x_ptr, rows * HIDDEN * 2, np.uint16)[:] = ramp.astype(np.uint16)
    view(w_ptr, rows * TOPK * 2, np.uint16).reshape(rows, TOPK)[:] = 0x3f00   # 0.5 in bf16

    def run():
        server._forward(0, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
        torch.cuda.synchronize(server.device)
        return view(out_ptr, rows * HIDDEN * 4, np.float32).reshape(rows, HIDDEN).copy()

    base, ref = None, None
    for limit in [original] + [t for t in limits if t != original]:
        _set_tier(server, limit)
        run()                     # warm: the recon tier builds its scratch on first use
        out = run()
        scale = float(np.abs(out).max()) or 1.0
        if ref is None:
            base, ref = limit, out
            print(f"SEAM_WS_TIER rows={rows} tier={limit} ref=1 absmax={scale:.6g} "
                  f"fin={bool(np.isfinite(out).all())}", flush = True)
            continue
        d = float(np.abs(out - ref).max())
        print(f"SEAM_WS_TIER_DIFF rows={rows} a={base} b={limit} max_abs={d:.6g} "
              f"rel={d / scale:.3g} fin={bool(np.isfinite(out).all())} "
              f"identical={bool(np.array_equal(out, ref))}", flush = True)
    _set_tier(server, original)
    del stage
    torch.cuda.empty_cache()



def _install_cpp_hook(srv, server):
    """SEAM_MOE_CPP=1: serve the cold-expert MoE from the C hook instead of _forward_piece.

    Opt-in and fail-open. Registration goes through server._cx_moe, which is the cache the load
    already filled when SEAM_MOE_PRELOAD=1, so no layer is restacked twice and no stacked bytes are
    copied: CppMoe hands C the data_ptr() of the tensors this server holds and allocates one int32
    remap table per layer. A layer whose restack is absent - the mixed-K module, i.e. 38/39, or any
    layer the load reported as SEAM_MOE_NOFIT - is not registered; C forwards that one request to
    the Python trampoline CppMoe leaves behind, which is server.compute, so it answers exactly as
    it does today instead of refusing. A missing .so or a failed registration prints why and
    returns None, and the caller installs the Python hook.

    install(handle=srv._h) makes qmc_compute the hook itself, through qes_server_set_fn and the C
    symbol's address: on the proxy threads a request for a registered layer goes from the slab
    straight into the kernels, with no Python frame and no GIL. Only the layers that were not
    registered take the trampoline. install() refuses a half out mode and closes the context before
    installing anything, so the state is either the C hook or the shipped Python one.

    The returned object must outlive the server: it owns the remap tables, the group arrays and the
    fallback trampoline whose addresses C holds.
    """
    if os.environ.get("SEAM_MOE_CPP", "0") == "0":
        return None
    if server.stub:
        # No pack and no restacks, so there is nothing to point C at; say it rather than let the
        # registration fail with the KeyError of an empty cache.
        print("SEAM_MOE_CPP_OFF stub=1 - the stub answers from the protocol, not from experts",
              flush = True)
        return None
    try:
        from qmesh.moe_cpp import CppMoe
        cpp = CppMoe(server).install(handle=srv._h)
    except Exception as e:   # noqa: BLE001 - any failure keeps the Python path
        print(f"SEAM_MOE_CPP_OFF err={type(e).__name__}: {str(e).splitlines()[0][:200]} "
              f"- serving the Python cold path", flush = True)
        return None
    print(f"SEAM_MOE_CPP_ON layers={len(cpp.layers)} fallback={cpp.fallback} "
          f"ws_rows={server.ws_rows} out_mode={cpp.out_mode()}", flush = True)
    return cpp


def main():
    global MOE_KERNEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/cold")
    # The client resolves the same variable (qmesh/placement.py), so one export names the table on
    # both halves of a pair; an explicit --placement still wins.
    ap.add_argument("--placement", default=os.environ.get("SEAM_PLACEMENT",
                                                           "/cold/placement.exl3.json"))
    ap.add_argument("--dev", default="rocep1s0f0")
    ap.add_argument("--gid", type=int, default=3)
    ap.add_argument("--port", type=int, default=47311)
    ap.add_argument("--stub", action="store_true",
                    help="answer every request with zeros (wire/protocol baseline, no expert compute)")
    ap.add_argument("--buckets", default=os.environ.get("SERVER_BUCKETS", "1,6,64,256,2048"),
                    help="row buckets this server can serve (ascending, one port per bucket from "
                         "--port+1). Clients request a subset; the superset default means one "
                         "long-lived server serves every benchmark arm without a reload. The largest "
                         "bucket is the per-call row limit the client splits prefill on, and that is "
                         "what prefill throughput rides on: a 64-row chunk pays the ~40 ms fixed cost "
                         "of one GX layer call, a 2048-row chunk pays it once for 32x the rows "
                         "(28.5 us/row at 2048 vs 78 us/row at 512, measured on GX2). The bucket is "
                         "paid twice: protocol v3 staging is ~30.2 KiB pinned per row per slot "
                         "(61 MiB for 2048 at one slot, 122 at two), and the cold-expert forward "
                         "needs ~147 KiB per row of transient device memory (measured: 301 MiB at "
                         "2048 rows, 602 at 4096, 1202 at 8192). With a 113.65 GiB pack resident a "
                         "GB10 has 0.7-1.4 GiB left, so 2048 is the largest bucket that fits with a "
                         "margin (the fit is printed as slab_mib/ws_mib/fits in SEAM_SERVER_SLAB); a "
                         "bigger bucket needs --ws-rows to cap the workspace, and a tighter box "
                         "should serve 1024.")
    ap.add_argument("--slab-slots", type = int,
                    default = int(os.environ.get("SEAM_SLAB_SLOTS", "1")),
                    help="payload slots per bucket (1..8): how many requests one client handle may "
                         "keep in flight. NS>1 lets a client pipeline prefill micro-batches so the "
                         "wire transfer of slot k+1 overlaps the GX compute of slot k. Costs "
                         "NS x rows x 30.2 KiB of pinned memory per bucket.")
    ap.add_argument("--wait", action="store_true",
                    help="queue a second client behind the active session instead of rejecting it "
                         "(also settable per client with SEAM_WAIT=1)")
    ap.add_argument("--ws-probe", default = "",
                    help = "measure the GX workspace of the real forward at these row counts "
                           "(comma separated, ascending) and exit without listening: prints one "
                           "SEAM_WS line per row count with peak memory and MemAvailable")
    ap.add_argument("--ws-probe-min-avail", type = int, default = 96,
                    help = "skip a --ws-probe row count when MemAvailable is below this many MiB "
                           "(a full pack plus a nearly full box would swap, not fail, and the "
                           "number would be meaningless)")
    ap.add_argument("--ws-rows", type = int, default = int(os.environ.get("SEAM_WS_ROWS", "0")),
                    help = "ceiling on the rows served per piece, so the peak device workspace "
                           "follows the piece instead of the row bucket (measured ~147 KiB per "
                           "row of scratch in the cold-expert forward: 301 MiB at 2048 rows, 602 "
                           "at 4096, 1202 at 8192). The piece actually used is the largest rung "
                           "of WS_LADDER under this ceiling that MemAvailable pays for, because "
                           "each piece re-reads the layer's cold-expert weights (2048 rows "
                           "measured 12.5-13.8 us/row on the 4460 pack against 18.8 for 1024). "
                           "0 = one piece per request, whatever it costs; SEAM_WS_ROWS_AUTO=0 "
                           "pins the split to exactly this number.")
    ap.add_argument("--ws-tiers", default = os.environ.get("SEAM_WS_TIERS", ""),
                    help = "comma separated fused_rows thresholds to sweep for --ws-probe / "
                           "--ws-tier-check: the threshold is the dispatch split, so experts above "
                           "it take the batched reconstruct tier (measured ~4x faster at prefill "
                           "widths). Every rung is warmed before it is timed; the load-time value "
                           "is restored afterwards.")
    ap.add_argument("--ws-check", type = int, default = 0, metavar = "ROWS",
                    help = "run the same ROWS-row request once in one piece and once in --ws-rows "
                           "pieces and print the largest difference (SEAM_WS_CHECK), then exit: "
                           "proves that capping the workspace by chunking does not change the "
                           "numbers. ROWS must be a multiple of --ws-rows.")
    ap.add_argument("--ws-tier-check", type = int, default = 0, metavar = "ROWS",
                    help = "run the same ROWS-row request once per --ws-tiers threshold and print "
                           "the largest difference from the load-time tier (SEAM_WS_TIER_DIFF), "
                           "then exit: how far forcing the batched tier moves the numbers.")
    ap.add_argument("--fail-after", type = int, default = int(os.environ.get("SEAM_FAIL_AFTER", "0")),
                    help = "test hook: raise inside the compute hook on call N+1 (stub servers "
                           "included), to prove that a failing compute fails the client instead of "
                           "handing it the previous call's bytes. 0 = never.")
    ap.add_argument("--moe-kernel", default = MOE_KERNEL, choices = ("cudaexl3", "mixedk"),
                    help = "cold-expert kernel: cudaexl3 (default) runs each layer on "
                           "vendor/cuda-exl3's MoE gemm after a one-off restack into per-bitrate "
                           "trellis stacks; mixedk keeps exllamav3's mixed-K module. Per-layer "
                           "fallback to mixedk if the restack raises.")
    ap.add_argument("--rows-bench", default = "",
                    help = "comma separated row counts: load the pack, then time the real "
                           "cold-expert forward at each of them without waiting for a client and "
                           "exit (one piece, plus the --ws-rows split when set). Reports "
                           "SEAM_ROWS_BENCH with p50/p99, us/row, MemAvailable and the swap "
                           "in/out pages seen during the timed calls.")
    ap.add_argument("--rows-bench-iters", type = int, default = 10,
                    help = "timed calls per row count for --rows-bench (default 10).")
    ap.add_argument("--rows-bench-layer", default = "0",
                    help = "layer(s) whose resident cold experts --rows-bench times, comma "
                           "separated so one load gives the spread across the stack (default 0).")
    ap.add_argument("--rows-bench-ws", default = "",
                    help = "comma separated piece sizes to time for --rows-bench, so one load "
                           "compares e.g. 1024,2048,4096 on the same rows (default: the server's "
                           "own --ws-rows split).")
    args = ap.parse_args()

    # A GX box holds one cold pack and one server, and two kinds of job want it: the pair release,
    # and a bench that measures against a loaded pack. They are serialised by the box gate,
    # <queue>/gx<N>.gate, whose lifecycle pair_gx_gate_take/pair_gx_gate_down own; the one launcher
    # that checks it and starts a pack load is seam_server_ensure.sh, and it passes the gate's
    # answer as SEAM_GATE_OK=1 through the docker exec. A direct `python3 qmesh/worker_seam_server.py`
    # ignores the claim and loads 113.65 GiB the release thinks is free (the 01:43 orphan on GX2
    # blocked the dev update for its whole run), so a launch that will map the pack is refused here
    # -- before SeamServer runs, so before a byte is mapped or a CUDA context exists and a refused
    # launch costs the box nothing. --stub is exempt: it returns before the pack and CUDA (see
    # SeamServer.__init__), so it takes no memory the gate protects, and a stub launch is how the
    # wire tests run against a full box.
    if os.environ.get("SEAM_GATE_OK") != "1" and not args.stub:
        # stderr, and the first line verbatim: the ensure/smoke refusal check and
        # qmesh/tests/server_gate.sh match that exact prefix.
        print("SEAM_SERVER_REFUSED SEAM_GATE_OK missing: start through seam_server_ensure.sh",
              file = sys.stderr, flush = True)
        print("  (that helper takes the gx gate and passes SEAM_GATE_OK=1; a direct launch would "
              "map the pack outside the claim)", file = sys.stderr, flush = True)
        return 3
    # Set before the pack loads: the switch is read per call, but a bad value must fail the
    # launch, not the first request.
    MOE_KERNEL = args.moe_kernel

    device = torch.device("cuda:0")
    t0 = time.time()
    server = SeamServer(args.model_dir, args.placement, device, stub = args.stub,
                        ws_rows = args.ws_rows, fail_after = args.fail_after)
    # build= is the release that started this process (SEAM_BUILD_ID from the pair release system),
    # so a release check can tell which build the *running* server came from; "-" = env not passed.
    print(f"SEAM_SERVER_LOADED {time.time() - t0:.1f}s stub={int(args.stub)} "
          f"fail_after={args.fail_after} build={os.environ.get('SEAM_BUILD_ID') or '-'} "
          f"avail_mib={_avail_mib()} swapped_mib={_swap_used_mib()}", flush=True)
    if not args.stub:
        # The device side of the bill, once per load: free_mib near zero with several GiB swapped
        # means this pack is too big for the box and any arm dies in a launch failure, whatever the
        # workspace knobs say.
        print(f"SEAM_SERVER_DEVMEM {_dev_mem_mib()} avail_mib={_avail_mib()} "
              f"swapped_mib={_swap_used_mib()}", flush=True)
    tiers = tuple(int(x) for x in args.ws_tiers.split(",") if x.strip())
    if tiers:
        print(f"SEAM_WS_TIERS={','.join(map(str, tiers))}", flush = True)
    if args.ws_probe or args.ws_check or args.ws_tier_check:
        # All modes share one pack load: the ladder first (it resets the peak counters), then the
        # equivalence checks, whose own peaks do not matter.
        if args.ws_probe:
            ws_probe(server, args.ws_probe, args.ws_probe_min_avail, tiers = tiers)
        if args.ws_check:
            ws_check(server, args.ws_check)
        if args.ws_tier_check:
            ws_tier_check(server, args.ws_tier_check, tiers)
        return
    if args.rows_bench:
        rows_bench(server, args.rows_bench, args.rows_bench_iters, args.rows_bench_layer,
                    ws_list = args.rows_bench_ws)
        return
    threading.Thread(target = _reporter, daemon = True).start()

    # One pinned slab per row bucket, registered once; each client session gets fresh QPs on
    # them and the server answers the control hello with the bucket -> port mapping. The slabs are
    # allocated here, after the pack load, so the list has to have been sized to the box's free
    # memory beforehand (seam_server_ensure.sh --buckets auto does that): print the bill first.
    buckets = sorted({int(r) for r in args.buckets.split(",") if r.strip()})
    want_mib = slab_mib(buckets, args.slab_slots)
    ws_mib = ws_mib_for(buckets[-1], args.ws_rows)
    avail_mib = _avail_mib()
    swapped_mib = _swap_used_mib()
    fits = int(avail_mib < 0 or want_mib + ws_mib <= avail_mib - 64)
    print(f"SEAM_SERVER_SLAB buckets={','.join(map(str, buckets))} slots={args.slab_slots} "
          f"slab_mib={want_mib} ws_rows={args.ws_rows} ws_mib={ws_mib} rows_max={buckets[-1]} "
          f"avail_mib={avail_mib} swapped_mib={swapped_mib} fits={fits}", flush = True)
    if not fits:
        # The pack is already resident, so memory that does not fit is paid by swapping the pack
        # out page by page: every expert call then faults, and the box looks alive while crawling.
        print(f"SEAM_SERVER_SLAB_WARN buckets=... slab_mib={want_mib} ws_mib={ws_mib} "
              f"avail_mib={avail_mib} over_mib={want_mib + ws_mib - avail_mib + 64}: lower "
              f"--buckets, set --ws-rows, or expect the pack to be swapped out (see "
              f"seam_server_ensure.sh --buckets auto)", flush = True)
    if swapped_mib > SWAP_WARN_MIB:
        # Not fatal to start (the pages fault back in), but every call pays for it and the box can
        # fail a launch outright: GX2 2026-11-01 ran with 4.4 GiB swapped, hit p50 47 ms / max
        # 4.15 s at rows=64 and then an "unspecified launch failure" in the decode arm.
        print(f"SEAM_SERVER_SWAP_WARN swapped_mib={swapped_mib} avail_mib={avail_mib}: the pack is "
              f"paging; expect multi-second SEAM_COMPUTE outliers and possible launch failures - "
              f"restart the box's server with a smaller --buckets/--slab-slots", flush = True)
    placement_sha = hashlib.sha256(open(args.placement, "rb").read()).hexdigest()[:12]
    srv = qes.Server(args.dev, args.gid, args.port, 0, buckets, build = qes.build_id(),
                     placement_sha = placement_sha, wait = args.wait, slots = args.slab_slots)
    # SEAM_MOE_CPP=1 installs the C hook instead (see _install_cpp_hook); its object is held
    # here, not dropped, because it owns the arrays the C context reads. With the variable
    # unset this is exactly the trampoline it has always been.
    _cpp_hook = _install_cpp_hook(srv, server)   # noqa: F841 - alive for srv.run()
    if _cpp_hook is None:
        srv.set_fn(lambda L, R, x, ids, w, o: server.compute(L, R, x, ids, w, o))
    try:
        srv.run()   # serves client sessions until SIGINT/SIGTERM
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        srv.close()


if __name__ == "__main__":
    # sys.exit, not a bare call: main returns 3 from the gate refusal, and a launcher that only sees
    # "the process ended" would read a refusal as a clean shutdown.
    sys.exit(main())
