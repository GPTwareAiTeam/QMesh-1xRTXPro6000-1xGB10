"""P7-side expert host over the native RDMA seam (qmesh/native/qes.py).

Drop-in for RemoteExpertHost (same method surface used by block_sparse_mlp_cpu.py), selected
by REMOTE_TRANSPORT=seam. Requests are issued and collected on the caller's CUDA stream via
qes_issue / qes_collect, so the path is graph-capturable (no host sync, no socket).

One control connection per host (SEAM_PEER:SEAM_PORT) carries the session negotiation; the
server answers with the row buckets it can serve and the data port of each, so a long-lived
GX server serves run after run without a reload. The requested bucket list need not match
the server's: this class adopts whatever the server grants.

Env:
  SEAM_DEV       RDMA device on P7 (default rocep142s0f0)
  SEAM_GID       GID index (default 5)
  SEAM_PEER      GX1 address (required unless supplied in host configuration)
  SEAM_PORT      seam control port (default 47311)
  SEAM_BUCKETS   row buckets to *request* (default 1,6,64,256,512,1024,2048,4096); the
                 server's answer wins, so a superset request adapts to any server list
  SEAM_WAIT      1 = queue behind the server's active session instead of failing fast
  SEAM_OVERLAP   0 = wait for each layer's reply inside submit_issue (the pre-split order); 1, the
                 default, issues and lets the caller's own experts run before the wait
  EXL3_SEAM_PROF 1 times issue -> collect per (bucket, rows) with CUDA events and dumps the per-layer
                 round trip at exit (wire + GX1 compute). Eager only: a captured graph records the
                 calls once at capture and then replays ops, so there is nothing to time per round.
"""
from __future__ import annotations

import atexit
import os
import statistics
import sys
import time

import torch

from qmesh.native import qes
from qmesh import placement as seam_placement

HIDDEN = qes.HIDDEN
TOPK = qes.TOPK

_PROF = os.environ.get("EXL3_SEAM_PROF", "0") == "1"
# Decode batches only by default: prefill chunks would land in the same bucket keys
_PROF_MAX_ROWS = int(os.environ.get("EXL3_SEAM_PROF_ROWS", "6"))
# Per-request stage trace: layer, bucket, cold count, stream and source pointers, printed host-side
# only so it cannot change the ordering it is meant to observe.
_ID_TRACE = os.environ.get("SEAM_ID_TRACE", "0") == "1"
# EXL3_SEAM_PAD_TRACE=1 prints the first request per bucket whose pad rows are *not* rewritten, and
# every request that does rewrite them (see _stage): the knob decides nothing, it is the proof that
# the branch is taken where the note says it is.
_PAD_TRACE = os.environ.get("EXL3_SEAM_PAD_TRACE", "0") == "1"
# SEAM_OVERLAP=1 issues a layer's cold-expert request and returns immediately, so the caller's own
# GPU experts (block_sparse_mlp.py:1194 submits, :1519 combines) run while the GX computes the cold
# share; 0 waits for the reply inside submit_issue, which is what the transport did before, so the
# same build can measure both orders.
_OVERLAP = os.environ.get("SEAM_OVERLAP", "1") == "1"
# SEAM_COPY_STREAM=1 (default) moves a request's wire copies off the caller's stream: qes_issue
# only enqueues (header H2D, x/ids/w D2H, publish kernel), so they can ride a stream of their own
# that waits on an event recorded right after the routing that produced their inputs. On the
# caller's stream the P7's own expert kernels queued behind ~1.5 ms of PCIe copies per layer, so
# neither the GX's receive nor the hot experts started until the copies were done. 0 keeps the
# single-stream order for measurement, and graph capture always uses it (see _wire_stream).
_COPY = os.environ.get("SEAM_COPY_STREAM", "1") == "1"
# EXL3_SEAM_MARK=1 records CUDA events at one prefill chunk's issue, at the caller's hot-expert span
# (block_sparse_mlp marks that with SeamProxy.mark, because only the module knows where its own expert
# kernels begin) and at the collect, then prints their device-side timestamps relative to the issue.
# That is the timeline the question "does the GX start receiving before the P7's hot experts" needs.
# issue/hot0/hot1/collect are on the caller's stream, payload is on the wire stream, so every gap is
# stream order on both sides and no reading is a host-timing artefact.
_MARK = os.environ.get("EXL3_SEAM_MARK", "0") == "1"
# The first few layers of the first chunk are enough: the order is the same for all 40
_MARK_LAYERS = int(os.environ.get("EXL3_SEAM_MARK_LAYERS", "4"))
_MARK_MIN_ROWS = int(os.environ.get("EXL3_SEAM_MARK_MIN_ROWS", "512"))
_marks = {}     # layer -> {"route"/"hot0"/"hot1"/"collect"/"payload": ev, +"_s": stream ptr, +"_h": host ns}
_base_ev = None     # first timeline event of the process, so the dump can print absolute device ms
_prof_rows = {}     # (bucket, n) -> [(host_ns_0, host_ns_1, ev0, ev1)]


def _prof_dump():
    if not _prof_rows:
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    print("[SEAMPROF] bucket rows    n   host_us   dev_ms  dev_us/row", flush = True)
    for (bucket, n), vals in sorted(_prof_rows.items()):
        hu = statistics.median((v[1] - v[0]) / 1e3 for v in vals)
        dv = statistics.median(v[2].elapsed_time(v[3]) for v in vals)
        print(f"[SEAMPROF] {bucket:6d} {n:5d} {len(vals):4d} {hu:9.3f} {dv:8.3f} "
              f"{dv * 1e3 / max(n, 1):10.2f}", flush = True)


def _mark_dump():
    if not _marks:
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    print("[SEAMMARK] layer route_to_payload route_to_hot0  hot_span  hot1_to_collect "
          "collect_to_done route_to_done   (ms of device time)", flush = True)
    for layer in sorted(_marks):
        m = _marks[layer]
        if not all(k in m for k in ("route", "hot0", "hot1", "collect", "done")):
            continue
        t0 = m["route"]
        # route is recorded on the caller's stream where the routing that produces this layer's
        # picks is done, i.e. the earliest a correct copy could start. With the copies on the wire
        # stream route_to_hot0 is the caller's own hand-off cost; on the caller's stream it carries
        # the copy as well, which is the difference this report exists to show.
        pay = (f"{t0.elapsed_time(m['payload']):17.3f}" if "payload" in m else
               f"{'n/a (same stream)':>17}")
        print(f"[SEAMMARK] {layer:5d} {pay} "
              f"{t0.elapsed_time(m['hot0']):14.3f} "
              f"{m['hot0'].elapsed_time(m['hot1']):9.3f} "
              f"{m['hot1'].elapsed_time(m['collect']):16.3f} "
              f"{m['collect'].elapsed_time(m['done']):16.3f} "
              f"{t0.elapsed_time(m['done']):16.3f}", flush = True)
        ids = " ".join(f"{k}={m[k + '_s']:x}" for k in ("route", "hot0", "hot1", "collect", "done",
                                                       "payload")
                       if k + "_s" in m)
        print(f"[SEAMMARK] {layer:5d} streams {ids}", flush = True)
        # Absolute device ms since the first event of the process, and the host time each point was
        # enqueued. A gap that exists on the device but not on the host is queued work between the
        # two points; a gap on the host that is not on the device is the host running ahead.
        h0 = m.get("route_h")
        keys = ("route", "issue", "payload", "hot0", "hot1", "collect", "done")
        abs_ms = " ".join(f"{k}={_base_ev.elapsed_time(m[k]):.2f}"
                          for k in keys if k in m and _base_ev is not None)
        host_us = " ".join(f"{k}={((m[k + '_h'] - h0) / 1e3 if h0 is not None else 0):.0f}"
                           for k in keys if k + "_h" in m)
        # A device gap with no host gap is queued work between the two points; the reverse is the
        # host running ahead of the device.
        print(f"[SEAMMARK] {layer:5d} abs {abs_ms}  host_us {host_us}", flush = True)


if _PROF:
    atexit.register(_prof_dump)
if _MARK:
    atexit.register(_mark_dump)

class _Prefill:
    """One layer call's seam work in flight, from submit_issue to submit_collect.

    Holds the caller's routing views (nothing is copied yet for the pieces not issued), the plan of
    micro-batches, the single output tensor they all write into, and the requests already on the
    wire. Kept as a plain object so the caller can run its own expert kernels in between without
    the host touching a stream or a synchronisation primitive."""
    __slots__ = ("host", "layer", "x", "ids", "w", "n_cold", "plan", "out", "rows",
                 "stream", "dev_index", "next", "inflight")

    def __init__(self, host, layer, x, ids, w, n_cold, plan, out, rows, stream, dev_index):
        self.host = host
        self.layer = int(layer)
        self.x, self.ids, self.w, self.n_cold = x, ids, w, n_cold
        self.plan = plan
        # `out` is in the reply's own dtype (fp32, or bf16/fp16 under the 16-bit wire caps).
        self.out, self.rows = out, rows
        self.stream, self.dev_index = stream, dev_index
        self.next = 0            # index into plan of the next piece to issue
        self.inflight = []       # issued pieces awaiting collection, in issue order


class RemoteSeamHost:
    """Expert host over the seam. connect() negotiates a session with the GX server, which
    answers with the row buckets it can serve and one data port per bucket; a batch of n rows is
    padded to the smallest granted bucket >= n with id -1 and weight 0 (exact zeros on GX1), and
    batches above the largest bucket are issued in bucket-sized chunks. All work stays on the
    caller's stream.

    Prefill therefore splits on the client: a chunk of R rows costs floor(R / GX_MAX) round trips
    at the largest granted bucket plus one for the remainder, which is padded to the smallest
    granted bucket >= it (exact, no padding, when R is a multiple of that bucket). GX_MAX is not a
    client constant: the server advertises its buckets, the client asks for the standard set and
    adopts the intersection, so a server that can only afford 256 rows makes every prefill chunk
    go 16 calls deep instead of failing."""

    # Row buckets *requested* from the server. The answer is the server's own list restricted to
    # this request, in the server's order, so a request that is a superset of any plausible server
    # list costs nothing and adapts to the memory a given GX box can afford: 1 and 6 serve
    # decode/verify, 256..4096 are the prefill widths (powers of two), and a server configured with
    # 64 still narrows a 2..63-row verify to the 64-row bucket as before. The largest granted
    # bucket is the per-call row limit used for prefill splitting.
    BUCKETS = tuple(int(b) for b in os.environ.get("SEAM_BUCKETS", "1,6,64,256,512,1024,2048,4096")
                    .split(",") if b.strip())


    @classmethod
    def for_config(cls, infer_params):
        host = getattr(infer_params, "remote_expert_host", None)
        if host is None:
            raise RuntimeError("remote_expert_host is not configured")
        hosts = getattr(infer_params, "remote_expert_hosts", None)
        if hosts is None:
            hosts = {}
            infer_params.remote_expert_hosts = hosts
        key = ("seam", host.get("host"), host.get("port"))
        if key not in hosts:
            hosts[key] = cls(host.get("host") or os.environ["SEAM_PEER"],
                             host.get("port") or int(os.environ.get("SEAM_PORT", "47311")))
        return hosts[key]

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.dev = os.environ.get("SEAM_DEV", "rocep142s0f0")
        self.gid = int(os.environ.get("SEAM_GID", "5"))
        self.ctl = None
        # Granted buckets, in the server's order; None until connect().
        self.buckets = None
        self.seams = {}
        # Force the issue/collect branch for every batch shape (no socket fallback).
        self.stream_min_rows = 10**9
        self.layer_index = {}
        # Micro-batches kept in flight per layer call (see submit_prefill). The server advertises
        # its payload slot count in the negotiation reply; SEAM_NS caps how many of those slots
        # this client uses, so one server with 4 slots can be measured at NS=1/2/4 without a
        # reload. Decode is a single piece and is unaffected.
        self.slots = 1
        self.ns = 1
        # Reply format of the session (protocol v5 wire caps, SEAM_QES_OUT on both halves), for the
        # log only: qes_collect widens a 16-bit reply into the fp32 rows itself. Set by connect().
        self.reply_fmt = "fp32"
        self._splits = set()
        # One wire stream per device (see _wire_stream): the copies and the publish ride it so the
        # caller's stream reaches its own expert kernels without them. Created on first use.
        self._cstreams = {}
        # Persistent per-(device, bucket) staging for the wire tensors (see _stage). These are the
        # only device buffers qes_issue reads, and they are never freed, so the copies it enqueues
        # can never race an allocation that reuses the same block.
        self._stage_bufs = {}
        # Per-(device, bucket) row count of the last request staged into that bucket: the pad rows
        # past `n` are already id -1 / weight 0 unless this request is narrower than the last one,
        # and only then do they need rewriting (see _stage).
        self._stage_rows = {}
        self._pad_printed = set()     # EXL3_SEAM_PAD_TRACE only: buckets that have reported a skip
        # SEAM_ID_TRACE only: per-bucket (pre, caller, staged, event, layer, rows) of the last
        # request staged into that bucket, printed one request late (see _trace_ids).
        self._trace_prev = {}
        self._trace_hooked = False

    def connect(self):
        """Negotiates the session with the GX server (idempotent). Raises RuntimeError with the
        server's own reason when it is busy, incompatible, or serves none of the requested
        buckets; SEAM_WAIT=1 queues instead of failing."""
        if self.ctl is not None:
            return self
        want = os.environ.get("SEAM_WAIT", "0") == "1"
        ctl = qes.Control(self.host, self.port, buckets = self.BUCKETS, wait = want)
        self.buckets = ctl.buckets
        self.chunk_rows = self.buckets[-1]
        self.ctl = ctl
        self.slots = max(1, int(getattr(ctl, "slots", 1)))
        mode = qes.out_mode_of(int(ctl.caps))
        self.reply_fmt = qes.OUT_NAMES[mode]
        cap = int(os.environ.get("SEAM_NS", "0") or 0)
        self.ns = self.slots if cap <= 0 else max(1, min(self.slots, cap))
        print(f"SEAM_NEGOTIATED host={self.host} port={self.port} "
              f"requested={','.join(map(str, self.BUCKETS))} "
              f"accepted={','.join(map(str, ctl.buckets))} "
              f"ports={','.join(map(str, ctl.ports))} chunk_rows={self.chunk_rows} "
              f"server_slots={self.slots} ns={self.ns} "
              f"server_build={ctl.build} placement_sha={ctl.placement_sha} "
              f"proto={qes.PROTO_VERSION} layers={ctl.layers} "
              f"caps=0x{int(ctl.caps):x} out={qes.OUT_NAMES[mode]}", flush = True)
        self._check_placement(ctl)
        return self

    @staticmethod
    def _check_placement(ctl):
        """Refuse, before the first request, a server whose cold tables were cut against another
        placement file.

        The Hello's placement sha is the one thing that pins every layer's hot/cold split at once, and
        this is the only place the check can be made: the client's data Hello carries no table id, so
        the server cannot see which file its peer loaded. The client derives each layer's cold count
        as num_experts - len(hot) from the file it loaded, so two files with different sets fail loudly
        anyway -- the ids arrive out of range and the server's range check raises on the first one
        (one line, the process survives, the session dies). Two files with the *same* sets in another
        order -- the ranked and plain 4460 tables, 10900 cold experts each -- fail silently instead:
        every id is in range and names another expert. Caught here it costs one line instead of a
        145 s reload (2026-11-01 18:03: the prefill client still read the repo default while the
        identity client read the pack's manifest).
        """
        srv = (getattr(ctl, "placement_sha", "") or "").strip()
        mine = (seam_placement.current_sha() or "").strip()
        if srv and mine and not (srv.startswith(mine[:16]) or mine.startswith(srv[:16])):
            raise RuntimeError(
                f"placement mismatch on {ctl.host}:{getattr(ctl, 'port', '?')}: the server's Hello "
                f"advertises {srv[:16]} but this client loaded {mine[:16]} from "
                f"{seam_placement.current_path()}; the server cannot see this side's table, so a "
                f"reordered table of the same shape would answer from the wrong experts and one of "
                f"another shape would raise on its first out-of-range id (set PLACEMENT to the "
                f"server's file, or let the pack's PACK_MANIFEST.json pick it)")
        print(f"SEAM_PLACEMENT server_sha={srv[:16] or 'unset'} client_sha={mine[:16] or 'unset'} "
              f"path={seam_placement.current_path() or 'unset'}", flush = True)

    def _seam(self, bucket: int, device_index: int):
        """Data handle for one granted bucket, created on first use. The server assigned this
        bucket's port in the negotiation reply, so no ordering with the other buckets matters."""
        key = (bucket, device_index)
        if key not in self.seams:
            ctl = self.connect().ctl
            port = ctl.ports[ctl.buckets.index(bucket)]
            print(f"SEAM_CONNECT host={self.host} port={port} rows={bucket} dev_idx={device_index} "
                  f"dev={self.dev} gid={self.gid} buckets={','.join(map(str, ctl.buckets))}",
                  flush = True)
            self.seams[key] = ctl.seam(self.dev, self.gid, bucket, device_index)
        return self.seams[key]

    def register_layer(self, key, *args, **kwargs):
        idx = int(key.split(".")[1])
        self.layer_index[key] = idx
        return idx

    def _stage(self, layer_idx, x, ids, w):
        """Copies one piece into this host's persistent per-bucket staging tensors.

        qes_issue only *enqueues* the device->host copies of x/ids/w, so the tensors it reads must
        still be alive when those copies run. Building them per call (`.to(dtype).contiguous()`)
        hands the allocator blocks it can reuse for the next layer's routing, and a block reused by
        a kernel queued *ahead* of our copy makes the wire carry another layer's picks -- measured
        2026-11-01 as layer 7 id 277 of 274 cold and layer 30 id 284 of 275, with the mirror copy
        reading the same bad value out of the caller's tensor. Staging per bucket, owned for the
        host's lifetime, means the seam never reads a block the allocator can hand out again (and a
        captured graph never keeps a pointer to a freed block).

        One staging set per bucket, reused across requests, is safe because the copies are
        stream-ordered: a later request's staging write runs after the earlier request's copy,
        however the slots overlap. The buffers are born with pad rows at id -1 / weight 0, and the
        only way a stale id can sit past `n` is a request narrower than the one before it in the same
        bucket, so only that case rewrites them (see below)."""
        n = x.shape[0]
        dev = x.device
        bucket = next(b for b in self.buckets if b >= n)
        key = (dev.index or 0, bucket)
        st = self._stage_bufs.get(key)
        if st is None:
            st = (torch.empty(bucket, HIDDEN, dtype = torch.bfloat16, device = dev),
                  torch.full((bucket, TOPK), -1, dtype = torch.int32, device = dev),
                  torch.zeros(bucket, TOPK, dtype = torch.bfloat16, device = dev))
            self._stage_bufs[key] = st
        xp, idp, wp = st
        trace = _ID_TRACE and not torch.cuda.is_current_stream_capturing()
        pre = self._trace_pre(dev, idp, n) if trace else None
        xp[:n] = x
        idp[:n] = ids
        wp[:n] = w
        # The wire carries hdr->rows and the client's range check scans hdr->rows * TOPK ids, so the
        # pad rows of a bucket never reach the peer. They only have to stay id -1 / weight 0 for a
        # local reader, and rows past `n` are still cleared from the last time this bucket was
        # wider, so rewriting them every request is two host calls that the decode steady state --
        # one fixed row count per round, well under the bucket -- pays 40x per round for nothing.
        last = self._stage_rows.get(key, 0)
        self._stage_rows[key] = n
        if n < bucket and n < last:
            if _PAD_TRACE:
                print(f"SEAM_STAGE_PAD bucket={bucket} n={n} last={last} action=clear", flush = True)
            idp[n:].fill_(-1)
            wp[n:].zero_()
        elif n < bucket and last and _PAD_TRACE and key not in self._pad_printed:
            self._pad_printed.add(key)
            print(f"SEAM_STAGE_PAD bucket={bucket} n={n} last={last} action=skip", flush = True)
        if trace:
            self._trace_ids(layer_idx, dev, key, ids, idp, n, pre)
        return bucket, xp, idp, wp

    def _trace_pre(self, dev, idp, n):
        """Diagnostic (SEAM_ID_TRACE): snapshot of the staging ids *before* this request's write.

        Enqueued on the caller's stream ahead of the write, so the pair (pre, staged) brackets it: a
        wire value equal to `pre` and not to `staged` means the transport's device-to-host copy ran
        before this request's write, i.e. it read a previous request's bytes."""
        if not self._trace_hooked:
            # A run can end on the request being traced (a forward failure contains the exception,
            # see Generator.reap_failed_job): the exit dump is the only hook left, and it may sync
            # because nothing else will use the stream again.
            self._trace_hooked = True
            atexit.register(self._trace_final)
        self._trace_flush()
        pre = torch.empty(n * TOPK, dtype = torch.int32, pin_memory = True)
        pre.copy_(idp[:n].reshape(-1), non_blocking = True)
        return pre

    def _trace_flush(self):
        """Print the snapshots of every request whose event has completed, without waiting."""
        for key, prev in list(self._trace_prev.items()):
            if not prev[3].query():
                continue
            del self._trace_prev[key]
            self._trace_print(prev)

    def _trace_final(self):
        """Diagnostic (SEAM_ID_TRACE): dump the snapshots still pending at exit."""
        if not self._trace_prev:
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        for prev in list(self._trace_prev.values()):
            self._trace_print(prev)
        self._trace_prev.clear()

    @staticmethod
    def _trace_print(prev):
        pre, caller, staged, _ev, layer_idx, rows = prev
        print(f"SEAM_ID_PY layer={layer_idx} rows={rows} "
              f"pre={pre[:rows * TOPK].tolist()} "
              f"caller={caller[:rows * TOPK].tolist()} "
              f"staged={staged[:rows * TOPK].tolist()}", file = sys.stderr, flush = True)

    def _trace_ids(self, layer_idx, dev, key, ids, idp, n, pre):
        """Diagnostic (SEAM_ID_TRACE): pins this request's caller ids and staged ids for a later
        print, when the event recorded here has completed.

        A readback on the issue path would sync the stream and hide the race it has to observe
        (EXL3_MOE_SPLIT_ID_CHECK=1 makes the failing arm pass, measured 2026-11-01). Deferring
        needs no wait: every flush only prints requests whose event has already signalled. The
        three snapshots bracket the staging write -- `pre` before it, `staged` after it, `caller`
        the module's tensor - so a wire value equal to `pre` and not to `staged` proves the
        transport's copy ran ahead of the write, while `caller == staged` puts the value in the
        module's own translate."""
        caller = torch.empty(n * TOPK, dtype = torch.int32, pin_memory = True)
        staged = torch.empty(n * TOPK, dtype = torch.int32, pin_memory = True)
        caller.copy_(ids.reshape(-1), non_blocking = True)
        staged.copy_(idp[:n].reshape(-1), non_blocking = True)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(dev))
        self._trace_prev[key] = (pre, caller, staged, ev, layer_idx, n)

    def _wire_stream(self, dev):
        """Stream the request's copies ride, or None to keep them on the caller's stream.

        Everything qes_issue enqueues for a request (the header H2D, the D2H copies of x/ids/w and
        the publish kernel that makes the slot visible to the proxy thread) is stream-ordered on
        the stream it is handed, and it never waits on the host. Handing it the caller's stream put
        those copies in front of the layer's own expert kernels, so the P7's hot compute began only
        after they finished; on this stream they run beside it, and the request is on the wire as
        soon as the routing that produced it is done.

        Graph capture keeps the single-stream order: a stream that took part in no capture cannot
        wait on an event recorded inside one, and the capture bakes the stream of the copies it
        records. Decode is captured, prefill is not, and only prefill has copies worth moving."""
        if not _COPY or torch.cuda.is_current_stream_capturing():
            return None
        idx = dev.index or 0
        cs = self._cstreams.get(idx)
        if cs is None:
            with torch.cuda.device(dev):
                cs = torch.cuda.Stream(device = dev)
            self._cstreams[idx] = cs
        return cs

    def _issue_piece(self, layer_idx, pos, x, ids, w, n_cold, dev_index):
        """Stages one micro-batch to the smallest granted bucket >= its rows, puts it on the wire,
        and returns the in-flight record _collect_piece consumes. Nothing waits here: the split
        from collect is the whole point, because the caller's own expert kernels run next."""
        n = x.shape[0]
        stream = torch.cuda.current_stream(x.device)
        prof = _PROF and n <= _PROF_MAX_ROWS and not torch.cuda.is_current_stream_capturing()
        t0 = ev0 = None
        if prof:
            ev0 = torch.cuda.Event(enable_timing = True)
            ev0.record(stream)
            t0 = time.perf_counter_ns()
        cs = self._wire_stream(x.device)
        if cs is None:
            bucket, xp, idp, wp = self._stage(layer_idx, x, ids, w)
            wire = stream
        else:
            # The routing outputs are ready where the caller's stream stands now; the staging
            # writes and the transport's own copies follow on the wire stream, and the caller's
            # stream does not wait for them (only submit_collect does, through the reply).
            cs.wait_event(self._record_ev(stream))
            with torch.cuda.stream(cs):
                bucket, xp, idp, wp = self._stage(layer_idx, x, ids, w)
            # These are views of the caller's routing buffers, allocated on the caller's stream and
            # read on the wire stream: without this the allocator may hand their block to the next
            # layer while the copy that reads them is still queued, and the wire would carry
            # another layer's routing (the failure _stage's per-bucket buffers were introduced
            # for, 2026-11-01, one stream later).
            for t in (x, ids, w):
                t.record_stream(cs)
            wire = cs
        seam = self._seam(bucket, dev_index)
        try:
            if _ID_TRACE:
                # Host-side only: a readback of ids here would sync the stream and hide the very
                # race this trace exists to see (measured 2026-11-01: EXL3_MOE_SPLIT_ID_CHECK=1
                # turned the out-of-range id into a clean run). stream/capturing say whether two
                # requests for the same bucket can be staged from different streams, which is the
                # only way the transport's per-request device-to-host copy can read a newer
                # request's bytes.
                print(f"SEAM_ID_STAGE layer={layer_idx} rows={n} bucket={bucket} n_cold={n_cold} "
                      f"stream=0x{stream.cuda_stream:x} wire=0x{wire.cuda_stream:x} "
                      f"capturing={int(torch.cuda.is_current_stream_capturing())} "
                      f"x=0x{x.data_ptr():x} ids=0x{ids.data_ptr():x} idp=0x{idp.data_ptr():x}",
                      file = sys.stderr, flush = True)
            # `n`, not the bucket: the transport carries the real row count to the GX, which then
            # computes and answers `n` rows (SlotHdr::rows). The bucket is only what the handle was
            # opened with, so a remainder piece no longer pays for the padding it sits in.
            seam.issue(layer_idx, xp.data_ptr(), idp.data_ptr(), wp.data_ptr(),
                       wire.cuda_stream, n_cold or 0, n)
        except RuntimeError:
            # The transport refused this request (e.g. an id its own cold set cannot hold): the
            # proxy's "qes:" line names layer, slot, values and cold count. The stack says which
            # module/path submitted them, which is what the id-space hunt needs.
            if os.environ.get("SEAM_ID_TRACE"):
                import traceback
                print(f"SEAM_ID_TRACE layer={layer_idx} rows={n} bucket={bucket} n_cold={n_cold}",
                      file = sys.stderr, flush = True)
                traceback.print_stack()
            raise
        if _MARK and layer_idx in _marks and "payload" not in _marks[layer_idx]:
            # Recorded on the wire stream, right behind the publish kernel: this is when the
            # request is complete in the slab and the proxy thread can put it on the wire. On the
            # caller's stream it would say the same thing only in the copies-on-compute order.
            self._mark_ev(layer_idx, "payload", wire)
        return (pos, n, bucket, xp, idp, wp, seam, t0, ev0)

    @staticmethod
    def _prep(y, selected, weights):
        """Views of the caller's routing output, cast and cast-safe-copied later by _stage:
        x [rows,H], ids [rows,K], w [rows,K]."""
        return (y.reshape(-1, HIDDEN), selected.reshape(-1, TOPK), weights.reshape(-1, TOPK))

    def _split(self, rows: int, step: int | None = None):
        """Prefill split of `rows` onto the granted buckets, chunked at `step` (default chunk_rows)
        with the remainder in the smallest granted bucket >= it. Returns [(start, stop, bucket)].
        The remainder's bucket is not a multiple of `step`; the transport carries the piece's own
        row count (`n`, see _issue_piece), so the bucket only sizes the slab and the wire geometry,
        and the padded tail rows cost the GX nothing: it computes `n` rows and answers `n` rows."""
        step = step or self.chunk_rows
        out = [(i, min(i + step, rows), step) for i in range(0, rows - rows % step, step)]
        rem = rows % step
        if rem:
            out.append((rows - rem, rows, next(b for b in self.buckets if b >= rem)))
        return out

    def _collect_piece(self, layer_idx, piece, out, stream):
        """Collects one in-flight piece into `out` at its row offset. The collect's host->device
        copy and the issue's device->host copy are on the caller's stream, so this never waits on
        the host: the wait kernel for piece m simply sits after piece m+1's issue in the stream."""
        pos, n, _bucket, _xp, _idp, _wp, seam, t0, ev0 = piece
        seam.collect(layer_idx, out[pos:pos + n].data_ptr(), stream.cuda_stream, n)
        if _MARK and layer_idx in _marks and n >= _MARK_MIN_ROWS:
            # The wait kernel is behind the collect above, so this event's device time is when the
            # reply had landed and its rows were in `out`: the far end of the layer's seam wait.
            self._mark_ev(layer_idx, "done", stream)

        if t0 is not None:
            t1 = time.perf_counter_ns()
            ev1 = torch.cuda.Event(enable_timing = True)
            ev1.record(stream)
            _prof_rows.setdefault((_bucket, n), []).append((t0, t1, ev0, ev1))

    def _announce(self, layer_idx, rows, plan):
        """One SEAM_SPLIT line per (rows, plan) shape: what the split looks like on this server."""
        shape = (rows, tuple(b for _, _, b in plan))
        if shape in self._splits:
            return
        self._splits.add(shape)
        widths = ",".join(f"{b}x{sum(1 for s, e, _ in plan if _ == b)}" for b in
                          dict.fromkeys(b for _, _, b in plan))
        print(f"SEAM_SPLIT layer={layer_idx} rows={rows} chunk_rows={self.chunk_rows} "
              f"calls={len(plan)} ns={self.ns} server_slots={self.slots} "
              f"buckets={','.join(map(str, self.buckets))} plan={widths}", flush = True)

    def submit_issue(self, layer_idx, y, selected, weights, n_cold = None):
        """Puts this layer's cold-expert request on the wire and returns a pending handle.

        The caller runs its own GPU experts between this and submit_collect, so the GX's cold
        compute overlaps the P7's hot compute instead of following it. This used to call
        submit_prefill, which collects before returning: every layer waited out the whole RDMA
        round trip at the top of the MoE block with the GPU idle, and the prefill numbers of
        2026-11-01 (292.5 tok/s at 32k) carried that stall on all 40 layers.

        A chunk is split into micro-batches at the largest granted bucket; `ns` of them may be in
        flight (one on today's server: one payload slot per handle), so a chunk that fits in one
        bucket -- every decode step, and every prefill call at chunk_rows -- is issued here and
        waited for in submit_collect, which is a full round trip of overlap on all 40 layers.

        n_cold is this layer's cold-expert count, carried on every request so the client proxy can
        refuse an id the placement cannot hold (see qes.Seam.issue)."""
        self.connect()
        x, ids, w = self._prep(y, selected, weights)
        rows = x.shape[0]
        plan = self._split(rows)
        self._announce(layer_idx, rows, plan)
        # One fp32 [rows,H] output that every micro-batch writes into: torch.cat would hold the
        # parts and the copy at once (+rows*H*4 bytes, 235 MiB at 8192 rows), which the prefill
        # memory budget does not have. A single piece writes into its own bucket-sized buffer and
        # is returned as a prefix, exactly as the one-shot path always did.
        # fp32 on both wires: under SEAM_QES_OUT=bf16|fp16 the seam's collect widens the 16-bit slot
        # straight into these rows (one kernel reading the mapped slab, no staging buffer).
        out = (torch.empty(plan[0][2], HIDDEN, dtype = torch.float32, device = x.device)
               if len(plan) == 1 else
               torch.empty(rows, HIDDEN, dtype = torch.float32, device = x.device))
        # Arm the timeline before the first issue: _issue_piece records the request's payload event
        # on the wire stream, and that only happens for the layers the report is about.
        mk = _MARK and _OVERLAP and rows >= _MARK_MIN_ROWS and layer_idx < _MARK_LAYERS
        if mk:
            # A new chunk starts at layer 0: drop the previous chunk's timestamps so the report is one
            # chunk's order and not a mixture of 13 (the chunk loop re-enters layer 0 every time)
            if layer_idx == 0:
                _marks.clear()
            _marks.setdefault(layer_idx, {})
        p = _Prefill(self, layer_idx, x, ids, w, n_cold, plan, out, rows,
                     torch.cuda.current_stream(x.device), x.device.index or 0)
        if mk:
            # The point the copies are allowed to start from, on the stream that owns the routing.
            # Recorded before the issue loop, so its gap to hot0 is the caller's own hand-off in
            # both arms -- the copy itself included only when the copies ride this stream.
            self._mark_ev(layer_idx, "route", p.stream)
        while p.next < len(p.plan) and len(p.inflight) < self.ns:
            s, e, _ = p.plan[p.next]
            p.inflight.append(self._issue_piece(p.layer, s, p.x[s:e], p.ids[s:e], p.w[s:e],
                                                n_cold, p.dev_index))
            p.next += 1
        if mk:
            self._mark_ev(layer_idx, "issue", p.stream)
        if not _OVERLAP:
            return self.submit_collect(p)
        return p

    def submit_collect(self, handle):
        """Waits for the pending request(s) and returns fp32 [rows, HIDDEN] on the caller's stream
        (under a 16-bit wire the seam widened the reply exactly, so the values are the bf16/fp16
        the GX sent).

        Pieces the ns window did not cover (a chunk split into more micro-batches than there are
        slots) are issued and drained here, keeping the window full, so those pieces cost the
        transport's usual alternation -- the same order the one-shot path used for all of them."""
        if not isinstance(handle, _Prefill):
            # Already a finished tensor: SEAM_OVERLAP=0 submitted and waited in one call.
            return handle
        p = handle
        # Same row floor as the issue side: a drafter forward reuses these layer indices with a
        # handful of rows and would otherwise overwrite the chunk's collect with its own.
        if _MARK and p.rows >= _MARK_MIN_ROWS and p.layer in _marks:
            self._mark_ev(p.layer, "collect", p.stream)
        while p.next < len(p.plan):
            while len(p.inflight) >= self.ns:
                self._collect_piece(p.layer, p.inflight.pop(0), p.out, p.stream)
            s, e, _ = p.plan[p.next]
            p.inflight.append(self._issue_piece(p.layer, s, p.x[s:e], p.ids[s:e], p.w[s:e],
                                                p.n_cold, p.dev_index))
            p.next += 1
        while p.inflight:
            self._collect_piece(p.layer, p.inflight.pop(0), p.out, p.stream)
        return p.out[:p.rows]

    @staticmethod
    def _record_ev(stream):
        global _base_ev
        ev = torch.cuda.Event(enable_timing = True)
        ev.record(stream)
        if _MARK and _base_ev is None:
            _base_ev = ev
        return ev

    def _mark_ev(self, layer_idx, key, stream):
        """One timeline point: the device event, the stream it was recorded on, and the host time it
        was enqueued at. The dump prints all three, because a device gap with no host gap is queued
        work between the two points while a host gap with no device gap is the host running ahead."""
        m = _marks.setdefault(layer_idx, {})
        m[key] = self._record_ev(stream)
        m[key + "_s"] = stream.cuda_stream
        m[key + "_h"] = time.perf_counter_ns()

    def mark(self, layer_idx, label):
        """Caller-side timestamp for the timeline (EXL3_SEAM_MARK): block_sparse_mlp marks the start
        and end of the GPU experts it runs between an issue and its collect. Recorded on the calling
        stream, so the gaps to the seam's own events are device order and not host order."""
        if not _MARK or layer_idx not in _marks or label in _marks[layer_idx]:
            return
        self._mark_ev(layer_idx, label, torch.cuda.current_stream())

    def submit_prefill(self, layer_idx, y, selected, weights, n_cold = None):
        """Single-phase form for callers with nothing to overlap (benches, the CPU host's streamed
        path): issue then collect back to back, the same code the two-phase callers use."""
        return self.submit_collect(self.submit_issue(layer_idx, y, selected, weights, n_cold))

    def submit_issue_fused(self, *args, **kwargs):
        raise NotImplementedError("seam path uses EXL3_MOE_SPLIT_FUSED=0")

    def unregister(self):
        pass

    def shutdown(self):
        for seam in self.seams.values():
            seam.close()
        self.seams.clear()
        if self.ctl is not None:
            self.ctl.close()
            self.ctl = None
            self.buckets = None
