"""Piecewise CUDA graph segments for the decode step.

A Segment wraps a position-free stretch of the decode forward. Tensor arguments are copied into static
device buffers; the first call after one eager warmup records a CUDA graph over the wrapped function, and
every later call copies the tensors in, replays, and returns the static outputs.

Non-tensor arguments (the per-step params dict) are passed through live on every call and are NOT baked
into the graph: the captured body may only read per-step state through static device buffers.

All segments share one graph memory pool so intermediate allocations of different segments reuse blocks.
"""
import torch

_POOL = None
_STREAMS: dict = {}


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = torch.cuda.graph_pool_handle()
    return _POOL


def _capture_stream() -> torch.cuda.Stream:
    """One capture stream per device for every Segment. A fresh torch.cuda.Stream() per capture rotates
    through torch's pool of 32 streams, and the captured bodies key per-stream device workspaces on it
    (exl3_dgemv.cu get_ws: 64 MiB per (device, stream), raw cudaMalloc, never freed): each new decode
    shape then grew non-torch memory by ~66 MiB until all 32 pool streams had one, ~2.1 GiB at B >= 2
    (graphbuckets memprobe 2026-11-05: +66 MiB per first capture, unrecovered by dropping every graph)."""
    dev = torch.cuda.current_device()
    s = _STREAMS.get(dev)
    if s is None:
        s = _STREAMS[dev] = torch.cuda.Stream()
    return s


class Segment:
    """fn(*args) -> tensor or tuple of tensors.

    carry: params keys whose tensor values flow between segments through the params dict (e.g. the mHC
    "hc_pre" collapse weights each site hands to the next). A carried tensor present on entry is a static
    input; the value the captured body leaves under that key is a static output written back into the live
    params after every replay (Python side effects of the body do not run on replay)."""

    def __init__(self, fn, carry = ()):
        self.fn = fn
        self.carry = tuple(carry)
        self.graph = None
        self.static_in = None
        self.static_out = None
        self.carry_in = {}
        self.carry_out = {}
        self.warm = False

    @staticmethod
    def _params(args):
        return next((a for a in args if isinstance(a, dict)), None)

    def __call__(self, *args):
        if self.graph is None:
            if not self.warm:
                # Eager warmup: first-use allocations (workspaces, handles, tensor caches) happen here,
                # outside the capture
                self.warm = True
                out = self.fn(*args)
                return tuple(out) if isinstance(out, (tuple, list)) else (out,)
            self._capture(args)
        tens = [a for a in args if isinstance(a, torch.Tensor)]
        # One batched copy per replay instead of one launch per input/carry tensor
        params = self._params(args)
        dsts = list(self.static_in)
        srcs = tens
        if self.carry_in:
            dsts += list(self.carry_in.values())
            srcs += [params[k] for k in self.carry_in]
        torch._foreach_copy_(dsts, srcs, non_blocking = True)
        self.graph.replay()
        for k, t in self.carry_out.items():
            params[k] = t
        return self.static_out

    def _capture(self, args):
        self.static_in = tuple(a.clone() for a in args if isinstance(a, torch.Tensor))
        params = self._params(args)
        cap_params = dict(params) if params is not None else None
        if cap_params is not None:
            for k in self.carry:
                v = cap_params.get(k)
                if isinstance(v, torch.Tensor):
                    self.carry_in[k] = cap_params[k] = v.clone()
        # Non-tensor args are fixed for the capture (the params dict is replaced by the capture copy)
        self.layout = [
            cap_params if a is params and params is not None else (a if not isinstance(a, torch.Tensor) else None)
            for a in args
        ]
        s = _capture_stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            # capture_begin/end directly: the torch.cuda.graph context manager device-syncs and
            # flushes the allocator on enter, once per segment, inside the decode window
            g.capture_begin(pool = _pool(), capture_error_mode = "thread_local")
            try:
                out = self.fn(*self._interleave(self.static_in))
            finally:
                g.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = g
        self.static_out = tuple(out) if isinstance(out, (tuple, list)) else (out,)
        if cap_params is not None:
            for k in self.carry:
                v = cap_params.get(k)
                if isinstance(v, torch.Tensor) and v is not self.carry_in.get(k):
                    self.carry_out[k] = v

    def _interleave(self, tens):
        it = iter(tens)
        return [next(it) if slot is None else slot for slot in self.layout]
