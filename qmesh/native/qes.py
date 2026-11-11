"""ctypes binding for libqmesh_expert_seam (see qmesh_expert_seam.h).

The GX server loads the cold experts once and listens on one control port forever
(worker_seam_server.py -> Server). Each client run negotiates a *session* on that port
(Control), gets the row buckets the server can serve plus one data port per bucket, and
opens one Seam per bucket. Closing the Control (or the process dying, incl. kill -9)
ends the session, the server destroys that session's QPs and takes the next client with
no reload.

Client (P7): Control(...).seam(dev, gid, rows, ord).issue/collect enqueue on a torch
CUDA stream (graph-capturable).
Server (GX1): Server(...).set_fn(fn) installs a Python compute callback that the C proxy
thread calls with pinned-slab pointers. The callback runs under the GIL on the proxy
thread, so it must do its CUDA work on its own stream and synchronize before returning.
"""
from __future__ import annotations

import ctypes
import os
import subprocess

HIDDEN = 5120
TOPK = 6
NL = 40
ROLE_CLIENT = 0
ROLE_SERVER = 1
PROTO_VERSION = 5
MAX_BUCKETS = 12
MAX_SLOTS = 8   # QES_MAX_SLOTS: payload slots in flight per handle
BUILD_LEN = 40
MSG_LEN = 192

ST_OK = 0
ST_BUSY = 1
ST_QUEUED = 2
ST_INCOMPAT = 3
ST_BADREQUEST = 4
ST_ERROR = 5
ST_STATUS = 6

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin",
                         f"libqmesh_expert_seam.{os.uname().machine}.so")


class Nego(ctypes.Structure):
    """Mirror of qes_nego."""
    _fields_ = [("status", ctypes.c_int32),
                ("slots", ctypes.c_int32),
                ("version", ctypes.c_int32),
                ("layers", ctypes.c_int32),
                ("n_buckets", ctypes.c_int32),
                ("n_requested", ctypes.c_int32),
                ("buckets", ctypes.c_int32 * MAX_BUCKETS),
                ("ports", ctypes.c_int32 * MAX_BUCKETS),
                ("requested", ctypes.c_int32 * MAX_BUCKETS),
                ("caps", ctypes.c_int32),
                ("build", ctypes.c_char * BUILD_LEN),
                ("placement_sha", ctypes.c_char * BUILD_LEN),
                ("message", ctypes.c_char * MSG_LEN)]

    @staticmethod
    def _text(raw):
        return bytes(raw).split(b"\0", 1)[0].decode(errors = "replace")

    @property
    def bucket_list(self):
        return tuple(int(b) for b in self.buckets[:self.n_buckets])

    @property
    def port_list(self):
        return tuple(int(p) for p in self.ports[:self.n_buckets])

    @property
    def requested_list(self):
        return tuple(int(b) for b in self.requested[:self.n_requested])

    @property
    def build_id(self):
        return self._text(self.build)

    @property
    def placement(self):
        return self._text(self.placement_sha)

    @property
    def text(self):
        return self._text(self.message)


# int: 0 = answered, nonzero = refused (the C proxy poisons the slot, tells the client the code and
# ends that session only). A void callback would leave ctypes to invent a return value, and the
# proxy would post whatever it invented - the stale-bytes failure the refactor exists to remove.
_SERVER_FN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


def _lib():
    lib = ctypes.CDLL(_LIB_PATH)
    lib.qes_last_error.restype = ctypes.c_char_p
    lib.qes_last_error.argtypes = []
    lib.qes_client_open.restype = ctypes.c_void_p
    lib.qes_client_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                                    ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.POINTER(Nego)]
    lib.qes_client_handle.restype = ctypes.c_void_p
    lib.qes_client_handle.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int]
    lib.qes_client_close.restype = None
    lib.qes_client_close.argtypes = [ctypes.c_void_p]
    lib.qes_server_open.restype = ctypes.c_void_p
    lib.qes_server_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p,
                                    ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                    ctypes.POINTER(Nego)]
    lib.qes_server_set_fn.restype = None
    lib.qes_server_set_fn.argtypes = [ctypes.c_void_p, _SERVER_FN, ctypes.c_void_p]
    lib.qes_server_run.restype = ctypes.c_int
    lib.qes_server_run.argtypes = [ctypes.c_void_p]
    lib.qes_server_shutdown.restype = None
    lib.qes_server_shutdown.argtypes = [ctypes.c_void_p]
    lib.qes_server_free.restype = None
    lib.qes_server_free.argtypes = [ctypes.c_void_p]
    lib.qes_status.restype = ctypes.c_int
    lib.qes_status.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(Nego)]
    # Trailing `rows` on both calls: the request's real row count (0 = the handle's whole bucket).
    # The bucket sizes the slab; the copies, the peer's compute and the response follow `rows`.
    lib.qes_issue.restype = ctypes.c_int
    lib.qes_issue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.qes_collect.restype = ctypes.c_int
    lib.qes_collect.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_int]
    lib.qes_destroy.restype = None
    lib.qes_destroy.argtypes = [ctypes.c_void_p]
    lib.qes_caps.restype = ctypes.c_int
    lib.qes_caps.argtypes = []
    lib.qes_out_mode.restype = ctypes.c_int
    lib.qes_out_mode.argtypes = []
    return lib


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = _lib()
    return _LIB


def last_error() -> str:
    """Human-readable detail for the last failed call on this thread."""
    e = lib().qes_last_error()
    return e.decode(errors = "replace") if e else ""


# Reply format of the 16-bit wire (protocol v5): QES_CAP_OUT_BF16 / QES_CAP_OUT_FP16, and the C
# hook's out_mode numbering that qes_out_mode() returns (0 fp32, 1 bf16, 2 fp16).
CAP_OUT_BF16 = 1
CAP_OUT_FP16 = 2
OUT_NAMES = ("fp32", "bf16", "fp16")


def caps() -> int:
    """This process's wire caps from SEAM_QES_OUT (-1 = an unknown word; the seam refuses to open)."""
    return int(lib().qes_caps())


def out_mode() -> int:
    """Reply format the caps select: 0 fp32, 1 bf16, 2 fp16 (-1 = unknown SEAM_QES_OUT)."""
    return int(lib().qes_out_mode())


def out_mode_of(caps_mask: int) -> int:
    """Reply format of a negotiated caps mask, same numbering as out_mode()."""
    return 1 if caps_mask & CAP_OUT_BF16 else 2 if caps_mask & CAP_OUT_FP16 else 0


_BUILD = None


def build_id() -> str:
    """Build id sent to the peer: SEAM_BUILD_ID, else the worktree's short commit (+-dirty).

    Container trees are rsynced without .git, so a shipped qmesh/BUILD_ID written by the sync
    script is the fallback - without it a GX server reports build=unknown, which is exactly the
    field needed to tell two builds apart when both pairs are busy.
    """
    global _BUILD
    if _BUILD is None:
        b = os.environ.get("SEAM_BUILD_ID", "").strip()
        if not b:
            repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            try:
                b = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                                   capture_output = True, text = True, timeout = 5,
                                   check = True).stdout.strip()
                dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                                       capture_output = True, text = True, timeout = 10,
                                       check = True).stdout.strip()
                if dirty:
                    b += "-dirty"
            except Exception:
                b = ""
        if not b:
            try:
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                       "BUILD_ID"), "r") as fh:
                    b = fh.read().strip()
            except OSError:
                b = ""
        if not b:
            b = "unknown"
        _BUILD = b[:BUILD_LEN - 1]
    return _BUILD


def _int_array(values):
    vals = [int(v) for v in values]
    if not vals:
        raise ValueError("at least one row bucket is required")
    return (ctypes.c_int * len(vals))(*vals)


class Control:
    """One negotiated client session with a running seam server.

    `buckets` is the *request*: the server answers with the subset it can serve and the data
    ports to use, exposed as .buckets / .ports. Raises RuntimeError (with the server's own
    reason) when the server is busy, incompatible or serves none of the requested buckets;
    pass wait=True (SEAM_WAIT=1) to queue behind the active session instead.
    """

    def __init__(self, host: str, port: int = 47311, buckets = (1, 6, 64), wait: bool = False,
                 build: str | None = None, layers: int = NL):
        self.host = host
        self.port = int(port)
        self.seams = {}
        self.nego = Nego()
        req = _int_array(buckets)
        self._h = lib().qes_client_open(host.encode(), self.port,
                                        (build or build_id()).encode(), req, len(req),
                                        int(layers), 1 if wait else 0,
                                        ctypes.byref(self.nego))
        if not self._h:
            raise RuntimeError(f"seam session {host}:{port} refused: {last_error()}")
        self.buckets = self.nego.bucket_list
        self.ports = self.nego.port_list
        self.build = self.nego.build_id
        self.placement_sha = self.nego.placement
        self.layers = int(self.nego.layers)
        self.slots = int(self.nego.slots)
        # The session's wire caps, equal to this side's by construction (the handshake refuses a
        # mismatch). They decide the reply's element type, so the reader of collect() must use it.
        self.caps = int(self.nego.caps)

    def seam(self, dev_name: str, gid_idx: int, bucket: int, device_ordinal: int = 0) -> "Seam":
        """Data handle for a negotiated bucket (created on first use, one per bucket/device)."""
        key = (int(bucket), int(device_ordinal))
        if key in self.seams:
            return self.seams[key]
        if self._h is None:
            raise RuntimeError("seam session is closed")
        h = lib().qes_client_handle(self._h, dev_name.encode(), int(gid_idx), int(bucket),
                                    int(device_ordinal))
        if not h:
            raise RuntimeError(f"seam data handle rows={bucket} on {self.host} failed: "
                               f"{last_error()}")
        seam = Seam(self, h, bucket, device_ordinal)
        self.seams[key] = seam
        return seam

    def close(self):
        for seam in list(self.seams.values()):
            seam.close()
        self.seams.clear()
        if self._h:
            lib().qes_client_close(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class Seam:
    """One bucket's RDMA handle inside a Control session."""

    def __init__(self, ctl: Control, handle, bucket: int, device_ordinal: int = 0):
        self.ctl = ctl
        self._h = handle
        self.bucket = int(bucket)
        self.rows = int(bucket)   # bucket == request rows, kept for the old call sites
        self.device_ordinal = int(device_ordinal)

    def issue(self, layer, x_ptr, ids_ptr, w_ptr, stream_ptr, n_cold: int = 0, rows: int = 0):
        """n_cold is the layer's cold-expert count on this placement: the client proxy refuses to
        send an id outside [-1, n_cold) instead of letting the server's placement check abort the
        server process. 0 = unknown, no check.

        rows is how many of this handle's bucket rows the request really carries (0 = all of them).
        The peer computes that many rows and answers that many, so a prefill chunk smaller than the
        granted bucket pays for its own rows only; see SlotHdr::rows."""
        if not self._h:
            raise RuntimeError("seam handle is closed")
        rc = lib().qes_issue(self._h, layer, x_ptr, ids_ptr, w_ptr, stream_ptr, int(n_cold),
                             int(rows))
        if rc:
            raise RuntimeError(f"qes_issue layer={layer} rc={rc}: {last_error()}")

    def collect(self, layer, out_ptr, stream_ptr, rows: int = 0):
        """rows must be the row count the matching issue() carried (0 = the whole bucket): only
        those rows are copied out of the response, which is all the peer answered."""
        if not self._h:
            raise RuntimeError("seam handle is closed")
        rc = lib().qes_collect(self._h, layer, out_ptr, stream_ptr, int(rows))
        if rc:
            raise RuntimeError(f"qes_collect layer={layer} rc={rc}: {last_error()}")

    def close(self):
        if self._h:
            lib().qes_destroy(self._h)
            self._h = None
        if self.ctl is not None:
            key = (self.bucket, self.device_ordinal)
            if self.ctl.seams.get(key) is self:
                del self.ctl.seams[key]


class Server:
    """Long-lived GX listener: one pinned slab + MR per row bucket, one session at a time."""

    def __init__(self, dev_name: str, gid_idx: int, port: int = 47311,
                 device_ordinal: int = 0, buckets = (1, 2, 3, 4, 5, 6, 64),
                 build: str | None = None, placement_sha: str = "", wait: bool = False,
                 slots: int = 1):
        self.port = int(port)
        # Payload slots per handle: a client may have this many requests in flight, so a prefill
        # chunk split into micro-batches keeps the wire and the GX compute overlapping instead of
        # alternating. Costs slots * rows * 30.2 KiB of pinned memory per bucket on the GX.
        self.slots = int(slots)
        if self.slots < 1 or self.slots > MAX_SLOTS:
            raise ValueError(f"slots must be 1..{MAX_SLOTS} (got {slots})")
        self.buckets = tuple(int(b) for b in buckets)
        self.nego = Nego()
        req = _int_array(self.buckets)
        self._h = lib().qes_server_open(dev_name.encode(), int(gid_idx), self.port,
                                        int(device_ordinal), req, len(req),
                                        (build or build_id()).encode(),
                                        str(placement_sha).encode(), 1 if wait else 0,
                                        self.slots, ctypes.byref(self.nego))
        if not self._h:
            raise RuntimeError(f"seam server open on {dev_name} port {port} failed: "
                               f"{last_error()}")
        self._cb = None

    def set_fn(self, fn):
        """fn(layer, rows, x_bf16, ids_i32, w_bf16, out_f32) -> int: raw pinned-slab pointers as
        ints; see worker_seam_server.py for the typed numpy wrapper. 0 = answered, nonzero = refused."""
        def tramp(ctx, layer, rows, x, ids, w, out):
            # A raise here would be swallowed by ctypes and its return value invented, which is the
            # one way a stale response could still be posted: refuse instead.
            try:
                return int(fn(layer, rows, x, ids, w, out) or 0)
            except BaseException as e:   # noqa: BLE001 - the proxy must hear "no answer"
                print(f"SEAM_COMPUTE_REFUSED hook raised {type(e).__name__}: {e}", flush = True)
                return 5
        self._cb = _SERVER_FN(tramp)   # keep alive: the C side calls it from proxy threads
        lib().qes_server_set_fn(self._h, self._cb, None)

    def run(self):
        """Serves sessions until shutdown(). Raises RuntimeError on a fatal local error."""
        if not self._h:
            raise RuntimeError("seam server is closed")
        rc = lib().qes_server_run(self._h)
        if rc:
            raise RuntimeError(f"seam server on port {self.port} failed: {last_error()}")

    def shutdown(self):
        if self._h:
            lib().qes_server_shutdown(self._h)

    def close(self):
        if self._h:
            lib().qes_server_free(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        self.close()
        return False


def status(host: str, port: int = 47311) -> Nego:
    """Queries a running server without starting or disturbing a session."""
    nego = Nego()
    if lib().qes_status(host.encode(), int(port), ctypes.byref(nego)):
        raise RuntimeError(f"seam status {host}:{port} failed: {last_error()}")
    return nego
