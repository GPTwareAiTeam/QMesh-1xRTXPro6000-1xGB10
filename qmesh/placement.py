"""The one cold-expert placement resolver: every loader and the seam client read it here.

A pack ships `PACK_MANIFEST.json`, whose `placement` names the file the pack was cut against, and that
file's sha256 is the `placement_sha` the GX seam server puts in its Hello. So the file a client loads
must be the pack's, not a repo default: the repo default is the 2.8535 pack's table, and against the
4620 packs it makes the client translate picks with the wrong per-layer cold count (layer 0: 259 cold
there, 248 here), which the server correctly refuses as "cold ids out of range" and then fail-stops on
(2026-11-01 18:03, GX2 arm A prefill: 2 of 384 picks at 251 for 248 cold -- exactly the old table's
range, while the same run's identity phase, which resolved through qmesh/e2e_prompts.py, was clean).

Order: an explicit PLACEMENT wins (operators overriding a pack on purpose), then the pack's manifest,
then the repo default. A manifest that names a placement this process cannot honour *raises*: the pack
is cut against exactly one table, so loading another one is how a client ends up asking for an expert
its hot half does not hold (2026-11-05 04:40, arm N on pack mix4460-nq: the manifest named a
external mount path the dev container does not mount, the client silently loaded the repo default and died
on `No tensors found for layers.0.ffn.experts.21.w3`). `load()` also publishes its sha through
`current_sha()`, so a client can refuse a server whose Hello advertises a different table *before* it
sends the first request.
"""
from __future__ import annotations

import hashlib
import json
import os

DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placement.final.json")
MANIFEST = "PACK_MANIFEST.json"

_last = {"path": None, "sha": None}


class PlacementError(RuntimeError):
    """A pack's manifest names a placement this process cannot honour."""


def _manifest(ckpt):
    """The pack's manifest as a dict, or None when the pack ships none (or it is not a JSON object)."""
    try:
        with open(os.path.join(ckpt, MANIFEST)) as fh:
            man = json.load(fh)
    except (OSError, ValueError):
        return None
    return man if isinstance(man, dict) else None


def _readable(path):
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False


def _check_sha(man, path, ckpt):
    """`path` if its sha256 is the manifest's `placement_sha256` (or the manifest records none)."""
    want = man.get("placement_sha256")
    if not isinstance(want, str) or not want:
        return path
    got = sha(path)
    if got != want:
        raise PlacementError(
            f"{os.path.join(ckpt, MANIFEST)} records placement_sha256 {want}, but {path} is {got} "
            f"(checkpoint {ckpt}); refusing to load a hot/cold split that is not this pack's")
    return path


def _manifest_placement(ckpt, man):
    """Absolute path of the placement `man` names for `ckpt`, or None when the pack names none.

    Raises PlacementError instead of returning None when the manifest names a placement that cannot be
    opened, or whose sha256 is not its `placement_sha256`: falling back to DEFAULT would hand the client
    a different table than the pack was cut against, and the mismatch only surfaces later as a missing
    tensor (or, worse, as numbers from a different split).
    """
    p = man.get("placement")
    if not isinstance(p, str) or not p:
        return None
    # Packs ship it relative to the pack root; an absolute entry (older packs) is joined to nothing.
    tried = (os.path.join(ckpt, p), p)
    for cand in tried:
        if _readable(cand):
            return _check_sha(man, os.path.abspath(cand), ckpt)
    raise PlacementError(
        f"{os.path.join(ckpt, MANIFEST)} names placement {p!r}, but neither {tried[0]} nor {tried[1]} "
        f"opens (checkpoint {ckpt}); refusing to fall back to {DEFAULT}. Stage the pack's placement "
        f"inside the pack so a client on another host can read it")


def resolve(ckpt = None):
    """Path of the placement this process must use for `ckpt` (default: $CHECKPOINT).

    An explicit $PLACEMENT wins unchanged (an operator overriding a pack on purpose). Otherwise the pack's
    manifest decides, and a manifest naming a placement that cannot be opened or whose sha256 is not the
    one it records raises PlacementError. DEFAULT serves only a pack that names no placement at all.
    """
    explicit = os.environ.get("PLACEMENT")
    if explicit:
        return explicit
    ckpt = ckpt or os.environ.get("CHECKPOINT")
    if ckpt:
        man = _manifest(ckpt)
        if man is not None:
            p = _manifest_placement(ckpt, man)
            if p is not None:
                return p
            # A manifest that keeps only the sha still pins the table: the default must be that one.
            _check_sha(man, DEFAULT, ckpt)
    return DEFAULT


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load(ckpt = None):
    """(placement dict, path, sha256 hex) for `ckpt`; remembers the sha for the seam handshake."""
    path = resolve(ckpt)
    digest = sha(path)
    with open(path) as fh:
        placement = json.load(fh)
    _last["path"], _last["sha"] = path, digest
    return placement, path, digest


def hot(placement):
    """The per-module hot list keyed the way the model's modules are: {"layers.N.ffn": [expert ids]}."""
    return {f"layers.{layer}.ffn": v["hot"] for layer, v in placement["layers"].items()}

def warm(placement):
    """The per-module cold list **in the file's order**, keyed like hot().

    The seam's cold-local ids are positions in the server's `layers.N.warm`, so the client must
    number its cold experts the same way. Ascending order happened to agree with the unranked
    file and silently stops agreeing with a reordered (ranked) one: every cold id then names a
    different expert on the two sides, which the transport cannot detect. block_sparse_mlp_cpu
    therefore prefers this list and asserts hot + warm is a permutation of 0..E-1.
    """
    return {f"layers.{layer}.ffn": v["warm"] for layer, v in placement["layers"].items()
            if "warm" in v}


def current_sha():
    """sha256 of the placement this process loaded, or None before any load."""
    return _last["sha"]


def current_path():
    return _last["path"]
