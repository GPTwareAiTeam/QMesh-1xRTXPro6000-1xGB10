"""
Routed-expert trellis layout of a pack (permutation: qmesh/pair_layout.py; device side: quant/trellis_pair.cuh;
notes/CoopKernel2.md §11).

A safetensors file is in the paired coop layout iff its header metadata carries LAYOUT_KEY = LAYOUT_PAIRED; every
other file is plain. Inside a paired file, routed-expert trellises at integer 2 / 3 / 4 bpw are permuted, every other
tensor is as written by the quantizer. The layout is a property of the pack: BlockSparseMLP reads it per layer at load
and passes `paired` to every routed-expert kernel entry point (coop, exl3_moe, reconstruct). Dense / attention /
shared-expert linears are always plain.

Fail-closed rules: a layer whose routed experts span both layouts is refused; a paired pack on an extension without
paired support (HAS_COOP_PAIR) is refused; QMESH_COOP_PAIR, when set, must match the pack (an operator's assertion of
which pack they meant to serve).
"""
import os

LAYOUT_KEY = "qmesh_trellis_layout"
LAYOUT_PAIRED = "coop_pair_v1"
ENV = "QMESH_COOP_PAIR"


def file_layout(metadata: dict | None) -> str:
    """Layout of one safetensors file from its header metadata: LAYOUT_PAIRED or "plain" """
    layout = (metadata or {}).get(LAYOUT_KEY, "plain")
    if layout not in ("plain", LAYOUT_PAIRED):
        raise RuntimeError(f"unknown trellis layout tag {layout!r}")
    return layout


def tensor_layout(stc, key: str) -> str:
    """Layout of the file that holds tensor `key` in a SafetensorsCollection"""
    return file_layout(stc.file_headers[stc.tensor_file_map[key]].get("__metadata__"))


def pairable(K: float) -> bool:
    """Rates stored paired inside a paired file (the rule trellis_pair.cuh applies on the device)"""
    return K in (2, 3, 4)


def check_runtime(pack_paired: bool, ext) -> None:
    """Fail closed: the extension must support the pack's layout, and QMESH_COOP_PAIR (if set) must match it"""
    if pack_paired and not getattr(ext, "HAS_COOP_PAIR", False):
        raise RuntimeError("paired trellis pack, but this extension has no paired-layout support (HAS_COOP_PAIR)")
    want = os.environ.get(ENV)
    if want is not None and (want == "1") != pack_paired:
        raise RuntimeError(f"paired trellis layout mismatch: {ENV}={want} but the pack is "
                           f"{'paired' if pack_paired else 'plain'}")
