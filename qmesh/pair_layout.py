"""
Paired routed-expert trellis layout (device side quant/trellis_pair.cuh, loader exllamav3/util/trellis_layout.py,
notes/CoopKernel2.md §11).

One permutation, shared by the layer repack tool below, the coop bench and the ColdSteal pool loader: the words of
an expert trellis (k/16, n/16, 16 K) int16 at K = 2, 3, 4 (mul1), viewed as uint32 per k-slice, are regrouped per
pair of k-slices (2j, 2j + 1) and per warp column chunk of WNT = 2 tiles so that lane L's words of both slices are
contiguous: one 16 B (K3, K4) or 8 B (K2) load per lane fills two ring slots. Size, shape and dtype are unchanged;
only the word order inside each slice-pair region moves. The words a lane reads, and so every output, are the same.

Plain order inside one slice and one chunk (what gemv_tile reads today): word l * NL + L for lane L < NL, load l <
LOADS, with (LOADS, NL) = (2, 32) at K4, (2, 24) at K3, (1, 32) at K2. Paired order inside one (pair, chunk) block:
lane-major, (L, slice, l).

    python3 -m qmesh.pair_layout --src layers.N.safetensors --dst out.safetensors     # repack one layer file (no re-encode)
    python3 -m qmesh.pair_layout --selftest                                            # round trip on random tensors
"""
import argparse
import re

import torch

WNT = 2                                   # MOE_COOP_WNT: tiles per warp column chunk
GEOM = {2: (1, 32), 3: (2, 24), 4: (2, 32)}   # K -> (LOADS, NL)
EXPERT_KEY = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.w[123]\.trellis$")

# Layout tag, runtime switch and the fail-closed check live in the engine (exllamav3/util/trellis_layout.py, which
# the BlockSparseMLP loader calls); re-exported here for the repack tool and the ColdSteal pool
from exllamav3.util.trellis_layout import ENV, LAYOUT_KEY, LAYOUT_PAIRED, check_runtime, file_layout  # noqa: F401


def _view(tr: torch.Tensor, K: int):
    assert tr.dtype == torch.int16 and tr.dim() == 3 and tr.shape[-1] == 16 * K, (tr.dtype, tr.shape, K)
    loads, nl = GEOM[K]
    ks, nt, _ = tr.shape
    assert ks % 2 == 0 and nt % WNT == 0, ("slice pairs / column chunks", tr.shape)
    assert loads * nl == WNT * 8 * K
    w = tr.contiguous().view(torch.int32).view(ks // 2, 2, nt // WNT, loads, nl)
    return w, (ks, nt)


def pair_permute(tr: torch.Tensor, K: int) -> torch.Tensor:
    """Plain trellis -> paired trellis (same shape / dtype / device)"""
    w, (ks, nt) = _view(tr, K)
    return w.permute(0, 2, 4, 1, 3).contiguous().view(torch.int16).view(ks, nt, 16 * K)


def pair_unpermute(tr: torch.Tensor, K: int) -> torch.Tensor:
    """Paired trellis -> plain trellis"""
    loads, nl = GEOM[K]
    ks, nt, _ = tr.shape
    w = tr.contiguous().view(torch.int32).view(ks // 2, nt // WNT, nl, 2, loads)
    return w.permute(0, 3, 1, 4, 2).contiguous().view(torch.int16).view(ks, nt, 16 * K)


def pairable(K) -> bool:
    """Integer 2 / 3 / 4 bpw (the rates the paired layout covers; trellis_pair.cuh pairable)"""
    return float(K) in (2.0, 3.0, 4.0)


def pair_permute_bytes_(buf: torch.Tensor, ks: int, nt: int, K: int) -> None:
    """In place on a contiguous uint8 region holding one plain trellis of shape (ks, nt, 16 K) int16 (e.g. a
    ColdSteal pool row's g_t / u_t / d_t region): rewrite it in the paired layout. Same bytes, permuted"""
    assert buf.dtype == torch.uint8 and buf.is_contiguous() and buf.numel() == ks * nt * 32 * K, (buf.dtype, buf.numel())
    tr = buf.view(torch.int16).view(ks, nt, 16 * K)
    tr.copy_(pair_permute(tr, K))


def repack_file(src: str, dst: str) -> dict:
    """Copy one layer's safetensors file, every routed-expert trellis at K 2 / 3 / 4 paired, everything else as is"""
    from safetensors import safe_open
    from safetensors.torch import save_file
    out, stats = {}, {"paired": 0, "kept": 0}
    with safe_open(src, framework = "pt", device = "cpu") as f:
        meta = f.metadata()
        for k in f.keys():
            t = f.get_tensor(k)
            K = t.shape[-1] / 16 if t.dim() == 3 and t.dtype == torch.int16 else None
            if EXPERT_KEY.match(k) and K is not None and pairable(K):
                K = int(K)
                p = pair_permute(t, K)
                assert torch.equal(pair_unpermute(p, K), t), k
                out[k] = p
                stats["paired"] += 1
            else:
                out[k] = t
                stats["kept"] += 1
    meta = dict(meta or {})
    assert file_layout(meta) == "plain", (src, "already paired")
    meta[LAYOUT_KEY] = LAYOUT_PAIRED
    save_file(out, dst, metadata = meta)
    return stats


def selftest():
    g = torch.Generator().manual_seed(0)
    for K in (2, 3, 4):
        for ks, nt in ((320, 144), (144, 320), (8, 4)):
            t = torch.randint(-32768, 32767, (ks, nt, 16 * K), generator = g, dtype = torch.int32).to(torch.int16)
            p = pair_permute(t, K)
            assert p.shape == t.shape and torch.equal(pair_unpermute(p, K), t)
            # the words the kernel's lane L reads for slice s, load l, chunk c land where gemv_tile's paired path reads
            loads, nl = GEOM[K]
            tw = t.view(torch.int32).view(ks, -1)
            pw = p.view(torch.int32).view(-1)
            gw = WNT * 8 * K
            stride = nt * 8 * K
            for (s, c, L, l) in ((0, 0, 0, 0), (1, 1, nl - 1, loads - 1), (ks - 1, nt // WNT - 1, 3, 0)):
                plain = tw[s, c * gw + l * nl + L]
                j, h = divmod(s, 2)
                paired = pw[2 * j * stride + c * 2 * gw + L * 2 * loads + h * loads + l]
                assert plain == paired, (K, s, c, L, l)
    print("pair_layout selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--selftest", action = "store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        print(repack_file(a.src, a.dst))
