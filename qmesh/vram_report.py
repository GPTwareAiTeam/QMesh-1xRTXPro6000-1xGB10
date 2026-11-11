"""P7 VRAM breakdown, from inside the serving process (VRAM_REPORT=1 in serve_pair.py). Every CUDA tensor reachable
from the target model, its KV cache, the drafter model / cache and ColdSteal is counted once (by storage), grouped
by owner; the rest of the process' footprint is split into allocator caching, CUDA graph pools / workspaces (torch
reserved but not referenced), and non-torch (CUDA context, cuBLAS / NCCL workspaces, pinned mappings' device side).
Logged as VRAM_REPORT lines, in GiB.
"""
import gc

import torch

GiB = 2 ** 30


def _tensors(root, seen_obj, out, depth=0):
    if depth > 10 or id(root) in seen_obj:
        return
    seen_obj.add(id(root))
    if isinstance(root, torch.Tensor):
        if root.is_cuda:
            out.append(root)
        return
    if isinstance(root, (str, bytes, int, float, bool, type(None))):
        return
    if isinstance(root, dict):
        for v in list(root.values()):
            _tensors(v, seen_obj, out, depth + 1)
        return
    if isinstance(root, (list, tuple, set)):
        for v in list(root):
            _tensors(v, seen_obj, out, depth + 1)
        return
    d = getattr(root, "__dict__", None)
    if d:
        for v in list(d.values()):
            _tensors(v, seen_obj, out, depth + 1)


def _storages(roots, claimed):
    out = {}
    for r in roots:
        ts = []
        _tensors(r, set(), ts)
        for t in ts:
            try:
                st = t.untyped_storage()
                p, n = st.data_ptr(), st.nbytes()
            except Exception:  # noqa: BLE001
                continue
            if p and p not in claimed:
                out[p] = max(out.get(p, 0), n)
    return out


def _group(key):
    if key.startswith("mtp."):
        return None
    if ".ffn.experts" in key or key.endswith(".ffn") or ".ffn." in key and "shared" not in key and "gate" not in key:
        return "routed experts (hot) + MoE tables"
    if ".ffn.shared" in key:
        return "shared experts"
    if ".ffn.gate" in key:
        return "router"
    if ".attn" in key:
        return "attention"
    if "engram" in key:
        return "engram (projection; tables on host)"
    if "embed" in key:
        return "embedding"
    if "head" in key:
        return "lm head"
    if "vision" in key or "aligner" in key:
        return "vision tower"
    return "norms / hc / other"


def report(tag, model, cache, draft_model, draft_cache):
    torch.cuda.synchronize()
    gc.collect()
    free, total = torch.cuda.mem_get_info(0)
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    claimed = {}
    rows = []
    # Model weights per group (each module's own tensors, leaf-first so a parent does not claim its children's)
    groups = {}
    for m in reversed(list(model.modules)):
        key = getattr(m, "key", "") or ""
        g = _group(key)
        if g is None:
            continue
        st = _storages([m], claimed)
        groups[g] = groups.get(g, 0) + sum(st.values())
        claimed.update(st)
    rows += sorted(groups.items(), key=lambda x: -x[1])
    for name, root in (("KV cache + recurrent state (target)", cache), ("drafter weights (MTP)", draft_model),
                       ("KV cache (drafter)", draft_cache)):
        if root is None:
            continue
        st = _storages([root], claimed)
        rows.append((name, sum(st.values())))
        claimed.update(st)
    try:
        from qmesh import coldsteal
        mgr = coldsteal.manager()
        slots = sum(L.slots.untyped_storage().nbytes() for L in mgr.layers)
        rows.append((f"ColdSteal cache slots ({len(mgr.layers)} layers x {mgr.layers[0].S if mgr.layers else 0})", slots))
        st = _storages(mgr.layers, claimed)
        for L in mgr.layers:
            st.pop(L.slots.untyped_storage().data_ptr(), None)
        rows.append(("ColdSteal tables / maps", sum(st.values())))
        claimed.update(st)
        for L in mgr.layers:
            claimed[L.slots.untyped_storage().data_ptr()] = 0
    except Exception as e:  # noqa: BLE001
        rows.append((f"ColdSteal: {e}", 0))
    counted = sum(n for _, n in rows)
    rows.append(("other live torch tensors (activations, workspaces, graph statics)", max(allocated - counted, 0)))
    rows.append(("torch cached, not in use (allocator + graph pools)", reserved - allocated))
    rows.append(("non-torch (CUDA context, cuBLAS / driver, other)", total - free - reserved))
    rows.append(("free", free))
    for name, n in rows:
        print(f"VRAM_REPORT {tag} {n / GiB:7.2f} GiB  {name}", flush=True)
    print(f"VRAM_REPORT {tag} {total / GiB:7.2f} GiB  TOTAL (allocated {allocated / GiB:.2f}, reserved "
          f"{reserved / GiB:.2f})", flush=True)
