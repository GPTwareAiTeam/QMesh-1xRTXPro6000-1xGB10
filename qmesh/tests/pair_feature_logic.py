#!/usr/bin/env python3
"""Offline checks for the two load-time knobs the pair release added: prefix cache and chunk sizing.

Both are configured from the environment and both decide something expensive at load -- the recurrent-state
pool's schedule, and the prefill chunk the generator starts with -- so a typo or an empty-string environment
variable costs a full pair start (two boxes, ~7 s P7 load plus GX worker) to discover. Nothing here touches a
GPU: the chunk picker takes its free-memory reading from `torch.cuda`, which this test replaces with numbers,
and the prefix cache is pure environment parsing.

The assertions are deliberately about *selection*, not about re-deriving the fit arithmetic: the chosen chunk
must fit (`prefill_rise(0, rows) <= room`) and the next larger ladder entry must not, which is the property
CHUNK_AUTO promises and the one an edit to the ladder or the reserve would break.

  python3 -u qmesh/tests/pair_feature_logic.py
  # expect: PASS lines and exit 0
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

FAILS = []


def check(name, cond, detail = ""):
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' ' + detail) if detail else ''}", flush = True)
    if not cond:
        FAILS.append(name)


def with_env(mod_name, env, fn):
    """Imports `mod_name` under `env` (both set and *unset* keys) and calls fn(module).

    Module-level parsing is the whole point of these knobs, so the test has to re-import rather than poke
    attributes: the empty-vs-unset distinction (`EXL3_PREFIX_CACHE_INTERVAL=""`) only exists at import.
    """
    saved = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        mod = importlib.import_module(mod_name)
        mod = importlib.reload(mod)
        return fn(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---- prefix cache: defaults, disable paths, interval validation ---------------------------------------
def pc_defaults(pc):
    check("pc.enabled_default", pc.enabled() is True)
    check("pc.capacity_default_mib", pc.PREFIX_CACHE_CAPACITY == 4 * 1024 ** 3,
          f"={pc.PREFIX_CACHE_CAPACITY}")
    check("pc.min_match_default", pc.PREFIX_CACHE_MIN_MATCH == 256, f"={pc.PREFIX_CACHE_MIN_MATCH}")
    kw = pc.generator_kwargs()
    check("pc.kwargs_default_no_interval",
          kw == {"recurrent_cache_size": 4 * 1024 ** 3}, f"={kw}")
    check("pc.summary_interval_model", pc.summary().endswith("interval=model"), pc.summary())


def pc_disabled(pc):
    check("pc.disabled_flag", pc.enabled() is False)
    check("pc.kwargs_disabled_zero_pool", pc.generator_kwargs() == {"recurrent_cache_size": 0},
          f"={pc.generator_kwargs()}")
    check("pc.summary_disabled", pc.summary() == "PREFIX_CACHE enabled=0", pc.summary())
    check("pc.check_false_when_disabled", pc.check() is False)


def pc_capacity_zero(pc):
    check("pc.capacity_zero_disables", pc.enabled() is False and pc.check() is False)


def pc_interval(pc):
    check("pc.interval_set", pc.PREFIX_CACHE_INTERVAL == 2048, f"={pc.PREFIX_CACHE_INTERVAL}")
    check("pc.interval_pp_follows_interval",
          pc.PREFIX_CACHE_INTERVAL_PP == 2048, f"={pc.PREFIX_CACHE_INTERVAL_PP}")
    kw = pc.generator_kwargs()
    check("pc.kwargs_interval",
          kw.get("recurrent_checkpoint_interval") == 2048 and
          kw.get("recurrent_checkpoint_interval_pp") == 2048, f"={kw}")


def pc_interval_pp_only(pc):
    check("pc.interval_pp_override", pc.PREFIX_CACHE_INTERVAL_PP == 1024,
          f"={pc.PREFIX_CACHE_INTERVAL_PP}")


def pc_interval_not_aligned(pc):
    try:
        pc.check()
        raised = False
    except ValueError as exc:
        raised = "positive multiple of 256" in str(exc)
    check("pc.interval_unaligned_raises", raised)


def pc_interval_zero(pc):
    try:
        pc.check()
        raised = False
    except ValueError:
        raised = True
    check("pc.interval_zero_raises", raised)


def pc_apply(pc):
    class Gen:
        prefix_cache_min_match = 0
    g = Gen()
    pc.apply(g)
    check("pc.apply_sets_min_match_on", g.prefix_cache_min_match == 256, f"={g.prefix_cache_min_match}")


def pc_apply_disabled(pc):
    class Gen:
        prefix_cache_min_match = 256
    g = Gen()
    pc.apply(g)
    check("pc.apply_clears_min_match_off", g.prefix_cache_min_match is None, f"={g.prefix_cache_min_match}")


# ---- chunk sizing: fixed passthrough, auto pick, empty-string, ladder parsing --------------------------
class StubCuda:
    """Replaces the three torch.cuda readings resolve_max_chunk makes; records whether it was consulted."""
    def __init__(self, torch, free_gib, reserved_gib = 0.0, allocated_gib = 0.0):
        self.calls = 0
        self.free, self.reserved, self.allocated = free_gib, reserved_gib, allocated_gib
        cuda = torch.cuda
        cuda.mem_get_info = lambda device = None: (int(self.free * 2 ** 30), 0)
        # mem_get_info must not be counted twice by the two lambdas: count in reserved/allocated only
        cuda.memory_reserved = self._count(lambda: int(self.reserved * 2 ** 30))
        cuda.memory_allocated = self._count(lambda: int(self.allocated * 2 ** 30))

    def _count(self, fn):
        def inner(device = None):
            self.calls += 1
            return fn()
        return inner


def chunk_fixed(pl, torch_mod):
    stub = StubCuda(torch_mod, 40.0)
    rows = pl.resolve_max_chunk("cuda:0", "8192")
    check("chunk.fixed_passthrough", rows == 8192, f"={rows}")
    check("chunk.fixed_no_cuda_read", stub.calls == 0, f"calls={stub.calls}")
    rows = pl.resolve_max_chunk("cuda:0", 4096)
    check("chunk.fixed_int_passthrough", rows == 4096, f"={rows}")


def chunk_auto_roomy(pl, torch_mod):
    stub = StubCuda(torch_mod, 40.0)
    rows = pl.resolve_max_chunk("cuda:0", "auto")
    room = (40.0 * 2 ** 30) - (pl.PREFILL_MARGIN_MIB + pl.CHUNK_VRAM_RESERVE_MIB +
                               (pl.DSPARK_RESERVE_MIB if pl.CHUNK_DSPARK else 0.0)) * 2 ** 20
    check("chunk.auto_reads_cuda", stub.calls == 2, f"calls={stub.calls}")
    check("chunk.auto_within_ladder", rows <= pl.PREFILL_CHUNK_LADDER[0],
          f"rows={rows} top={pl.PREFILL_CHUNK_LADDER[0]}")
    check("chunk.auto_is_ladder_entry", rows in pl.PREFILL_CHUNK_LADDER, f"rows={rows}")
    check("chunk.auto_under_room", pl.prefill_rise(0, rows) <= room,
          f"rise={pl.prefill_rise(0, rows) / 2 ** 20:.0f}MiB room={room / 2 ** 20:.0f}MiB")
    bigger = [c for c in pl.PREFILL_CHUNK_LADDER if c > rows]
    if bigger:
        check("chunk.auto_next_ladder_too_big", pl.prefill_rise(0, bigger[-1]) > room,
              f"next={bigger[-1]} rise={pl.prefill_rise(0, bigger[-1]) / 2 ** 20:.0f}MiB")
    else:
        check("chunk.auto_next_ladder_too_big", True, "top of ladder")


def chunk_auto_tight(pl, torch_mod):
    stub = StubCuda(torch_mod, 2.0)
    rows = pl.resolve_max_chunk("cuda:0", "auto")
    check("chunk.auto_tight_floors_at_min", rows == pl.PREFILL_CHUNK_MIN,
          f"rows={rows} min={pl.PREFILL_CHUNK_MIN}")


def chunk_auto_empty(pl, torch_mod):
    stub = StubCuda(torch_mod, 40.0)
    rows = pl.resolve_max_chunk("cuda:0", "")
    check("chunk.empty_string_is_auto", rows in pl.PREFILL_CHUNK_LADDER and stub.calls == 2,
          f"rows={rows} calls={stub.calls}")


def chunk_ladder_parse(pl):
    check("chunk.ladder_descending_deduped",
          list(pl.PREFILL_CHUNK_LADDER) == sorted(set(pl.PREFILL_CHUNK_LADDER), reverse = True),
          f"={pl.PREFILL_CHUNK_LADDER}")
    check("chunk.ladder_page_aligned",
          all(c >= 256 and c % 256 == 0 for c in pl.PREFILL_CHUNK_LADDER), f"={pl.PREFILL_CHUNK_LADDER}")
    check("chunk.ladder_drops_junk", 999999 not in pl.PREFILL_CHUNK_LADDER)


def main():
    import torch  # noqa: F401  (imported once here; the stubs below only replace torch.cuda readings)

    with_env("qmesh.prefix_cache", {
        "EXL3_PREFIX_CACHE": None, "EXL3_PREFIX_CACHE_CAPACITY": None,
        "EXL3_PREFIX_CACHE_MIN_MATCH": None, "EXL3_PREFIX_CACHE_INTERVAL": None,
        "EXL3_PREFIX_CACHE_INTERVAL_PP": None}, pc_defaults)
    with_env("qmesh.prefix_cache", {"EXL3_PREFIX_CACHE": "0"}, pc_disabled)
    with_env("qmesh.prefix_cache", {"EXL3_PREFIX_CACHE": None, "EXL3_PREFIX_CACHE_CAPACITY": "0"},
             pc_capacity_zero)
    with_env("qmesh.prefix_cache", {"EXL3_PREFIX_CACHE_INTERVAL": "2048"}, pc_interval)
    with_env("qmesh.prefix_cache", {"EXL3_PREFIX_CACHE_INTERVAL": "2048",
                                    "EXL3_PREFIX_CACHE_INTERVAL_PP": "1024"}, pc_interval_pp_only)
    with_env("qmesh.prefix_cache", {"EXL3_PREFIX_CACHE_INTERVAL": "1000"}, pc_interval_not_aligned)
    with_env("qmesh.prefix_cache", {"EXL3_PREFIX_CACHE_INTERVAL": "0"}, pc_interval_zero)
    with_env("qmesh.prefix_cache", {"EXL3_PREFIX_CACHE_INTERVAL": None}, pc_apply)
    with_env("qmesh.prefix_cache", {"EXL3_PREFIX_CACHE": "0"}, pc_apply_disabled)

    with_env("qmesh.prefill_limit", {"MAX_CHUNK": None, "PREFILL_CHUNK_STEPS": None,
                                     "PREFILL_CHUNK_MIN": None, "CHUNK_DSPARK": None,
                                     "DSPARK_RESERVE_MIB": None}, lambda pl: chunk_fixed(pl, torch))
    with_env("qmesh.prefill_limit", {"PREFILL_CHUNK_STEPS": None, "PREFILL_CHUNK_MIN": None},
             lambda pl: chunk_auto_roomy(pl, torch))
    with_env("qmesh.prefill_limit", {"PREFILL_CHUNK_STEPS": None, "PREFILL_CHUNK_MIN": None},
             lambda pl: chunk_auto_tight(pl, torch))
    with_env("qmesh.prefill_limit", {"PREFILL_CHUNK_STEPS": None}, lambda pl: chunk_auto_empty(pl, torch))
    with_env("qmesh.prefill_limit", {"PREFILL_CHUNK_STEPS": "999999,abc,192,4096,4096,8192"},
             chunk_ladder_parse)

    print(f"RESULT {'PASS' if not FAILS else 'FAIL'} fails={len(FAILS)}", flush = True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
