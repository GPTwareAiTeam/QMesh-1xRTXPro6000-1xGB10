"""
Prefix cache (recurrent-state pool) configuration for the pair.

DeepSeek-V4 carries recurrent state per sequence (DSpark / pool compressors), so a KV-page prefix match
alone cannot resume a prompt: the resumed position must have a stashed state. exllamav3 keeps those
checkpoints in a system-memory pool (`Generator.recurrent_cache_size`, default 4 GiB) every
`recurrent_checkpoint_interval` tokens, and `PageTable.allocate_pages` only lets a matched prefix restore a
checkpoint when the match clears `generator.prefix_cache_min_match` tokens -- a page hash fixes the whole
chain, so a short match otherwise claims a restore from state stashed at another prompt's checkpoint and
pays the replay to the checkpoint anyway.

The pool was already on by default here (nothing disabled it), but the pair's drivers never named it, so
there was no way to A/B a repeated prefix or to require a minimum match. These four knobs configure it:

  EXL3_PREFIX_CACHE            1 enables the pool, 0 disables it (capacity 0, no checkpoints)
  EXL3_PREFIX_CACHE_CAPACITY   pool bytes (default 4 GiB); 0 disables
  EXL3_PREFIX_CACHE_MIN_MATCH  resume position, tokens, below which a checkpoint may not be restored
                               (default 256, one page = permissive). A restore resumes at the last page
                               whose checkpoint exists, so with the model's own 2048/32768 spacing every
                               checkpoint already sits past a 256 threshold: the gate is inert at this
                               default and only bites with page-granular checkpoints or a threshold
                               longer than the spacing.
  EXL3_PREFIX_CACHE_INTERVAL   checkpoint spacing, tokens; must be a positive multiple of the page size.
                               Unset = leave the model's schedule alone (DeepSeek-V4 declares 2048 for
                               generation and 32768 for prompt ingestion, which is what this pair has
                               always run and what every quoted number was taken with). The 2xRTX server
                               defaults it to 1024, which buys finer reuse at 2x the recurrent state and
                               is a measurement, not a free win.
  EXL3_PREFIX_CACHE_INTERVAL_PP  prompt-ingestion spacing when INTERVAL is set (default = INTERVAL)

generator_kwargs() goes into Generator(...), apply(generator) after construction; both are no-ops when the
cache is disabled.
"""
from __future__ import annotations

import os

PAGE_SIZE = 256

PREFIX_CACHE = os.environ.get("EXL3_PREFIX_CACHE", "1") == "1"
PREFIX_CACHE_CAPACITY = int(os.environ.get("EXL3_PREFIX_CACHE_CAPACITY", str(4 * 1024 ** 3)))
PREFIX_CACHE_MIN_MATCH = int(os.environ.get("EXL3_PREFIX_CACHE_MIN_MATCH", "256"))
_INTERVAL_ENV = os.environ.get("EXL3_PREFIX_CACHE_INTERVAL", "").strip()
PREFIX_CACHE_INTERVAL = int(_INTERVAL_ENV) if _INTERVAL_ENV else None
PREFIX_CACHE_INTERVAL_PP = int(os.environ.get("EXL3_PREFIX_CACHE_INTERVAL_PP", "") or
                               (_INTERVAL_ENV or "0")) or None


def check():
    """Validates the knob values; raises before a load so a typo cannot cost a load."""
    if not PREFIX_CACHE or PREFIX_CACHE_CAPACITY == 0:
        return False
    if PREFIX_CACHE_INTERVAL is not None and (PREFIX_CACHE_INTERVAL <= 0 or PREFIX_CACHE_INTERVAL % PAGE_SIZE):
        raise ValueError("EXL3_PREFIX_CACHE_INTERVAL must be a positive multiple of 256")
    if PREFIX_CACHE_INTERVAL_PP is not None and (PREFIX_CACHE_INTERVAL_PP <= 0 or
                                                 PREFIX_CACHE_INTERVAL_PP % PAGE_SIZE):
        raise ValueError("EXL3_PREFIX_CACHE_INTERVAL_PP must be a positive multiple of 256")
    if PREFIX_CACHE_CAPACITY < 0 or PREFIX_CACHE_MIN_MATCH < 0:
        raise ValueError("EXL3_PREFIX_CACHE_CAPACITY and EXL3_PREFIX_CACHE_MIN_MATCH must be nonnegative")
    return True


def enabled() -> bool:
    return check()


def generator_kwargs() -> dict:
    """The Generator(...) keywords that size and schedule the pool."""
    if not enabled():
        return {"recurrent_cache_size": 0}
    kwargs = {"recurrent_cache_size": PREFIX_CACHE_CAPACITY}
    # Not stated unless asked for: None here means the Generator keeps model.caps' schedule, so this
    # module never silently re-tunes the checkpoint density of a shipped measurement
    if PREFIX_CACHE_INTERVAL is not None:
        kwargs["recurrent_checkpoint_interval"] = PREFIX_CACHE_INTERVAL
        kwargs["recurrent_checkpoint_interval_pp"] = PREFIX_CACHE_INTERVAL_PP or PREFIX_CACHE_INTERVAL
    return kwargs

def apply(generator) -> None:
    """Sets the restore gate on a constructed generator (no-op when disabled)."""
    generator.prefix_cache_min_match = PREFIX_CACHE_MIN_MATCH if enabled() else None


def summary() -> str:
    if not enabled():
        return "PREFIX_CACHE enabled=0"
    interval = PREFIX_CACHE_INTERVAL if PREFIX_CACHE_INTERVAL is not None else "model"
    return (f"PREFIX_CACHE enabled=1 capacity_mib={PREFIX_CACHE_CAPACITY // 2**20} "
            f"min_match={PREFIX_CACHE_MIN_MATCH} interval={interval}")


def log(message: str) -> None:
    print(message, flush=True)
