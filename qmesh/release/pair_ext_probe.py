"""Build the exllamav3 JIT extension from a release's sources into that release's cache.

Run inside the pair container with the script on stdin (see pair_build_ext.sh):

    TORCH_EXTENSIONS_DIR=/pair/releases/<sha>/ext EXPECT_SRC=/pair/releases/<sha>/src/exllamav3 \
      python3 -u - < pair_ext_probe.py

Two things are checked, because both have silently produced a wrong library before:
  * the image ships a prebuilt exllamav3_ext.so; it is hidden with a find_spec shim so exllamav3.ext
    takes the JIT path (otherwise the run "builds" nothing and imports the image's library), and
  * e.sources_dir really is this release's tree (without an explicit EXPECT_SRC the import can fall
    back to site-packages and compile a stale tree into the cache).
"""
import importlib.util as u
import os
import sys

EXPECT = os.path.realpath(os.environ["EXPECT_SRC"])
CACHE = os.environ["TORCH_EXTENSIONS_DIR"]
if not CACHE:
    sys.exit("pair_ext_probe: TORCH_EXTENSIONS_DIR unset")
if not os.path.isdir(EXPECT):
    sys.exit(f"pair_ext_probe: EXPECT_SRC {EXPECT} does not exist")

_orig = u.find_spec


def _find_spec(name, *a, **k):
    if name == "exllamav3_ext":
        return None
    return _orig(name, *a, **k)


u.find_spec = _find_spec

import exllamav3.ext as e  # noqa: E402

src = os.path.realpath(e.sources_dir)
if not src.startswith(EXPECT + os.sep):
    sys.exit(f"pair_ext_probe: built from {src}, expected under {EXPECT}/")

mod = getattr(e, "exllamav3_ext", None)
so = getattr(mod, "__file__", "") or ""
print("SRC_DIR", src, flush=True)
print("EXT_CACHE", os.path.realpath(CACHE), flush=True)
print("EXT_SO", so, flush=True)
print("EXT_BUILD_OK", os.path.basename(so), flush=True)
if not so or not os.path.exists(so):
    sys.exit("pair_ext_probe: extension module has no file on disk")
