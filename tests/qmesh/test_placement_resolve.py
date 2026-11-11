"""qmesh/placement.py: a placement named by a pack's manifest is mandatory, not optional.

2026-11-05 04:40, arm N on pack mix4460-nq: PACK_MANIFEST.json named
/network/pack/placement.json, which the dev container does
not mount. resolve() returned the repo default (qmesh/placement.final.json, sha 21e981f6, the 4460t
table) and the client asked the hot half for `layers.0.ffn.experts.21.w3`, an expert that pack keeps
cold: `ValueError: No tensors found ... matching supported quantization format` after the arm had
already loaded. These tests pin the load-time refusal that replaces that silent fallback; they need no
device.
"""
import hashlib
import importlib.util
import json
import os

import pytest

pytestmark = pytest.mark.nogpu

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLACEMENT_PY = os.path.join(REPO_DIR, "qmesh", "placement.py")

TABLE = {"schema": "qmesh.exl3-expert-placement/v1",
         "layers": {"0": {"hot": [0, 2], "warm": [1]}}}


@pytest.fixture(autouse = True)
def _no_explicit_placement(monkeypatch):
    # resolve() reads $PLACEMENT and $CHECKPOINT from the environment: the tests set their own.
    monkeypatch.delenv("PLACEMENT", raising = False)
    monkeypatch.delenv("CHECKPOINT", raising = False)


@pytest.fixture
def placement():
    spec = importlib.util.spec_from_file_location("_placement_under_test", PLACEMENT_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pack(tmp_path, name = "placement.exl3.json", body = True, sha = "match", table = TABLE):
    """A hot pack dir: manifest naming `name` (or nothing when name is None), and the file when `body`."""
    pack = tmp_path / "exl3-p7hot-test"
    pack.mkdir()
    if name is not None:
        data = json.dumps(table).encode()
        if body is True or body == "file":
            (pack / name).write_bytes(data)
        elif body == "dir":
            (pack / name).mkdir()  # exists, but no open() for reading succeeds on it
    man = {"mode": "hot", "keys": 0}
    if name is not None:
        man["placement"] = name
        if sha == "match":
            written = body is True or body == "file"
            man["placement_sha256"] = _digest((pack / name).read_bytes()) if written else "0" * 64
        elif sha is not None:
            man["placement_sha256"] = sha
    (pack / "PACK_MANIFEST.json").write_text(json.dumps(man))
    return pack


def test_relative_placement_inside_the_pack_resolves(placement, tmp_path):
    pack = _pack(tmp_path)
    path = placement.resolve(str(pack))
    assert path == os.path.join(str(pack), "placement.exl3.json")
    table, loaded_path, digest = placement.load(str(pack))
    assert table == TABLE
    assert loaded_path == path
    assert digest == _digest(json.dumps(TABLE).encode())
    assert placement.current_sha() == digest


def test_absolute_placement_resolves(placement, tmp_path):
    elsewhere = tmp_path / "packs"
    elsewhere.mkdir()
    (elsewhere / "placement.abs.json").write_text(json.dumps(TABLE))
    pack = _pack(tmp_path, name = str(elsewhere / "placement.abs.json"))
    assert placement.resolve(str(pack)) == str(elsewhere / "placement.abs.json")


def test_missing_placement_raises_instead_of_falling_back(placement, tmp_path, monkeypatch):
    pack = _pack(tmp_path, body = False)
    monkeypatch.setenv("CHECKPOINT", str(pack))
    with pytest.raises(placement.PlacementError) as e:
        placement.resolve()
    msg = str(e.value)
    assert "PACK_MANIFEST.json" in msg
    assert "placement.exl3.json" in msg
    assert str(pack) in msg                       # the checkpoint that names it
    assert placement.DEFAULT in msg               # the table it refuses to substitute
    assert "refusing to fall back" in msg


def test_unreadable_placement_raises(placement, tmp_path):
    pack = _pack(tmp_path, body = "dir")
    with pytest.raises(placement.PlacementError) as e:
        placement.resolve(str(pack))
    assert "opens" in str(e.value)


def test_sha_mismatch_raises(placement, tmp_path):
    pack = _pack(tmp_path, sha = "0" * 64)
    with pytest.raises(placement.PlacementError) as e:
        placement.resolve(str(pack))
    msg = str(e.value)
    assert "0" * 64 in msg                        # what the manifest records
    assert _digest(json.dumps(TABLE).encode()) in msg   # what the file is
    assert "refusing" in msg


def test_pack_naming_no_placement_still_gets_the_default(placement, tmp_path):
    pack = _pack(tmp_path, name = None)
    assert placement.resolve(str(pack)) == placement.DEFAULT
    # A pack dir without any manifest is the same case (older hot dirs, plain checkpoints).
    empty = tmp_path / "no-manifest"
    empty.mkdir()
    assert placement.resolve(str(empty)) == placement.DEFAULT


def test_manifest_with_only_the_sha_pins_the_default(placement, tmp_path):
    pack = _pack(tmp_path, name = None)
    man = json.loads((pack / "PACK_MANIFEST.json").read_text())
    man["placement_sha256"] = _digest(open(placement.DEFAULT, "rb").read())
    (pack / "PACK_MANIFEST.json").write_text(json.dumps(man))
    assert placement.resolve(str(pack)) == placement.DEFAULT
    man["placement_sha256"] = "0" * 64
    (pack / "PACK_MANIFEST.json").write_text(json.dumps(man))
    with pytest.raises(placement.PlacementError):
        placement.resolve(str(pack))


def test_explicit_placement_wins_unchecked(placement, tmp_path):
    pack = _pack(tmp_path, body = False)          # the manifest would raise
    os.environ["PLACEMENT"] = "/nonexistent/operator/placement.json"
    try:
        assert placement.resolve(str(pack)) == "/nonexistent/operator/placement.json"
    finally:
        del os.environ["PLACEMENT"]
