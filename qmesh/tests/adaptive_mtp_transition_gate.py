"""CPU regression for MTP draft-cache settlement on target-only transitions.

This executes the actual MTP settlement block from Generator.iterate_gen, using a
position-tracking fake cache. It is not a numerical model or GPU correctness test.
Run: python qmesh/tests/adaptive_mtp_transition_gate.py
"""
import ast
import pathlib
import unittest

source = pathlib.Path(__file__).resolve().parents[2] / "exllamav3/generator/generator.py"
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Generator")
iterate_gen = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "iterate_gen")
settlement = next(n for n in iterate_gen.body if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Attribute) and n.test.attr == "mtp_draft")
code = compile(ast.Module(body=[settlement], type_ignores=[]), str(source), "exec")


class Tensor:
    def __getitem__(self, key):
        return self

    def __add__(self, value):
        return self

    def clone(self):
        return self


class Job:
    def __init__(self, hidden=True):
        self.mtp_last_hidden = Tensor() if hidden else None


class Draft:
    def __init__(self):
        self.positions = set()

    def prefill(self, ids, params):
        self.positions.add(params["cache_seqlens"].position)


class Position(Tensor):
    def __init__(self, position):
        self.position = position

    def __getitem__(self, key):
        return self

    def __add__(self, value):
        return Position(self.position + value)


def settle(jobs, drafted, accepted_length=1):
    """Simulate one selected target round; execute the production settlement block.

    A real MTP draft pass consumes position K for every selected job. With no
    draft pass, only the settlement block can populate that position. Cache
    tracking focuses on the already-primed first row, not bootstrap semantics.
    """
    draft = Draft()
    position = 7
    if drafted:
        draft.positions.add(position)
    owner = type("Owner", (), {"mtp_draft": True, "draft_model": draft,
                               "draft_cache": object(), "active_jobs": jobs})()
    env = {"self": owner, "p_export_states": [Tensor()],
           "logit_mapping": list(range(len(jobs) + 1)),
           "accepted_lengths": [accepted_length] * len(jobs), "rewound_jobs": set(),
           "batch_ids": Tensor(), "block_index": Tensor(),
           "p_cache_seqlens": Position(position)}
    exec(code, env)
    return draft.positions, position


class TransitionGate(unittest.TestCase):
    def test_drafted_baseline(self):
        positions, k = settle([Job()], drafted=True)
        self.assertIn(k, positions)

    def test_window_positive_to_zero_to_positive(self):
        positions, k = settle([Job()], drafted=False)
        self.assertIn(k, positions, "K=0 target-only round left the existing MTP cache row stale")

    def test_unprimed_peer_forces_target_only_for_primed_row(self):
        positions, k = settle([Job(), Job(hidden=False)], drafted=False)
        self.assertIn(k, positions, "unprimed peer forced a target-only round without advancing a primed row")


if __name__ == "__main__":
    unittest.main()
