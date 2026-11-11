"""CPU control-flow gate for DSpark cache settlement across changing draft widths.

Executes Generator.iterate_gen's DFlash settlement with a position-tracking
fake drafter; it does not validate GPU kernels or numerical token identity.
Run: python qmesh/tests/adaptive_dspark_transition_gate.py
"""
import ast
import pathlib
import unittest

source = pathlib.Path(__file__).resolve().parents[2] / "exllamav3/generator/generator.py"
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Generator")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "iterate_gen")
block = next(n for n in method.body if isinstance(n, ast.If)
             and isinstance(n.test, ast.Attribute) and n.test.attr == "dflash_draft")
code = compile(ast.Module(body=[block], type_ignores=[]), str(source), "exec")


class State:
    def __init__(self):
        self.position = 0


class Job:
    def __init__(self):
        self.recurrent_state = type("Pair", (), {"draft": State()})()


class Drafter:
    def update_kv_from_target(self, *, target_hidden, cache, lengths, params):
        for state, start, length in zip(params["recurrent_states"], params["cache_seqlens"], lengths):
            if length:
                assert state.position == start
                state.position += length


def settle(jobs, accepted, rewound=()):
    starts = [j.recurrent_state.draft.position for j in jobs]
    owner = type("Generator", (), {"dflash_draft": True, "draft_recurrent": True,
                                   "draft_model": Drafter(), "draft_cache": object(),
                                   "active_jobs": jobs})()
    exec(code, {"self": owner, "block_index": object(), "p_cache_seqlens": starts,
                "accepted_lengths": accepted, "logit_mapping": list(range(len(jobs) + 1)),
                "rewound_jobs": {id(j) for j in rewound}, "p_export_states": object()})


class DSparkTransitions(unittest.TestCase):
    def test_draft_to_target_only_and_back_with_parked_job(self):
        a, b = Job(), Job()
        settle([a, b], [3, 2])       # speculative round, different accepted lengths
        settle([a], [1])             # b parked; target-only round for a
        self.assertEqual((a.recurrent_state.draft.position, b.recurrent_state.draft.position), (4, 2))
        settle([a, b], [2, 1])       # speculation resumes for both
        self.assertEqual((a.recurrent_state.draft.position, b.recurrent_state.draft.position), (6, 3))

    def test_rewound_row_is_not_updated(self):
        a, b = Job(), Job()
        settle([a, b], [2, 3], rewound=[a])
        self.assertEqual((a.recurrent_state.draft.position, b.recurrent_state.draft.position), (0, 3))


if __name__ == "__main__":
    unittest.main()
