"""CPU checks of fail-closed rowtrace digest and comparator alignment."""
import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('rowtrace_dspark', ROOT / 'qmesh/rowtrace_dspark.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DigestGate(unittest.TestCase):
    def test_candidate_accepts_verifier_without_rope_offsets(self):
        from types import SimpleNamespace as S
        jobs = [S(sequences=[object()]) for _ in range(2)]
        ids, positions = torch.zeros((2, 6)), torch.zeros(2)
        params = {'recurrent_history': True, 'positions': None}
        self.assertTrue(module._candidate(params, ids, positions, jobs, 2, 6, False))
        self.assertTrue(module._candidate({**params, 'positions': positions}, ids, positions, jobs, 2, 6, False))
        self.assertFalse(module._candidate({**params, 'recurrent_history': False}, ids, positions, jobs, 2, 6, False))
        self.assertFalse(module._candidate(params, ids, positions, jobs[:1], 2, 6, False))
        self.assertFalse(module._candidate({**params, 'prefill': True}, ids, positions, jobs, 2, 6, False))
        self.assertFalse(module._candidate(params, ids, positions, jobs, 2, 6, True))
        self.assertTrue(module._candidate({}, torch.zeros((1, 1)), torch.zeros(1), jobs[:1], 1, 1, False))
        jobs8 = [S(sequences=[object()]) for _ in range(8)]
        self.assertTrue(module._candidate(params, torch.zeros((8, 6)), torch.zeros(8), jobs8, 8, 6, False))
        self.assertFalse(module._candidate(params, torch.zeros((8, 6)), torch.zeros(8), jobs8[:7], 8, 6, False))

    def test_digest_tracks_bytes_shape_and_metadata(self):
        a = {'position': 4, 'x': [torch.tensor([1, 2], dtype=torch.int32)]}
        self.assertEqual(module._digest(a), module._digest({'x': [a['x'][0].clone()], 'position': 4}))
        self.assertNotEqual(module._digest(a), module._digest({'position': 4, 'x': [torch.tensor([1, 3], dtype=torch.int32)]}))
        self.assertNotEqual(module._digest(a), module._digest({'position': 5, 'x': a['x']}))
        self.assertNotEqual(module._digest(a), module._digest({'position': 4, 'x': [a['x'][0].view(1, 2)]}))

    def test_snapshot_fingerprints_localize_differences(self):
        a = {'paged': {'layer0': [torch.tensor([1])], 'layer1': [torch.tensor([2])]},
             'recurrent': {'position': 3, 'tensor': torch.tensor([4])}}
        b = {'paged': {'layer0': [torch.tensor([1])], 'layer1': [torch.tensor([5])]},
             'recurrent': {'position': 3, 'tensor': torch.tensor([4])}}
        x, y = module._snapshot_digests(a), module._snapshot_digests(b)
        self.assertEqual(x['recurrent'], y['recurrent'])
        self.assertEqual(x['paged']['layer0'], y['paged']['layer0'])
        self.assertNotEqual(x['paged']['layer1'], y['paged']['layer1'])

    def test_unknown_snapshot_type_refused(self):
        with self.assertRaises(TypeError):
            module._digest(object())

    def test_comparator_rejects_unequal_cache_state(self):
        meta = {'batch': 1, 'width': 1, 'target_position': 1, 'prefix': [3],
                'state_digest': 'a', 'modules': [], 'graph_scope': 'eager-only',
                'input_ids': torch.tensor([[4]]), 'positions': [1]}
        other = {**meta, 'batch': 2, 'state_digest': 'b',
                 'input_ids': torch.tensor([[4], [4]]), 'positions': [1, 1]}
        with tempfile.TemporaryDirectory() as d:
            a, b = pathlib.Path(d) / 'a.pt', pathlib.Path(d) / 'b.pt'
            torch.save({'metadata': meta, 'records': []}, a)
            torch.save({'metadata': other, 'records': []}, b)
            p = subprocess.run([sys.executable, str(ROOT / 'qmesh/tests/compare_dspark_rowtrace.py'), str(a), str(b)],
                               capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('state_digest mismatch', p.stderr + p.stdout)

    def test_comparator_attributes_first_difference_to_module_inputs(self):
        t = lambda *rows: torch.stack([torch.tensor(r) for r in rows])
        meta = {'batch': 1, 'width': 2, 'target_position': 1, 'prefix': [3], 'state_digest': 'a',
                'modules': [('0', 0, 'M')], 'graph_scope': 'eager-only',
                'input_ids': t([4, 5]), 'positions': [1]}
        trace = lambda batch: {'metadata': {**meta, 'batch': batch, 'input_ids': t(*[[4, 5]] * batch),
                                            'positions': [1] * batch},
                               'records': [{'order': 0, 'module': '0', 'class': 'M',
                                            'inputs': [{'path': 'input', 'status': 'tensor_batch_axis_unverified',
                                                        'tensor': t(*[[1.0, 2.0]] * batch)}],
                                            'outputs': [{'path': 'output', 'status': 'tensor_batch_axis_unverified',
                                                         'tensor': t(*[[3.0, 4.0]] * batch)}]}]}
        with tempfile.TemporaryDirectory() as d:
            def run(b):
                p = pathlib.Path(d) / f'{b}.pt'
                torch.save(trace(b), p)
                return p
            same = subprocess.run([sys.executable, str(ROOT / 'qmesh/tests/compare_dspark_rowtrace.py'),
                                   str(run(1)), str(run(2))], capture_output=True, text=True)
            self.assertEqual(same.returncode, 0, same.stdout + same.stderr)
            self.assertIn('INPUT EXACT 0 0', same.stdout)
            moved = trace(2)
            moved['records'][0]['outputs'][0]['tensor'][0] += 1
            p = pathlib.Path(d) / 'moved.pt'
            torch.save(moved, p)
            diff = subprocess.run([sys.executable, str(ROOT / 'qmesh/tests/compare_dspark_rowtrace.py'),
                                   str(pathlib.Path(d) / '1.pt'), str(p)], capture_output=True, text=True)
            self.assertEqual(diff.returncode, 1, diff.stdout + diff.stderr)
            self.assertIn('first differing output: 0', diff.stdout)

            # The batched arm's second row must not decide the verdict: row 0 equal is EXACT.
            other = trace(2)
            other['records'][0]['outputs'][0]['tensor'][1] += 1
            p2 = pathlib.Path(d) / 'other.pt'
            torch.save(other, p2)
            rows = subprocess.run([sys.executable, str(ROOT / 'qmesh/tests/compare_dspark_rowtrace.py'),
                                   str(pathlib.Path(d) / '1.pt'), str(p2)], capture_output=True, text=True)
            self.assertEqual(rows.returncode, 0, rows.stdout + rows.stderr)
            self.assertIn('EXACT 0 0', rows.stdout)
    def test_write_pages_and_recurrent_slots_are_distinct(self):
        from types import SimpleNamespace as S
        target_cache, draft_cache = object(), object()
        def job(page, slot):
            state = S(target=S(cache=target_cache, slot=slot), draft=S(cache=draft_cache, slot=slot))
            return S(recurrent_state=state, sequences=[S(kv_position=0,
                     allocated_pages=[S(page_index=page, ref_count=1)])])
        a, b = job(0, 0), job(1, 1)
        module._check_row_ownership([a, b], 256, 1)
        b.recurrent_state.target.slot = 0
        with self.assertRaisesRegex(RuntimeError, 'SHARED_SLOT'):
            module._check_row_ownership([a, b], 256, 1)
        b.recurrent_state.target.slot = 1
        b.sequences[0].allocated_pages[0].page_index = 0
        with self.assertRaisesRegex(RuntimeError, 'SHARED_WRITE_PAGE'):
            module._check_row_ownership([a, b], 256, 1)
        b.sequences[0].allocated_pages[0].page_index = 1
        a.sequences[0].kv_position = 255
        a.sequences[0].allocated_pages.append(S(page_index=1, ref_count=1))
        with self.assertRaisesRegex(RuntimeError, 'SHARED_WRITE_PAGE'):
            module._check_row_ownership([a, b], 256, 2)
        a.sequences[0].allocated_pages.pop()
        with self.assertRaisesRegex(RuntimeError, 'MISSING_WRITE_PAGE'):
            module._check_row_ownership([a, b], 256, 2)



if __name__ == '__main__':
    unittest.main()
