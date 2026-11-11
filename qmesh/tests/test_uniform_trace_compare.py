"""CPU tests for the fail-closed matched-logit reporter."""
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

SCRIPT = Path(__file__).with_name('uniform_trace_compare.py')


def test_aligned_logits_and_digest_gate():
    metadata = {'batch': 2, 'width': 1, 'target_position': 1, 'prefix': [3],
                'state_digest': 'same', 'modules': [], 'graph_scope': 'eager-only',
                'input_ids': torch.tensor([[4], [4]]), 'positions': [1, 1]}
    logits = torch.arange(12, dtype=torch.float32).reshape(2, 1, 6)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for tag in ('R1', 'C1', 'C2', 'R2'):
            (root / tag).mkdir()
            torch.save({'metadata': metadata, 'records': [{'module': '__model_output__',
                        'outputs': [{'path': 'output', 'status': 'tensor_batch_axis_unverified',
                                     'tensor': logits}]}]}, root / tag / 'trace.pt')
        result = subprocess.run([sys.executable, str(SCRIPT), str(root)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert 'reference_floor row=1 pos=0 max_abs=0' in result.stdout
        data = torch.load(root / 'C1' / 'trace.pt', weights_only=False)
        data['metadata']['state_digest'] = 'different'
        torch.save(data, root / 'C1' / 'trace.pt')
        result = subprocess.run([sys.executable, str(SCRIPT), str(root)], capture_output=True, text=True)
        assert result.returncode != 0
        assert 'INVALID alignment: state_digest' in result.stderr
