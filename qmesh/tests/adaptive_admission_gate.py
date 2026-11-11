"""Exercise actual Engine shared-pool admission with fake tokenizer and generator.

CPU-only control-flow gate, not evidence of GPU safety or real long-context inference.
"""
import ast
import pathlib
import queue
import threading
import time
import types

source = pathlib.Path(__file__).resolve().parents[1] / 'serve_pair.py'
tree = ast.parse(source.read_text())
engine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Engine')

class Ids:
    def __init__(self, n):
        self.n = n
        self.shape = (1, n)
    def numel(self):
        return self.n
    def __getitem__(self, key):
        return self
    def tolist(self):
        return [1]  # scheduler does not inspect token content

class Tokenizer:
    eos_token_id = 0
    def encode(self, prompt, **kwargs):
        return Ids(int(prompt))

class Job:
    serial = 0
    def __init__(self, **kwargs):
        Job.serial += 1
        self.serial_number = Job.serial

class Generator:
    max_total_tokens = 5000192
    max_draft_tokens = 5
    def __init__(self):
        self.jobs = []
        self.started = []
        self.finish = threading.Event()
    def enqueue(self, job):
        self.jobs.append(job)
        self.started.append(job.serial_number)
    def iterate(self):
        time.sleep(.001)
        if self.finish.is_set() and self.jobs:
            job = self.jobs.pop(0)
            self.finish.clear()
            return [{'serial': job.serial_number, 'eos': True}]
        return []

ns = dict(queue=queue, threading=threading, time=time, os=types.SimpleNamespace(environ={}),
          BATCH=8, MAX_PENDING=8, TOKEN_MARGIN_DIAG=False,
          torch=types.SimpleNamespace(tensor=lambda x, **kwargs: x, long=int),
          Job=Job, ComboSampler=lambda **kwargs: None)
exec(compile(ast.Module(body=[engine], type_ignores=[]), str(source), 'exec'), ns)
gen = Generator()
e = ns['Engine'](gen, Tokenizer())
def req(prompt, completion=1):
    return dict(prompt=str(prompt), max_tokens=completion, temperature=0, top_p=1,
                presence=0, frequency=0, seed=42, stop=[], tools=False)

at_limit = req(gen.max_total_tokens - 1 - 1 - gen.max_draft_tokens)
e.validate_request(at_limit)
assert at_limit['cache_reservation'] == gen.max_total_tokens
try:
    e.validate_request(req(gen.max_total_tokens - 1 - gen.max_draft_tokens))
except ValueError:
    pass
else:
    raise AssertionError('request exceeding shared pool accepted')
above_old_cap = req(1048576)
e.validate_request(above_old_cap)
assert above_old_cap['cache_reservation'] > 1048576

requests = [req(1048576 - 1 - 1 - gen.max_draft_tokens) for _ in range(5)]
for r in requests:
    e.validate_request(r)
    assert e.submit(r)[1] is None
thread = threading.Thread(target=e.run, daemon=True)
thread.start()
def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('timed out waiting for scheduler')
        time.sleep(.002)
wait_for(lambda: len(gen.started) == 4)
time.sleep(.05)
assert len(gen.started) == 4, 'fifth reservation exceeded 5M pool'
gen.finish.set()
wait_for(lambda: len(gen.started) == 5)
for remaining in range(4, 0, -1):
    gen.finish.set()
    wait_for(lambda: len(gen.jobs) == remaining - 1)
e.stop()
thread.join(5)
assert not thread.is_alive() and e.failure is None

# Trace-only coalescing admits peers together even when the second arrives later.
ns['BATCH'] = 2
ns['os'].environ['DSPARK_ROWTRACE_OUT'] = '/diagnostic'
gen = Generator()
e = ns['Engine'](gen, Tokenizer())
first, second = req(10), req(11)
for r in (first, second):
    e.validate_request(r)
assert e.submit(first)[1] is None
thread = threading.Thread(target=e.run, daemon=True)
thread.start()
time.sleep(.05)
assert not gen.started, 'trace barrier did not wait for peer'
assert e.submit(second)[1] is None
wait_for(lambda: len(gen.started) == 2)
for remaining in (1, 0):
    gen.finish.set()
    wait_for(lambda: len(gen.jobs) == remaining)
e.stop()
thread.join(5)
assert not thread.is_alive() and e.failure is None
print('PASS: diagnostic admission coalesces staggered peers')
print('PASS: shared pool boundary, requests above 1M, reservation wait and progress')
