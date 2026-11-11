"""CPU check that a trace width override cannot alter graphed serving windows."""
import ast
import pathlib
import types

source = pathlib.Path(__file__).resolve().parents[2] / 'exllamav3/architecture/deepseek_v4_mtp.py'
tree = ast.parse(source.read_text())
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_trace_window')
ns = {}
exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), 'exec'), ns)
width = ns['_trace_window']

assert width({'DSPARK_ROWTRACE_WIDTH': '4', 'EXL3_WHOLE_GRAPH': '0'}) == 4
assert width({'DSPARK_ROWTRACE_WIDTH': '6', 'EXL3_WHOLE_GRAPH': '0'}) == 6
assert width({'DSPARK_ROWTRACE_WIDTH': '2', 'EXL3_WHOLE_GRAPH': '0'}) == 2
assert width({'DSPARK_ROWTRACE_WIDTH': '4', 'EXL3_WHOLE_GRAPH': '1'}) is None
assert width({'DSPARK_ROWTRACE_WIDTH': '1', 'EXL3_WHOLE_GRAPH': '0'}) is None
assert width({'DSPARK_ROWTRACE_WIDTH': '0', 'EXL3_WHOLE_GRAPH': '0'}) is None
assert width({}) is None

model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DeepseekV4MTPModel')
sample = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_from_state')
assignment = next(n for n in ast.walk(sample) if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Subscript)
                  and ast.unparse(n.targets[0]) == "params['draft_confidence_len']")


def chosen(window, adaptive):
    return eval(compile(ast.Expression(assignment.value), str(source), 'eval'),
                {'self': types.SimpleNamespace(diag_draft_window=window), 'int': int,
                 'lens': types.SimpleNamespace(max=lambda: types.SimpleNamespace(item=lambda: adaptive))})


assert chosen(4, 5) == 3, 'trace width 4 must verify three draft tokens'
assert chosen(6, 3) == 5, 'trace width 6 must verify five draft tokens'
assert chosen(None, 5) == 5, 'serving must keep the adaptive window'
print('PASS: fixed verifier width only under eager tracing; graphed serving window untouched')

# Execute the real generator method without importing torch: the budget is global, follows
# ready jobs, and includes bucket padding rather than imposing a limit on individual users.
generator_source = source.parents[1] / 'generator/generator.py'
generator_tree = ast.parse(generator_source.read_text())
generator_class = next(n for n in generator_tree.body if isinstance(n, ast.ClassDef) and n.name == 'Generator')
method = next(n for n in generator_class.body if isinstance(n, ast.FunctionDef) and n.name == 'row_limited')
env = {'pad_bucket': types.SimpleNamespace(bucket_rows=lambda n: 8 if 2 <= n <= 8 else n),
       'logger': types.SimpleNamespace(debug=lambda *args: None)}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(generator_source), 'exec'), env)
job = lambda ready: types.SimpleNamespace(is_prefill_done=lambda: ready, sequences=[None])
state = types.SimpleNamespace(draft_max_rows=48, active_jobs=[job(True)] * 8,
                              cache=types.SimpleNamespace(pad_slot=None), num_draft_tokens=5,
                              _draft_window_last=None)
limited = env['row_limited']
assert limited(state, 5) == 5
state.active_jobs = [job(True)] * 9
assert limited(state, 5) == 4
state.active_jobs = [job(True)] * 3 + [job(False)]
state.cache.pad_slot = object()
state.draft_max_rows = 24
assert limited(state, 5) == 2, 'padded execution must count all eight rows'
state.draft_max_rows = 0
assert limited(state, 5) == 5
print('PASS: adaptive verifier row budget tracks active and padded rows')
