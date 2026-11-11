#!/usr/bin/env python3
import importlib.util, os, re, sys, io, contextlib
os.environ["PAIR_QUEUE_ROOT"] = "/tmp/sweepq"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PAIR_CONF"] = os.path.join(HERE, "tests", "pairs.conf")
os.environ["PAIR_COLD_PACK"] = "/home/test/cold-pack"
os.environ["PAIR_ART_MOUNT"] = "/home/test/artifacts"
spec = importlib.util.spec_from_file_location("pc", os.path.join(HERE, "pair_containers.py"))
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)
_REAL_PLAN = pc.plan_side   # the apply test below stubs the module attribute; the arrow tests need the real one

fails = []
def check(name, cond, detail=""):
    print(("OK   " if cond else "FAIL ") + name + (" | " + detail if detail else ""))
    if not cond: fails.append(name)

def die_msg(fn, *a):
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            fn(*a)
        return ""
    except SystemExit:
        return err.getvalue().strip()

for _key, _token in (("PAIR_COLD_PACK", "__PAIR_COLD_PACK__"),
                     ("PAIR_ART_MOUNT", "__PAIR_ART_MOUNT__")):
    _value = os.environ.pop(_key)
    _m = die_msg(pc.resolve_mounts, {"mounts": [[_token, "/target"]]})
    check(f"{_key} is required", _key in _m and "absolute" in _m, _m)
    os.environ[_key] = "relative/path"
    _m = die_msg(pc.resolve_mounts, {"mounts": [[_token, "/target"]]})
    check(f"{_key} must be absolute", _key in _m and "absolute" in _m, _m)
    os.environ[_key] = _value
    _resolved = pc.resolve_mounts({"mounts": [[_token, "/target"]]})
    check(f"{_key} supplies the mount source", _resolved["mounts"] == [[_value, "/target"]], str(_resolved))

rows = pc._conf_rows()
names = [n for _, p, g in rows for n in (p, g)]
check("container names unique across pairs", len(names) == len(set(names)), str(rows))
d = dict((r[0], (r[1], r[2])) for r in rows)
check("dev owns seam-p7-dev / seam-gx2-dev", d["dev"] == ("seam-p7-dev", "seam-gx2-dev"), str(d["dev"]))
check("prod owns seam-p7-prod / seam-gx1-prod", d["prod"] == ("seam-p7-prod", "seam-gx1-prod"), str(d["prod"]))
check("the containers arms actually use belong to no pair",
      not any(n in names for n in ("seam-p7v2", "seam-gx1v2", "seam-gx1b", "seam-p7v2-robust")), str(names))

cfg = pc.load_conf("dev"); cfg["p7ct"] = "seam-p7-prod"
m = die_msg(pc.check_ownership, cfg)
check("foreign pair's container refused", "belongs to pair prod" in m, m)

import subprocess
os.makedirs("/tmp/sweepq/dev/gpu_queue", exist_ok=True)
h = subprocess.Popen(["bash", "-c", "sleep 60", "seam-p7-dev"])
t = f"/tmp/sweepq/dev/gpu_queue/01-1-{h.pid}-SomeAgent"; open(t, "w").write("x")
m = die_msg(pc.check_ownership, pc.load_conf("dev"))
check("container named by a live ticket refused", "named by a live ticket" in m, m)
h.kill(); os.remove(t)

# apply: existing container must not be recreated; absent ones are created under their final name
calls = []; state = {}
def fake_docker(cfg, side, args, stdin=None, **kw):
    calls.append((side, " ".join(args)))
    if args[0] == "run":
        state[("p7" if "seam-p7-dev" in " ".join(args) else "gx",
               "seam-p7-dev" if "seam-p7-dev" in " ".join(args) else "seam-gx2-dev")] = True
    if args[0] == "inspect":
        return (0, "true\n", "") if (side, args[-1]) in state else (1, "", "no such container")
    return (0, "", "")
pc.docker = fake_docker
pc.plan_side = lambda cfg, side: None if (side, "seam-p7-dev" if side == "p7" else "seam-gx2-dev") in state else ["absent"]
links = []; inners = []
pc.cmd_link = lambda cfg, sha: links.append("link")
pc.cmd_inner = lambda cfg, sha=None, verify=True: inners.append(verify)

root = "/tmp/sweeppair"; os.makedirs(f"{root}/releases/abc123/src", exist_ok=True)
cfg = pc.load_conf("dev"); cfg["root"] = root
state[("p7", "seam-p7-dev")] = True; state[("gx", "seam-gx2-dev")] = True
_old_plan = pc.plan_side
pc.plan_side = lambda cfg, side: ["spec-differs"]
m = die_msg(pc.cmd_apply, cfg, "abc123", "T")
check("existing container not recreated", "refusing to recreate an existing" in m, m)
check("no rename/stop/rm before the refusal", not any(a.split()[0] in ("rename", "stop", "rm") for _, a in calls))

pc.plan_side = _old_plan
calls.clear(); state.clear(); links.clear(); inners.clear()
msg = io.StringIO()
with contextlib.redirect_stdout(msg):
    pc.cmd_apply(cfg, "abc123", "T")
out = msg.getvalue()
runs = [a for _, a in calls if a.startswith("run ")]
check("creates seam-p7-dev and seam-gx2-dev", any("seam-p7-dev" in a for a in runs) and any("seam-gx2-dev" in a for a in runs), str(runs)[:200])
check("never renames an existing container", not any(a.startswith("rename") for _, a in calls))
# The owner's rule: a release reaches a pair's `current` only through a passed promote. prepare may
# arm the containers (they are idle, no current, nothing that resolves through /pair/current can
# import), but creating prod/current at 00:00 on 2026-11-02 before dev had passed was a promote.
check("prepare never creates current", "LINK_UNPROMOTED" in out and "LINK_BOOTSTRAP" not in out,
      out.strip()[:300])
check("prepare promotes nothing", links == [] and inners == [False], f"links={links} inners={inners}")
check("no PAIR_ALLOW_RECREATE needed for a fresh pair", True)

# inner without current
cfg2 = dict(cfg); cfg2["root"] = "/tmp/sweeppair2"; os.makedirs("/tmp/sweeppair2/releases/abc123/src", exist_ok=True)
m = die_msg(pc.cmd_inner_real if hasattr(pc, "cmd_inner_real") else None, None) if False else None
spec2 = importlib.util.spec_from_file_location("pc2", os.path.join(HERE, "pair_containers.py"))
pc2 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(pc2)
m = die_msg(pc2.cmd_inner, cfg2, "abc123")
check("inner refuses a release root without current", "current is missing" in m, m)
execed = []
def fake_docker2(cfg, side, args, stdin=None, **kw):
    if args[0] == "inspect":
        return (0, "true\n", "")
    if args[0] == "exec":
        execed.append(args[-1])
    return (0, "", "")
pc2.docker = fake_docker2
msg = io.StringIO()
with contextlib.redirect_stdout(msg):
    pc2.cmd_inner(cfg2, "abc123", verify=False)
check("an unpromoted container is wired straight at the release",
      any("ln -sfn /pair/releases/abc123/src /src/exllamav3" in c for c in execed)
      and "target=/pair/releases/abc123/src" in msg.getvalue(), msg.getvalue().strip()[:200])

# The ext cache is only reusable by a client if the builder spells the include path the way the
# client does; these greps keep the two halves of that invariant from drifting apart again.
def src(name):
    return open(os.path.join(HERE, name)).read()

be = src("pair_build_ext.sh")
check("ext built through the client path (/src/exllamav3)",
      "ln -sfn $rel_c/src $link" in be and "PYTHONPATH=$link" in be, "")
# The container-side trap cannot be the guarantee: it does not run when the build is killed (2026-11-02,
# dev GX left wired at an unflipped release). The parent puts the link back after the exec returns,
# and prepare re-asserts /pair/current/src on every exit, so a dead build cannot drift a container.
check("ext build restores the link afterwards, from the parent",
      "trap 'ln -sfn $back $link' EXIT" in be and "EXT_LINK_RESTORE" in be
      and be.index("EXT_LINK_RESTORE") > be.index('bash -lc "$inner"'), "")
check("prepare re-asserts /pair/current/src however it exits",
      "trap relink_current EXIT" in src("pair_prepare.sh"), "")
check("client import is proven, not assumed",
      "EXT_CLIENT_HIT" in be and "EXT_CLIENT_MISS" in be and "EXT_CLIENT_TIMEOUT" in be, "")
check("ext build runs the probe as a file, never on stdin",
      "python3 -u /tmp/pair_ext_probe.py" in be and "python3 -u -" not in be, "")

check("client check uses the release's own cache, from a neutral cwd",
      "cd /tmp && PYTHONPATH=$link TORCH_EXTENSIONS_DIR=$rel_c/ext" in be, "")

check("the smoke's ext probe runs as a client would (neutral cwd, image env)",
      "cd /tmp && timeout -k 5 ${SMOKE_EXT_PROBE_S:-60}" in src("pair_smoke.sh")
      and "TORCH_CUDA_ARCH_LIST" not in src("pair_smoke.sh"), "")

sm = src("pair_smoke.sh")
check("smoke keeps the arm's bytes", "SMOKE_ARM_LOG" in sm, "")
check("smoke prints the server log tail", "SMOKE_SERVER_TAIL" in sm, "")
# A decode takes ~6 rows per layer and touches the whole CPU-expert split; a worker whose own memory
# is in swap answers it from the disk. The smoke therefore decodes first (on the fresh load), prefills
# last, and prints what the worker's own swap was at every phase, beside the box's avail and swap and
# the worker's pid, so a row can be labelled with the state it was taken in.
check("smoke samples the worker's own swap at every phase, selecting it by comm",
      all(k in sm for k in ("ws_check after_load", "ws_check before_arm", "ws_check after_decode",
                            "ws_check after_prefill", "worker_seam_server[.]py"))
      and "paging_level" in sm, "")
# Option B (owner, 2026-11-03): the absolute 256 MiB refusal is gone, because GX2 walks 1.2-1.7 GiB
# into swap merely by mapping the pack on a ~121 GiB box with nothing of ours running -- it refused
# every honest fresh load (344 MiB at after_load, 22:02). What is refused is GROWTH inside the decode
# window: baseline at `before_arm` in the same run, sample right after the decode rows, > 512 MiB. The
# prefill window is warned, never refused (356 MiB at ws 256 / 734 at ws 1024 are expected), and one
# absolute level survives for a second tenant's pack.
check("the swap gate is growth over the decode window, not an absolute level",
      "SMOKE_WSWAP_GROW_MAX=${SMOKE_WSWAP_GROW_MAX:-512}" in sm
      and "SMOKE_WSWAP_MAX=${SMOKE_WSWAP_MAX:-2048}" in sm
      and 'ws_growth_check decode "$ws_before_arm" "$ws_after_decode" 1' in sm
      and 'ws_growth_check prefill "${ws_after_decode:-$ws_before_arm}" "$ws_after_prefill" 0' in sm
      and "paging_growth" in sm and "paging_before_decode" not in sm, "")
check("the decode window closes before any prefill or parsing",
      sm.index('ws_check after_decode') < sm.index('ws_growth_check decode')
      and sm.index('ws_check after_decode') < sm.index('ws_growth_check prefill')
      and sm.index('ws_check after_decode') < sm.index('pair_step "SMOKE_PREFILL'), "")
check("a post-load reading that cannot be post-residency is printed, not judged",
      "SMOKE_WSWAP_JUDGE_MIN=${SMOKE_WSWAP_JUDGE_MIN:-2000}" in sm
      and sm.count("SMOKE_WSWAP_UNJUDGED") >= 2
      and sm.index("SMOKE_WSWAP_JUDGE_MIN") < sm.index("paging_level"), "")
check("every reading names the load's box pressure and the worker it read",
      all(k in sm for k in ("load_swapped_mib", "load_avail_mib", "box_swap_used_mib",
                            "avail_mib=%d box_swap_used_mib=%d", "pid=%d vm_swap_mib=%d rss_mib=%d")), "")
check("smoke refuses a box another tenant is holding",
      "SMOKE_MEM_STRICT" in sm and 'fail "memory box_busy' in sm
      and sm.index("SMOKE_MEM_READY") < sm.index('fail "memory box_busy'), "")
# The worker's own SEAM_SERVER_LOADED line names the release it loaded; the smoke has to read it and
# refuse a worker that is not the release under test, before it prints a single reading. Without it a
# number taken on a stale worker is indistinguishable from the release's (2026-11-03: the check was
# lost in an edit and every reading printed `load_swapped_mib=-`).
check("smoke verifies the worker loaded the release it is smoking",
      'grep -m1 \'^SEAM_SERVER_LOADED\'' in sm and "SMOKE_SERVER_LOADED $loaded" in sm
      and "server_build_mismatch" in sm
      and sm.index("server_build_mismatch") < sm.index("ws_check after_load")
      and sm.index("SMOKE_SERVER_LOADED") < sm.index("SRV_LOAD_SWAPPED"), "")
# The growth gate is only meaningful over steady state: on a box at 900 MiB avail the *first* forward
# in a client swaps ~330 MiB of the worker out, so a window opened on a cold client measures the
# box's first touch and refuses an arm whose own decode grew nothing (2026-11-03 04:05). The smoke
# therefore runs the same arm once, discarded, before it takes the before_arm baseline.
check("the decode window opens after one discarded arm, not on the client's first touch",
      'SMOKE_ARM_WARM:-1' in sm
      and sm.index("SMOKE_ARM_WARM") < sm.index("ws_check before_arm")
      and sm.index("ws_check before_arm") < sm.index('pair_step "SMOKE_ARM pair='), "")
check("smoke decodes before it prefills",
      sm.index('pair_step "SMOKE_ARM') < sm.index('pair_step "SMOKE_PREFILL'), "")

ur = src("pair_update_run.sh")
check("update records whether the pair was serving", "UPDATE_SERVER_BEFORE" in ur, "")
check("a recreate refuses while any P7 GPU 0/1 has a foreign compute process",
      'gpu_busy p7_gpus=0,1' in ur
      and ur.index('pair_gpu_foreign_users 0 1 "$P7CT"') < ur.index('pair_containers.py" apply')
      and 'PAIR_GPU_WAIT_MIN' in ur, "")
check("the GPU probe covers every P7 GPU by UUID, and the pair's own container is exempt",
      'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name' in src("pair_lib.sh")
      and 'pair_pid_in_container' in src("pair_lib.sh")
      and 'not this recipe, not touched' in ur, "")

# The canary gate: prod is reachable only through pair_promote.sh, and only for a sha dev proved by
# generating token ids. These are source invariants because the alternative -- trusting the flags at
# run time -- is exactly the drift this tooling exists to stop.
check("prod update refuses without the promote flag", 'PAIR_PROMOTE:-0' in src("pair_update.sh"), "")
check("prod update names the only way in", "pair_promote.sh" in src("pair_update.sh"), "")
check("promote passes the flag and nothing else does",
      'PAIR_PROMOTE=1 "${cmd[@]}"' in src("pair_promote.sh")
      and 'PAIR_PROMOTE=1' not in src("pair_update_run.sh"), "")
check("promote queues a flip-only ticket (prepares out of band)",
      "cmd+=(--flip-only)" in src("pair_promote.sh"), "")
check("promote requires per-sha dev evidence", "dev/evidence/$SHA.json" in src("pair_promote.sh"), "")
check("promote wants decode ids, not just a row", "pair_canary_gate" in src("pair_promote.sh"), "")
check("one definition of the canary gate",
      sum(s.count("no_decode_ids") for s in (src("pair_lib.sh"), src("pair_update.sh"),
                                            src("pair_promote.sh"), src("pair_update_run.sh"),
                                            src("pair_smoke.sh"))) == 1, "")
check("the script that flips checks the canary too",
      "pair_canary_gate" in ur, "")
check("update_run needs the sha for prod", '[ "$PAIR" = prod ]' in ur, "")
check("box memory refusals are classified, not blamed on the release",
      ur.count("SMOKE_FAIL memory ") == 1 and "memory_fail" in ur, "")
check("a memory failure keeps the code (no restore)",
      "rollback=none" in ur and "restore\n}" not in ur.split("memory_fail()")[1].split("\n}")[0], "")
check("smoke runs the arm with ids", "--ids" in sm, "")
# The stamp names the artifact, but only an import can say whether the next client reuses it. A cache
# written under another include-path spelling has a full plan waiting; inside a smoke that plan runs
# under load and evicts the pack (2026-11-02: 565.2 tok/s). The probe must come before the load.
check("a smoke proves the client cache before it loads a pack",
      "SMOKE_EXT_HIT" in sm and "ext_client_miss" in sm
      and sm.index("ext_client_miss") < sm.index('pair_step "SMOKE_SERVER_START'), "")

check("missing ids fail the smoke", "arm_no_ids" in sm, "")
check("smoke totals the ids across arms", "SMOKE_ARM_IDS_TOTAL" in sm, "")
check("update records the ids total, not one arm",
      "SMOKE_ARM_IDS_TOTAL n=" in ur, "")
# the success line is the one that also claims the check; the memory-failure line above it says
# rollback=none and must not be mistaken for it
_ok = 'PAIR_UPDATED pair=$PAIR sha=$SHA build=$BUILD_ID check='
check("update writes evidence before announcing",
      ur.index("evidence ok") < ur.index(_ok), "")
check("failed update writes failed evidence", 'evidence failed "$*"' in ur, "")
# An admission is a historical fact about the sha, not a receipt for the latest attempt: a dev re-smoke
# that fails must not erase the ok file prod was promoted on (dev, 2026-11-02 03:21: a re-run of the live
# release wrote ok=false over the admission, and the record prod refuses without was gone).
check("a failed re-run never erases an admission",
      "admitted = bool(json.load(open(path)).get(\"ok\"))" in ur
      and '.failed.json' in ur and "admission=" in ur, "")


def code_lines(text):
    return [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]

# The client adopts the intersection of what it asks for and what the server grants, and its
# fallback default lists 1,6,64,256,512,1024,2048,4096 -- no 8192. So a client that asks for
# nothing gets 1,6,64 from an 8192-granting server and caps every prefill chunk at 64 rows (dev,
# 2026-11-02). The arm asks for the pair's list, which is the server's list; ask for a superset
# and the pair would still run, but on a width the release was never measured at.
check("the client asks for the pair's buckets",
      'SEAM_BUCKETS=$BUCKETS' in "".join(l.replace('"', '') for l in code_lines(src("pair_arm.sh"))), "")
check("the server is asked for the wide grant",
      all(int(pc.load_conf(q)["buckets"].split(",")[-1]) >= 2048 for q in ("prod", "dev")),
      str([pc.load_conf(q)["buckets"] for q in ("prod", "dev")]))
check("a release prefix resolves to the commit before it is compared",
      'rev-parse --verify "$SHA_ARG^{commit}"' in src("pair_update_run.sh")
      and sum('rev-parse --verify "$SHA^{commit}"' in src(f)
              for f in ("pair_build_lib.sh", "pair_build_ext.sh")) == 2, "")
check("an update ticket carries a command",
      'cmd=("$HERE/pair_update_run.sh"' in src("pair_update.sh"), "")
check("an empty ticket is refused, not queued",
      "no command given" in src("pair_lib.sh"), "")
# The smoke is the canary for what ships, so its server gets the pair's own grant -- not less: with
# only 1,6,64 the 32k prefill would be served in 64-row chunks and the rate gate would measure a shape
# no pair runs. What it must not do is ask for a bucket it never fills, so the prefill phase has to be
# deep enough to reach the widest bucket of both pairs.
check("the smoke grants the pair's own buckets",
      "SMOKE_BUCKETS=${SMOKE_BUCKETS:-$BUCKETS}" in sm
      and "--buckets $SMOKE_BUCKETS" in sm, "")
_depth = [l for l in code_lines(sm) if l.startswith("SMOKE_PREFILL_DEPTH=")][0]
_depth = int(_depth.split(":-")[1].rstrip("}").rstrip('"'))
check("the smoke's prefill is deep enough to fill the widest bucket it grants",
      all(_depth >= int(pc.load_conf(q)["buckets"].split(",")[-1]) for q in ("prod", "dev")),
      f"depth={_depth} " + str([pc.load_conf(q)["buckets"] for q in ("prod", "dev")]))
# One payload slot is the shipping shape (seam_server_ensure.sh's own default, and the slab the validated
# server printed); a second is only memory. The rollback has to start the server the same way.
check("the smoke asks for the shipping slot count", "SMOKE_SLAB_SLOTS:-1" in sm, "")
check("smoke passes the slot count to the ensure", "SEAM_SLAB_SLOTS=$SMOKE_SLAB_SLOTS" in sm, "")
check("the rollback serves the same slots", "SEAM_SLAB_SLOTS=${SMOKE_SLAB_SLOTS:-1}" in ur, "")
check("nothing sets the slower wide-dispatch tier",
      not any("EXL3_MOE_FUSED_ROWS_WIDE" in l
              for f in ("pair_arm.sh", "pair_env.sh", "pair_smoke.sh", "pair_update_run.sh")
              for l in code_lines(src(f))), "")
check("a pair root that is not absolute is refused",
      "pair root must be absolute" in src("pair_lib.sh"), "")
def _tracked_release_trees():
    out = subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(HERE)),
                          "ls-tree", "-r", "--name-only", "HEAD", "--", "qmesh/release"],
                         capture_output=True, text=True).stdout.split()
    return [f for f in out if f.endswith("/RELEASE.json") or "/src/" in f]


check("no extracted release tree is tracked inside the tooling tree",
      not _tracked_release_trees(), ",".join(_tracked_release_trees()[:3]))
check("a JIT lock left by a dead build is cleared, and only when nothing alive owns it",
      'SEAM_ENSURE stale torch build lock(s):' in src("seam_server_ensure.sh")
      and src("seam_server_ensure.sh").index("seam_gxct \"rm -f $SEAM_LOG\"")
          < src("seam_server_ensure.sh").index("stale torch build lock")
      and 'pair_torch_locks' in src("pair_smoke.sh")
      and 'pair_torch_build_live' in src("pair_smoke.sh"), "")
check("the busy-box probe neither counts stubs nor its own wrapper",
      '--stub' in src("pair_lib.sh")
      and 'worker_seam_server[.]py' in src("pair_lib.sh"), "")
check("a busy box is refused before anything is cut or re-created",
      'pair_gx_foreign_server' in src("pair_update_run.sh")
      and src("pair_update_run.sh").index("pair_gx_foreign_server")
          < src("pair_update_run.sh").index("phase containers"), "")
check("the smoke repairs a container that lost NVML, but only after it owns the box",
      'pair_ensure_nvml p7 "$P7CT"' in src("pair_smoke.sh")
      and 'pair_ensure_nvml gx "$GXCT"' in src("pair_smoke.sh")
      and src("pair_smoke.sh").index("pair_gx_gate_take") < src("pair_smoke.sh").index("pair_ensure_nvml p7")
      and 'PREPARE_NVML_WARN' in src("pair_prepare.sh")
      and "docker restart" not in src("pair_prepare.sh"), "")
check("a claim carries its host, and only a same-host pid may declare it dead",
      "host=%s" in src("pair_lib.sh")
      and 'hostname -s' in src("pair_lib.sh")
      and 'flock' in src("pair_lib.sh")
      and r'host=\([^ ]*\)' in src("seam_server_ensure.sh"), "")
check("a dry run cannot declare a free GPU for a real ticket",
      '[ "$PAIR_DRY" = 1 ] && [ -n "${PAIR_DRY_GPU:-}" ]' in src("pair_update_run.sh"), "")
check("a rehearsed cut keeps its seeded release stand-in",
      '[ "$PAIR_DRY" != 1 ]' in src("pair_cut.sh"), "")
check("a release is stamped once for both sides",
      "CUT_ADOPT" in src("pair_cut.sh") and "does not describe this commit" in src("pair_cut.sh"), "")
check("release dir resolves a sha prefix", 'releases/"$1"*' in src("pair_lib.sh"), "")

# The client's CPU-host worker is started with multiprocessing "spawn", which re-executes the arm
# driver by path. Fed on stdin the driver's main file is "<stdin>": the child dies on
# FileNotFoundError and the parent only ever sees EOFError, which is not a release fault.
check("the arm driver is run by path, not fed on stdin",
      "python3 -u -" not in code_lines(src("pair_arm.sh"))
      and "pair_ct_path" in src("pair_arm.sh"), "")
check("the ids driver runs only as the main module",
      'if __name__ == "__main__":' in src("pair_arm_ids.py"), "")
check("a path handed to the containers is translated to their mount",
      'echo "/pair/${p#"$PAIR_ROOT"/}"' in src("pair_lib.sh")
      and 'echo "/artifacts/${p#"$art"/}"' in src("pair_lib.sh"), "")
check("the smoke reads the worker's own refusal, not the wrapper's rc",
      "*SEAM_NO_MEMORY*" in src("pair_smoke.sh")
      and 'fail "memory server_no_memory' in src("pair_smoke.sh"), "")

check("a failed smoke stops the server it started",
      "RESTORE_SERVER_STOP" in src("pair_update_run.sh")
      and "'^python3 -u qmesh/worker_seam_server.py'" in src("pair_update_run.sh"), "")

# 2026-11-02: the worker is launched as `bash -lc 'cd /src/exllamav3 && stdbuf … python3 -u
# qmesh/worker_seam_server.py …'`, so while the wrapper lives its cmdline mentions the script and an
# unanchored pgrep/pkill reaches it first. Recording that pid and TERMing it stops a shell and orphans
# the process holding 116 GiB of pack. Every site that records or stops a server must anchor on the
# worker's own argv; the counts that only ask "is anything serving" may over-match.
_stops = []
for _n in ("pair_lib.sh", "pair_containers.py", "pair_update_run.sh", "seam_server_ensure.sh",
           "pair_smoke.sh", "pair_build_ext.sh"):
    for _l in src(_n).splitlines():
        if "worker_seam_server" in _l and "pkill" in _l and "^python3" not in _l:
            _stops.append(_n)
        if "pair_server_record" in _l and "pgrep" in _l and "^python3" not in _l:
            _stops.append(_n)
check("a server stop or record is anchored on the worker's argv, never the launcher's",
      not _stops, ",".join(sorted(set(_stops))))

# A released tree carries no .git and its container carries no SEAM_BUILD_ID, so qes.build_id()
# answered "unknown" and the session handshake accepted a mixed pair on proto alone (2026-11-01).
# Both halves must name the same build: the cut writes <tree>/qmesh/BUILD_ID, the arm exports it.
check("the cut writes the build id into the release tree",
      src("pair_cut.sh").count('$tmp/src/qmesh/BUILD_ID') == 2, "")
check("the client half is told which build it runs",
      'envs+=(-e "SEAM_BUILD_ID=$BUILD_ID")' in src("pair_arm.sh")
      and '"$PAIR_ROOT/current/RELEASE.json"' in src("pair_arm.sh"), "")

# Measurement-only instrumentation has to be reachable without adding a second driver, and it has to
# be opt-in: appended after the shipping list (so a dry run of a plain arm is unchanged), one place,
# never a shipping value.
_pa = src("pair_arm.sh")
_loop = 'for kv in ${PAIR_ARM_EXTRA_ENVS:-}; do envs+=(-e "$kv"); done'
# A client that imports another checkout must not share the release's extension cache: torch's
# version hash follows the include-path spelling, so the second tree rebuilds the pair's .so in
# place (2026-11-03: /artifacts/dspark-tpr/src wrote 5780205b into a cache stamped 04a61140 and
# pair_serve refused the pair as DRIFT for an hour). One supported way in, and it is isolated.
_pa = src("pair_arm.sh")
# The queue stopped being idle for long enough to install out of band, so a mid-ticket install has to
# be possible without a guard bypass: the caller proves membership by ancestry, and every other
# ticket still refuses.
_pi = src("pair_install.sh")
check("pair_install runs inside the ticket it is part of, and only that one",
      "in_ancestry() {" in _pi and '"${PAIR_INSTALL_SELF_TICKET:-0}" = 1 ] && in_ancestry "$pid"' in _pi
      and _pi.count('die "ticket_in_flight') == 1 and "ps -o ppid=" in _pi, "")

# A container whose pid 1 is `sleep infinity` never reaps: every build a killed probe leaves behind
# becomes a Z-state child of init with a fresh etimes, and the compiler guard then waits the full
# budget on processes that cannot finish anything (2026-11-03: 14 zombies held pair_build_ext for
# 20 min). Zombies are not builds.
_pe = src("pair_build_ext.sh"); _pl = src("pair_build_lib.sh")
# The baton clear has to happen in the shell that imports: the first clear runs before the symlink
# flip, and /pair/extcache is its own directory, so clearing one spelling left the other -- and a
# surviving lock does not fail the build, it sleeps in file_baton.wait with no compiler running.
check("the ext build clears the baton where it imports",
      _pe.count('for lck in $rel_c/ext/exllamav3_ext/lock $MOUNT/extcache/exllamav3_ext/lock; do') == 1
      and _pe.index('EXT_LOCK cleared=\\$lck') < _pe.index('EXPECT_SRC=$rel_c/src/exllamav3 timeout'), "")

check("the compiler guard counts only live compilers",
      _pe.count("ps -eo stat=,etimes=,comm=") == 1 and _pl.count("ps -eo stat=,etimes=,comm=") == 1
      and "\$1 !~ /^Z/" in _pe and "\$1 !~ /^Z/" in _pl
      and "ps -eo etimes=,comm=" not in _pe, "")

check("a client run from another src gets its own extension cache",
      "--src) SRC=$2" in _pa
      and "/src/exllamav3|/src/exllamav3/) EXT_DIR=/pair/extcache" in _pa
      and 'EXT_DIR=$(pair_ct_path "$PAIR_STATE_DIR/extcache/$ext_tag")' in _pa
      and _pa.count('mkdir -p "$PAIR_STATE_DIR/extcache/$ext_tag"') == 1
      and '-e "PYTHONPATH=$SRC" -e "TORCH_EXTENSIONS_DIR=$EXT_DIR"' in _pa
      and _pa.count("cd $SRC &&") == 2 and "cd /src/exllamav3 &&" not in _pa, "")
# `/pair/current/src` is the release tree under another spelling, and torch's version hash covers the
# spelling: a client that imports through it while pointed at the release cache rebuilds
# exllamav3_ext.so in place and leaves BUILD_STAMP.json naming the artifact it replaced. On 2026-11-03
# the dev cache was left at ext_so=c7908310 want=e74d4951 (build.ninja `-I/pair/current/src/exllamav3`)
# and every dev ticket read DRIFT until the cache was rebuilt through /src/exllamav3. So the alias is
# canonicalised, and only the container's own link may name the release cache.
check("the /pair/current/src alias is canonicalised, never given the release cache",
      "ARM_SRC_CANON src=$SRC -> /src/exllamav3" in _pa
      and "/pair/current/src|/pair/current/src/)" in _pa
      and "/pair/current/src) EXT_DIR=/pair/extcache" not in _pa
      and _pa.index("ARM_SRC_CANON") < _pa.index("EXT_DIR=/pair/extcache"), "")
_pp = src("pair_prepare.sh")
check("prepare creates /src, not only repoints it",
      "relink_cmd='mkdir -p /src && ln -sfn /pair/current/src /src/exllamav3" in _pp
      and _pp.count('sh -c "$relink_cmd"') + _pp.count("sh -c '$relink_cmd'") == 2
      and "if [ -L /src/exllamav3 ]; then ln -sfn" not in _pp, "")
_ps = src("pair_sync_check.py")
check("...and verify names the tree that wrote the cache",
      _ps.count('EXTSRC $(grep -ho') == 1
      and 'extcache_foreign_src={src_spelled}->{src_real} want={want_src}' in _ps
      and 'want_src = f"{rel_dir}/src"' in _ps, "")
check("...and names a cache built through an alias of the same tree",
      'f"extcache_spelling={src_spelled} want=/src/exllamav3/..."' in _ps
      and 'not src_spelled.startswith("/src/exllamav3")' in _ps, "")

check("the client can take extra env without a second driver",
      _pa.count(_loop) == 1
      and _pa.index("SEAM_COPY_STREAM=1") < _pa.index(_loop)
      < _pa.index('envs+=(-e "PREFILL_ARMS=$PREFILL_DEPTH:8192"'), "")
# The server half needs the same door, for the knobs the *worker* reads. SEAM_MOE_CPP is one of them:
# set in the client's env it is inert, and on 2026-11-03 04:16/04:31 an A/B "hook-on vs hook-off" ran
# the identical Python path twice (rows of 115.53 and 114.64 ms/tok, one ids digest). The hook must
# reach the worker's own process through the ensure's SEAM_ENV, must be empty by default, and the
# smoke has to print what the worker actually installed.
_lb, _sm = src("pair_lib.sh"), src("pair_smoke.sh")
check("the server can take extra env, and the smoke prints what the worker installed",
      'SRV_ENV_EXTRA=${PAIR_SRV_EXTRA_ENVS:-}' in _lb
      and 'for kv in ${PAIR_SRV_EXTRA_ENVS:-${SRV_ENV_EXTRA:-}}' in _lb
      and 'SEAM_*) extra="$extra $kv"' in _lb
      and 'SMOKE_SERVER_CPP asked=' in _sm
      and 'SEAM_MOE_CPP_(ON|OFF)' in _sm, "")
# The hook is not a knob any more: it measured 113.98 -> 77.19 ms/tok with identical ids and text, so
# the shipped server env carries SEAM_MOE_CPP=1 (MOE_CPP=0 is the A/B escape). A release that silently
# dropped it would serve the slow path at every quoted number.
check("the shipped server env serves the C cold path by default",
      'MOE_CPP=${MOE_CPP:-1}' in _lb
      and 'SEAM_MOE_CPP=$MOE_CPP' in _lb
      and re.search(r"export .*\bMOE_CPP\b.*\bSRV_ENV_EXTRA\b", _lb), "")

# Installing tooling is a release-system operation too: a `cp` over a script a running ticket has open
# makes bash read shifted lines (2026-11-03 01:29Z, a prefill thrown away on `drv_host: unbound
# variable` from lines that had already run). The installer takes the queue, then renames.
_pi = src("pair_install.sh")
check("installs wait for the queue and rename into place",
      'ticket_in_flight queue=$q agent=' in _pi and 'kill -0 "$pid"' in _pi
      and 'mv -f -- "$tmp" "$DST/$f"' in _pi
      and 'cp -- "$src" "$DST/$f"' not in _pi, "")

# No load may run on the worker's own 2048 MiB default: the pair states what the pack load must
# leave behind, and both places that start a server have to pass it (measured GX numbers).
check("the load margin is the pair's number, not the worker's default",
      pc.load_conf("dev")["load_margin"] == "2048"
      and pc.load_conf("prod")["load_margin"] == "2048", "")
# The server's environment is a release property in four parts -- the load margin, the tuner cache, the
# MOE kernel with its memory margin, and the piece size with the picker off -- and it used to be written
# out at each start. Two copies is how the rollback once restarted a server with the worker's own
# defaults while the smoke used the pair's numbers; there is one builder (pair_server_env in pair_lib.sh)
# and both starts call it, so a knob cannot reach one and not the other.
check("the server environment has one definition",
      "pair_server_env() {" in src("pair_lib.sh")
      and '%s %s%s\' "$1" "$LOAD_MARGIN" "$PAIR_TUNE_CACHE" "$MOE_ENV" "$WS_ENV" "$extra"' in src("pair_lib.sh")
      and "SEAM_MOE_KERNEL=cudaexl3 SEAM_MOE_PRELOAD=1" in "\n".join(code_lines(src("pair_lib.sh")))
      and "SEAM_CX_MARGIN_MIB=$CX_MARGIN_MIB" in "\n".join(code_lines(src("pair_lib.sh"))), "")
check("both server starts use that definition",
      sm.count('SEAM_ENV="$(pair_server_env ') == 1
      and ur.count('SEAM_ENV="$(pair_server_env ') == 1
      and not any("SEAM_LOAD_MARGIN_MIB=" in l or "SEAM_WS_ROWS=" in l
                  for f in ("pair_smoke.sh", "pair_update_run.sh") for l in code_lines(src(f))), "")
check("the piece size is the pair's number, pinned",
      pc.load_conf("dev")["ws_rows"] == "1024"
      and pc.load_conf("prod")["ws_rows"] == "1024"
      and 'WS_ENV="SEAM_WS_ROWS=$WS_ROWS SEAM_WS_ROWS_AUTO=0"' in src("pair_lib.sh"), "")
# One pack per GX box. A build and a smoke both load or hammer that box, so both must claim it through
# the take helper (which reads a foreign claim and waits); writing the file directly would let two jobs
# believe they own the box, which is what collided on 2026-11-02. Serving is the third job that loads a
# pack, so it takes the claim the same way.
check("every GX job claims the box through the helper",
      all(any("pair_gx_gate_take " in l for l in code_lines(src(f)))
          for f in ("pair_build_ext.sh", "pair_build_lib.sh", "pair_smoke.sh", "pair_serve.sh"))
      and not any("pair_gx_gate_up " in l
                  for f in ("pair_build_ext.sh", "pair_build_lib.sh", "pair_smoke.sh", "pair_serve.sh")
                  for l in code_lines(src(f)))
      and "pair_gx_gate_state() {" in src("pair_lib.sh")
      and "pair_gx_gate_down() {" in src("pair_lib.sh"), "")
# "The pair is down" must be a state somebody chose, reached from the release `current` names and only
# after the sync check says the artifact on disk is the stamped one; and the worker it leaves behind has
# to be named in the pidfile, or the next ticket can only stop it by pattern.
check("serving starts from `current`, refuses drift, and records the worker",
      "pair_current_sha" in src("pair_serve.sh")
      and "pair_check.sh" in src("pair_serve.sh")
      and "PAIR_SERVE_DRIFT" in src("pair_serve.sh")
      and "pair_server_record" in src("pair_serve.sh")
      and "--fresh" in src("pair_serve.sh")
      and "pair_gx_foreign_server" in src("pair_serve.sh"), "")
check("the gate test is in the battery",
      os.path.exists(os.path.join(HERE, "tests", "gx_gate.sh")), "")
# The smoke's two pre-load guards (NVML on both ends, a JIT lock nobody owns) are the difference between
# a 30 s diagnosis and a lost ticket; the third job that loads a pack must not rediscover that.
check("serving checks NVML on both ends and the torch lock before it loads",
      "pair_ensure_nvml p7" in src("pair_serve.sh")
      and "pair_ensure_nvml gx" in src("pair_serve.sh")
      and "pair_torch_locks" in src("pair_serve.sh")
      and src("pair_serve.sh").index("pair_gx_gate_take") < src("pair_serve.sh").index("pair_ensure_nvml p7"), "")

# A GX command runs on the host unless it says docker exec, and the host has no /pair: a container path
# passed to a bare pair_gxsh lookup reports files missing that are there, and a bare chown does nothing.
# (2026-11-02: the cuda-exl3 build succeeded and the stamp said `no_so_in_release_tree`.)
check("GX host commands never take a container path",
      not any("rel_c" in l and "pair_gxsh" in l and "docker exec" not in l
              for l in code_lines(src("pair_build_ext.sh")))
      and not any("rel_c" in l and "pair_gxsh" in l and "docker exec" not in l
                  for f in ("pair_smoke.sh", "pair_update_run.sh") for l in code_lines(src(f))), "")
check("a glob in a docker exec goes through a shell",
      not any("docker exec" in l and "_C*" in l and "sh -c" not in l
              for f in ("pair_build_ext.sh", "pair_smoke.sh", "pair_update_run.sh", "pair_lib.sh")
              for l in code_lines(src(f))), "")

# `python3 -` reads its *program* from stdin, so a heredoc plus a pipe (`... | python3 - <<'PY'`) leaves
# the parser an empty stream. That silently turned a complete 8-token arm row into "arm_no_tokens NONE"
# on dev (2026-11-02) and would have written empty references. The arm's rows are parsed by a helper the
# battery can test; no shell script may pipe data into a heredoc'd python again.
check("arm rows are parsed by a file, not through a heredoc's stdin",
      'pair_arm_rows.py' in src("pair_smoke.sh")
      and os.path.exists(os.path.join(HERE, "pair_arm_rows.py"))
      and not any(re.search(r"\|\s*python3\s+-\s+[^|]*<<", l)
                  for f in ("pair_smoke.sh", "pair_ref.sh", "pair_update_run.sh")
                  for l in code_lines(src(f))), "")
check("the arm-row parser is in the battery",
      os.path.exists(os.path.join(HERE, "tests", "arm_rows.sh")), "")
# The smoke asks a box that has just unloaded 113 GiB to prove it came back, before the worker's own
# refusal decides the ticket: a load started into a box still releasing memory reads as a bad release.
check("the smoke waits for the GX box before the load preflight",
      'SMOKE_MEM_READY' in src("pair_smoke.sh")
      and any("SMOKE_MEM_WAIT_S" in l for l in code_lines(src("pair_smoke.sh")))
      and "LOAD_MARGIN" in src("pair_smoke.sh"), "")
# The arm must ask for the pair's list. The client adopts the intersection of its request and the
# server's grant, so a request that misses the shipping width (8192) is a silent 64-row cap: on
# 2026-11-02 the dev arm negotiated accepted=1,6,64 against a server granting 1,6,64,8192 and every
# prefill chunk went 128 calls deep. Pairs.conf is where the list lives; the arm quotes it.
conf = open(os.environ["PAIR_CONF"]).read()
conf_buckets = [l.split()[11] for l in conf.splitlines()
                if l.split()[:1] and l.split()[0] in ("prod", "dev") and len(l.split()) > 11]
check("every pair ships the prefill width its release is measured on",
      len(conf_buckets) == 2 and all(b == "1,6,64,8192" for b in conf_buckets),
      "pairs.conf buckets: %s" % conf_buckets)
check("the arm requests the pair's buckets, not a client fallback",
      'SEAM_BUCKETS=$BUCKETS' in "".join(l.replace('"', '') for l in code_lines(src("pair_arm.sh"))), "")
check("the smoke fails when the grant is not the pair's list",
      'bucket_grant' in src("pair_smoke.sh") and 'SMOKE_BUCKETS_GRANT' in src("pair_smoke.sh"), "")

check("the grant comparator is a tested file, not an inline comparison",
      'pair_bucket_grant.py' in src("pair_smoke.sh")
      and os.path.exists(os.path.join(HERE, "pair_bucket_grant.py"))
      and os.path.exists(os.path.join(HERE, "tests", "bucket_grant.sh")), "")
# The stub smoke must not take the live gate: it is a test of one failure path, and waiting out the
# real dev smoke turned the battery into a 20-minute hang.
check("the stub smoke keeps its own gate directory",
      any("PAIR_GATE_DIR" in l for l in code_lines(src("tests/smoke_memory.sh"))), "")
# An interrupted ticket used to leave the pack resident (23:32 on 2026-11-02) and the next smoke
# refused `pair_busy gx_procs=2`. The fix is ownership, not a pattern kill: record the worker's pid,
# stop only that pid, and reclaim it at the next ticket's preflight.
check("the smoke records the worker it started",
      'pair_server_record' in src("pair_smoke.sh"), "")
check("an interrupted smoke stops its own server, a verified one keeps it",
      'SMOKE_VERDICT' not in src("pair_smoke.sh")
      and 'PAIR_SMOKE_KEEP_SERVER' not in src("pair_smoke.sh")
      and 'PAIR_SMOKE_KEEP_SERVER' not in src("pair_update_run.sh")
      and 'pair_server_stop' in src("pair_smoke.sh")
      and src("pair_smoke.sh").index("pair_server_stop") < src("pair_smoke.sh").index("pair_gx_gate_down")
      and 'smoke_exit' in src("pair_smoke.sh"), "")
check("the interrupt path stops by recorded pid, not by pattern",
      'pair_server_stop' in src("pair_update_run.sh")
      and 'pair_client_stop' in src("pair_update_run.sh")
      and not any('pkill -f worker_seam_server' in l for l in src("pair_update_run.sh").split("on_term")[1].split("restore()")[0].splitlines()), "")
_upd = src("pair_update_run.sh")
check("the preflight reclaims this pair's own orphan before refusing",
      _upd.count("others=$(pair_others_busy)") == 1
      and 'UPDATE_ORPHAN' in _upd
      and _upd.index('pair_server_stop 2>&1') < _upd.index('others=$(pair_others_busy)'), "")
check("a verified flip leaves no server behind it",
      'PAIR_SMOKE_KEEP_SERVER' not in src("pair_update_run.sh")
      and 'rm -f "$(pair_server_pidfile)"' not in src("pair_update_run.sh"), "")
# A pair's first release has no containers to reuse, and the lib build runs inside them: prepare that
# answers "unchanged" to a CREATE plan leaves the release cut and the pair unable to start a server.
check("prepare creates a missing pair container instead of reporting it unchanged",
      '*CREATE*' in src("pair_prepare.sh") and 'pair_containers.py" apply' in src("pair_prepare.sh"), "")
check("prepare leaves a matching pair alone",
      'unchanged' in src("pair_prepare.sh"), "")
# A release's stamp must not travel on the gx ssh stdin: a redirect written outside
# `$(pair_gxsh …)` is not reliably applied inside the substitution, the remote `cat` reads the
# host's own stdin, and the release ships a 0-byte RELEASE.json that only verify notices -- after
# a green smoke has already moved `current` (gx2, 9fc3ca8a, 2026-11-02).
_cut = src("pair_cut.sh")
check("the gx stamp travels in the command, not on stdin",
      "base64 -w0" in _cut and "cat > '$tmp/RELEASE.json'" not in _cut, "")
check("a cut refuses a stamp it cannot read back",
      "STAMP_EMPTY" in _cut and "STAMP_WRITE_FAILED" in _cut, "")
check("a cut refuses stamps that differ between the sides", "STAMP_MISMATCH" in _cut, "")
check("an existing release with an empty stamp is repaired", "STAMP_REPAIR" in _cut, "")

# The stamp and the binary are two files. A build in place (a stray client import from another tree)
# rewrites the .so and leaves the stamp intact, so the pair reads IN_SYNC while the container serves
# an artifact no build produced: on 2026-11-02 a probe build's 27,513,952-byte .so became the client's
# 27,509,856-byte one and pair_check still printed OK. Both hashes are now compared against the
# stamp -- the remote probe stubbed, the comparison real.
_json = __import__("json")
_spec2 = importlib.util.spec_from_file_location("psc", os.path.join(HERE, "pair_sync_check.py"))
psc = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(psc)
_sha = "9fc3ca8aa8388340e324507aae237c40c1019a29"
_bid = "dev-9fc3ca8aa838-20261102T035748Z"
_root = "/pair"
_paths = ["a.py"]
_ref = {p: p for p in _paths}
_mf = {"pair": "dev", "release_sha": _sha, "build_id": _bid, "_paths": _paths,
       "p7": {"container": "seam-p7-dev", "arch": "x86_64", "root": _root, "sm": "sm_120",
              "cuda_arch": "12.0"},
       "gx": {"container": "seam-gx2-dev", "arch": "aarch64", "root": _root, "sm": "sm_121",
              "cuda_arch": "12.1"}}
def _state(side, **over):
    st = {"LIVE": _root + "/releases/" + _sha, "EXT_LIVE": _root + "/releases/" + _sha + "/ext",
          "RELJSON": _json.dumps({"sha": _sha, "build_id": _bid}),
          "LIBSTAMP": _json.dumps({"build_id": _bid, "source_sha": _sha, "sha256": "aa"}),
          "LIBSHA": "aa",
          "EXTSTAMP": _json.dumps({"build_id": _bid, "source_sha": _sha,
                                   "arch_list": [_mf[side]["cuda_arch"]], "sha256": "beef"}),
          "EXTSHA": "beef"}
    if side == "gx":
        st["EXTSTAMP"] = _json.dumps({"build_id": _bid, "source_sha": _sha,
                                      "arch_list": ["12.1"], "sha256": "beef",
                                      "cx": {"build_id": _bid, "source_sha": _sha,
                                             "arch_list": ["12.1"], "sha256": "cafe"}})
        st["CXSHA"] = "cafe"
    st.update(over)
    return st
def _check_side(side, st):
    psc.side_state = lambda m, s, remote, skip_ct=False, skip_srv=False: (st, dict(_ref), "")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = psc.check_side(_mf, side, dict(_ref), True, skip_ct=True, skip_srv=True)
    return ok, buf.getvalue()
_ok, _out = _check_side("p7", _state("p7"))
check("a cache whose .so is the stamp's .so verifies", _ok, _out.strip()[-120:])
_ok, _out = _check_side("p7", _state("p7", EXTSHA="26de612b"))
check("a cache rebuilt under its stamp is DRIFT, not IN_SYNC",
      not _ok and "ext_so=26de612b want=beef" in _out, _out.strip()[-160:])
_ok, _out = _check_side("p7", _state("p7", EXTSHA=""))
check("an unreadable ext .so is DRIFT", not _ok and "ext_so=unreadable" in _out, _out.strip()[-160:])
_ok, _out = _check_side("gx", _state("gx"))
check("a gx cache whose artifacts are the stamp's verifies", _ok, _out.strip()[-120:])
_ok, _out = _check_side("gx", _state("gx", CXSHA="0bad"))
check("a cuda-exl3 .so rebuilt under its stamp is DRIFT",
      not _ok and "cx_so=0bad want=cafe" in _out, _out.strip()[-160:])

# The release carries a second compiled artefact, the C moe hook, and until now nothing hashed it: a
# release root with a stale or missing libqmesh_moe_cpp.<arch>.so read IN_SYNC. The declaration in
# RELEASE.json (native.moe_cpp, written by the cut when the commit tracks build_moe.sh) decides
# whether it is required, so a release cut before the hook had a build script is not asked for one.
_decl = _json.dumps({"sha": _sha, "build_id": _bid, "native": {"moe_cpp": True}})
_cpp = _json.dumps({"build_id": _bid, "source_sha": _sha, "arch": "x86_64", "sha256": "ff"})
_ok, _out = _check_side("p7", _state("p7", RELJSON=_decl))
check("a release that declares the hook but carries no stamp is DRIFT",
      not _ok and "cpp_stamp=missing" in _out, _out.strip()[-160:])
_ok, _out = _check_side("p7", _state("p7", RELJSON=_decl, CPPSTAMP=_cpp, CPPSHA="ff"))
check("a hook .so that is the stamp's .so verifies", _ok, _out.strip()[-160:])
_ok, _out = _check_side("p7", _state("p7", RELJSON=_decl, CPPSTAMP=_cpp, CPPSHA="ee"))
check("a hook rebuilt under its stamp is DRIFT, not IN_SYNC",
      not _ok and "cpp_so=ee want=ff" in _out, _out.strip()[-160:])
_ok, _out = _check_side("p7", _state("p7"))
check("a release that does not declare the hook is not asked for it", _ok, _out.strip()[-160:])
check("the stage that builds the seam lib builds and stamps the hook too",
      all(s in src("pair_build_lib.sh") for s in
          ("CPP_BUILD", "CPP_OK", "build_moe.sh", "PAIR_CPP_LIB", "MOE_OBJ=$moe_tmp")), "")
check("the build skips the hook when the release does not declare it",
      'native") or {}).get("moe_cpp")' in src("pair_build_lib.sh")
      and "CPP_SKIP $side $arch release declares" in src("pair_build_lib.sh"), "")
check("both compiled artefacts are stripped before they are stamped, so a rebuild reproduces the cut's bytes",
      src("pair_build_lib.sh").count("strip --strip-unneeded") == 2
      and src("pair_build_lib.sh").index("strip --strip-unneeded $tmp/$PAIR_LIB.$arch.so")
      < src("pair_build_lib.sh").index("sha256sum bin/$PAIR_LIB.$arch.so")
      and src("pair_build_lib.sh").index("strip --strip-unneeded $moe_tmp/$moe")
      < src("pair_build_lib.sh").index("sha256sum bin/$moe"), "")
check("the cut declares whether the hook is required",
      'cat-file -e "$SHA:qmesh/native/build_moe.sh"' in src("pair_cut.sh")
      and '"native": {"moe_cpp"' in src("pair_cut.sh"), "")
check("pair_check reads the hook stamp and hashes the hook",
      "CPPSTAMP" in src("pair_sync_check.py")
      and "libqmesh_moe_cpp.%(arch)s.so" in src("pair_sync_check.py"), "")


# Owner rule 2026-11-02: an expert pack lives on the local drive of the box that serves it. The P7
# containers used to be handed a network mount so that the hot pack's links could be
# resolved; a pack reached that way makes serving speed depend on the network, and the GX half reads
# that same drive over NFS. The rule is two source invariants and one behaviour:
#   * no P7 spec mounts /mnt -- the mount is gone from both pairs' specs;
#   * prepare and smoke both run the checker (a check only one path remembers is not a gate);
#   * the spec change lands through a queued ticket (--recreate), never by hand.
import json as _json
for _pair, _ct in (("prod", "seam-p7-prod"), ("dev", "seam-p7-dev")):
    _sp = _json.load(open(os.path.join(HERE, "containers", f"{_pair}-p7.json")))
    _mt = _sp.get("mounts") or []
    check(f"{_pair}'s P7 spec mounts the pool itself at /models/dsv41",
          any(d == "/models/dsv41" and s == "/models/dsv41" for s, d in _mt), str(_mt))
    check(f"{_pair}'s P7 spec does not mount the /models directory (root disk)",
          not any(d == "/models" for s, d in _mt), str(_mt))
check("prepare gates on the pack's disk before it cuts",
      "pair_pack_disk.py" in src("pair_prepare.sh")
      and src("pair_prepare.sh").index("pair_pack_disk.py") < src("pair_prepare.sh").index("pair_cut.sh"),
      "")
check("prepare refreshes the pair manifest when it rebuilds the release the pair points at",
      "pair_manifest.py" in src("pair_prepare.sh")
      and "pair_current_sha" in src("pair_prepare.sh")
      and src("pair_prepare.sh").index("pair_manifest.py")
      > src("pair_prepare.sh").index("pair_build_ext.sh"), "")
check("...and only for the current release, never for a candidate prepared beside a serving pair",
      'if [ "$cur" = "$SHA" ]' in src("pair_prepare.sh")
      and "PREPARE_MANIFEST pair=$PAIR skipped" in src("pair_prepare.sh"), "")
check("...and a manifest write that fails stops prepare (phase=manifest), it is not warn-only",
      "PREPARE_FAILED $PAIR phase=manifest" in src("pair_prepare.sh"), "")
check("the smoke gates on the pack's disk too",
      "pair_pack_disk.py" in src("pair_smoke.sh") and 'fail "pack_disk' in src("pair_smoke.sh"), "")
check("the pack check is a tested file",
      os.path.exists(os.path.join(HERE, "tests", "pack_disk.py")), "")
_pd = src("pair_pack_disk.py")
check("the pack check names the P7 drive as the allowed root",
      'PAIR_PACK_P7_ROOT", "/models/dsv41"' in _pd, "")
check("the pack check refuses /mnt and /net on the GX side",
      '("/mnt", "/net")' in _pd, "")
check("the pack check refuses a network filesystem",
      '"nfs"' in _pd and "remote_fs(" in _pd, "")
check("the smoke skips the pack probe when the harness has no filesystem",
      "PAIR_PACK_DISK=0" in src("tests/smoke_memory.sh") and "PAIR_PACK_DISK=0" in src("tests/memory_fail.sh"), "")

check("the GX pack judged is the one behind /cold now, not the spec's (a pack switch must not read as drift)",
      'pc.live(cfg, "gx")' in _pd and 'get("/cold"' in _pd
      and "PACK_DISK_CONFIG gx /cold spec=" in _pd, "")
check("...and a switched pack is labelled with both paths, so a run cannot be quoted under the wrong pack",
      "pack switch: judging live" in _pd, "")
check("the GX row records the placement_sha256 the live pack declares",
      "placement_sha=" in _pd and "def cold_placement_sha" in _pd, "")
check("the two halves must declare the same placement_sha256 (one checkpoint, two packs)",
      "def p7_placement_sha" in _pd and 'w == "placement"' not in _pd
      and '("gx", "placement", src' in _pd, "")
check("pack_disk.py is tested against a switched /cold and against a half-mismatched sha",
      "live=" in src("tests/pack_disk.py") and "placement_sha256 are refused" in src("tests/pack_disk.py"), "")

# A spec change cannot reach a running pair any other way: the flip-only ticket refuses a RECREATE
# plan unless the operator queued it with --recreate, and then the ticket itself re-creates from the
# spec inside the queue (old container renamed, never removed). Without this branch the mount could
# only be dropped by hand, which is exactly what the owner forbade.
_pu = src("pair_update.sh")
check("--recreate is the only thing that permits a recreate",
      _pu.count("PAIR_ALLOW_RECREATE=1") == 1 and "--recreate) FLIP_ONLY=1; RECREATE=1" in _pu, "")
check("the flip-only ticket refuses a spec change it was not asked to apply",
      'PAIR_ALLOW_RECREATE:-0' in ur and "--recreate" in ur, "")
check("the flip-only ticket re-creates through the containers tool",
      ur.count("pair_containers.py\" apply") >= 2 and "UPDATE_CONTAINERS pair=$PAIR recreated" in ur, "")
check("promote can carry the same permission",
      "--recreate) RECREATE=1" in src("pair_promote.sh") and 'cmd+=(--recreate)' in src("pair_promote.sh"), "")


# The gate is binding for every launch, not only for the smoke that takes it. seam_server_ensure.sh is
# the one program that starts a GX server, so the check lives there: on a missing, expired, stale or
# foreign claim it refuses with rc 3 and names the state, and it runs before anything on the box is
# touched. The dispatch must pass that status through -- the old `|| exit 2` turned every refusal into
# "server_timeout_or_died", which is how a foreign pack load hid for a whole 900 s timeout.
_ens = src("seam_server_ensure.sh")
_ensure_body = _ens[_ens.index("seam_server_ensure() {"):]
check("the ensure checks the gate before it touches the box",
      "seam_gate_check" in _ensure_body
      and _ensure_body.index("seam_gate_check") < _ensure_body.index("seam_pids"),
      "")
check("a missing or foreign claim is refused by name, with the reason",
      all(x in _ens for x in ("gate_free owner=none", "gate_expired owner=", "gate_stale owner=",
                              "gate_foreign owner=", "PREFLIGHT_FAIL gate_bad")), "")
_disp = _ens[_ens.rindex('if [ "${BASH_SOURCE[0]}" = "$0" ]'):]
check("the ensure exits with its real status",
      'seam_server_ensure "$@"' in _disp and "exit $?" in _disp and "|| exit 2" not in _disp,
      " | ".join(_disp.splitlines()))
check("only the ensure can start a server: it marks the launch SEAM_GATE_OK=1",
      'local denv="-e SEAM_GATE_OK=1 ' in _ens, "")
check("the ensure only checks the claim, it never takes or drops it",
      not re.search(r"^\s*pair_gx_gate_(take|down)\b", _ens, re.M), "")
check("the dead-holder rule lives in pair_lib too",
      "kill -0 \"$p\"" in src("pair_lib.sh"), "")
check("the smoke hands the ensure the gate directory it actually claims in",
      'SEAM_GATE_DIR="$(dirname "$(pair_gx_gate_file)")"' in src("pair_smoke.sh"), "")
check("a busy box is classified apart from a memory-tight one",
      '*gx-busy*) fail "gx_busy server_preflight $line"' in src("pair_smoke.sh")
      and "*gx-mem*)  fail \"memory server_preflight" in src("pair_smoke.sh"), "")

# Preflight order: the other container's server explains the missing memory, so it is reported first,
# and both refusals name the holder. On 2026-11-02 the memory line fired first and read
# "avail=0GiB need=115GiB" about a box whose pack (and port) belonged to seam-gx2-4460, pid 1083950.
_pf = _ens[_ens.index("seam_preflight() {"):_ens.index("seam_server_ensure() {")]
check("a foreign server is named with its container and pid",
      'echo "server_in=\\$c:\\$p"' in _pf, "")
check("the busy-box refusal comes before the memory refusal",
      _pf.index("gx-busy") < _pf.index("gx-mem avail="), "")
check("the memory refusal names the holder and the heaviest container",
      "held_by=${other:-none} top_mem=$top" in _pf and "memory.current" in _pf, "")

# Plan arrows: `mount:-` is a removal, `~` a wrong source, `+` a wanted mount that is absent. The
# removal used to print as `mount:+<dst>=<src>`, i.e. exactly like an add, on the very line that
# documents dropping the 12 TB mount from the P7 spec.
_cfg = pc.load_conf("dev")
_want = pc.spec(_cfg, "p7")
_got = {"running": True, "image": _want["image"], "workdir": _want["workdir"],
        "entrypoint": [_want["entrypoint"]], "cmd": list(_want["cmd"]),
        "mounts": {d: s for s, d in _want["mounts"]}, "env": dict(_want["env"]),
        "network": "host", "ipc": "host", "shm": 64 * 1024 ** 3, "security": ["label=disable"],
        "ulimits": {"memlock": -1}, "devices": ["/dev/infiniband"],
        "gpus": pc.spec_gpus(_want), "restart": "no"}
pc.live = lambda cfg, side: _got
_base = _REAL_PLAN(_cfg, "p7") or []
check("an unchanged container plans nothing (baseline for the mount arrows)",
      not [w for w in _base if w.startswith("mount")], str(_base))
# GPU scope is part of the plan: the P7 containers ran with every GPU on the host (three, one of
# them another owner's A4000) and every prepare read them as matched, because the check only asked
# whether a device request existed. A container that can see the neighbour's card is drift.
_got["gpus"] = "all"
_gd = " ".join(_REAL_PLAN(_cfg, "p7") or [])
check("a container that sees more GPUs than its spec is planned as drift",
      "gpus=all want=device=1" in _gd and bool(pc.real_reasons(_REAL_PLAN(_cfg, "p7") or [])), _gd)
_got["gpus"] = pc.spec_gpus(_want)
_got["mounts"]["/mnt/network/models"] = "/mnt/network/models"
_why = " ".join(_REAL_PLAN(_cfg, "p7") or [])
check("an extra live mount is planned as a removal, not an add",
      "mount:-/mnt/network/models=/mnt/network/models (not in spec)" in _why
      and "mount:+/mnt/network" not in _why, _why)
del _got["mounts"]["/mnt/network/models"]
# The difference between "recreate this container" and "note this and keep running" is the
# difference between a refused prepare and a release that ships; dev's prepare died on it on
# 2026-11-02 because both live P7 containers mount /models where the spec names the pack directory.
check("a removable extra mount is an advisory, not a verdict",
      "advisory:mount:-/mnt/network/models=/mnt/network/models (not in spec)" in _why
      and not pc.real_reasons(["advisory:mount:-/x=/x (not in spec)"])
      and pc.real_reasons(["mount:+/x=absent want=/x"]) == ["mount:+/x=absent want=/x"], _why)
del pc.live
_got2 = dict(_got)
_got2["mounts"] = dict(_got["mounts"], **{"/models": "/models"})
_got2["mounts"].pop("/models/dsv41", None)
pc.live = lambda cfg, side: _got2
_cov = " ".join(_REAL_PLAN(_cfg, "p7") or [])
check("a parent mount that carries the wanted path is covered, not absent",
      "mount:+/models/dsv41=absent" not in _cov and "covered_by /models=/models" in _cov
      and not [w for w in pc.real_reasons(_REAL_PLAN(_cfg, "p7") or []) if w.startswith("mount")], _cov)
del pc.live
pc.live = lambda cfg, side: _got
_d0 = sorted(_got["mounts"])[0]
_s0 = _got["mounts"][_d0]
_got["mounts"][_d0] = "/wrong/place"
_why = " ".join(_REAL_PLAN(_cfg, "p7") or [])
check("a mount from the wrong source is planned as ~", f"mount:~{_d0}=" in _why, _why)
_got["mounts"].pop(_d0)
_why = " ".join(_REAL_PLAN(_cfg, "p7") or [])
check("a wanted mount that is absent is planned as + … absent",
      f"mount:+{_d0}=absent want={_s0}" in _why, _why)
del pc.live

# A spec that names /mnt is refused by the loader (plan, apply, smoke) and reported, not aborted, by the
# pack check: an operator who pastes the 12 TB mount back must get a refusal that says which mount and
# why, not a traceback.
import tempfile as _tf
_W = _tf.mkdtemp(prefix="specmnt.")
_old_spec = pc.SPEC_DIR
try:
    pc.SPEC_DIR = _W
    _bad = {"name": "seam-p7-dev", "image": "img", "workdir": "/src", "entrypoint": "/bin/bash",
            "mounts": [["/mnt/network/models", "/mnt/network/models"]], "env": {}, "args": [],
            "cmd": []}
    _json.dump(_bad, open(os.path.join(_W, "dev-p7.json"), "w"))
    _m = die_msg(pc.spec, pc.load_conf("dev"), "p7")
    check("a P7 spec that mounts /mnt is refused by the loader",
          "SPEC_REFUSED" in _m and "own disk" in _m, _m)
    _bad["name"] = "seam-gx2-dev"
    _json.dump(_bad, open(os.path.join(_W, "dev-gx.json"), "w"))
    _m = die_msg(pc.spec, pc.load_conf("dev"), "gx")
    check("a GX spec that mounts /mnt is refused by the loader", "SPEC_REFUSED" in _m, _m)
finally:
    pc.SPEC_DIR = _old_spec
    import shutil as _sh
    _sh.rmtree(_W, ignore_errors=True)
check("the pack check reads the spec's mounts raw so a bad mount is a report, not an abort",
      "spec_mounts(" in src("pair_pack_disk.py") and "pc.spec(cfg, \"p7\")" not in src("pair_pack_disk.py")
      and "pc.spec(cfg, \"gx\")" not in src("pair_pack_disk.py"), "")
lib = src("pair_lib.sh")
check("a box claim outside a queue ticket is refused, not taken",
      "PAIR_GATE_REFUSED outside_ticket" in lib and "pair_ticket_hardcap" in src("pair_lib.sh"),
      "")
check("a claim never runs past the ticket's hard cap",
      "PAIR_GATE_REFUSED ticket_hardcap_passed" in lib and "PAIR_GATE_CLAMP" in lib, "")
# Both halves run for real: the same claim that a ticket may take is refused from a shell with no
# ticket in its ancestry, and inside one it is clamped to the cap the wrapper exported.
import subprocess as _sp, tempfile as _tf, shutil as _sh
_gd = _tf.mkdtemp(prefix="gategate")
_env = dict(os.environ, PAIR_GATE_DIR=_gd, PAIR_QUEUE_ROOT="/tmp/sweepq", PAIR_QUEUE="/tmp/sweepq",
            GXCT="seam-gx2-dev")
for _k in ("GPU_RUN_STATUS", "GPU_TICKET_HARDCAP"):
    _env.pop(_k, None)
_probe = ('. "$(dirname "$0")/pair_lib.sh"; printf "%s\n" "claim_outside_$?"; exit 0')
_r = _sp.run(["bash", "-c", '. "%s/pair_lib.sh"; pair_gx_gate_up outsider 30; echo "rc=$?"; '
              'ls "%s" 2>/dev/null' % (HERE, _gd)],
             env=_env, capture_output=True, text=True)
check("gate_up refuses outside a ticket and writes no claim",
      "rc=5" in _r.stdout and ".gate" not in _r.stdout and "PAIR_GATE_REFUSED outside_ticket" in _r.stderr,
      (_r.stdout + _r.stderr).strip()[:200])
_env["GPU_TICKET_HARDCAP"] = str(1)
_r = _sp.run(["bash", "-c", '. "%s/pair_lib.sh"; pair_gx_gate_up pastcap 30; echo "rc=$?"' % HERE],
             env=_env, capture_output=True, text=True)
check("an expired ticket's cap refuses the claim",
      "rc=5" in _r.stdout and "ticket_hardcap_passed" in _r.stderr, (_r.stdout + _r.stderr).strip()[:200])
_env["GPU_TICKET_HARDCAP"] = str(int(__import__("time").time()) + 600)
_r = _sp.run(["bash", "-c", '. "%s/pair_lib.sh"; pair_gx_gate_up clamps 120; echo "rc=$?"; '
              'cat "%s"/*.gate 2>/dev/null' % (HERE, _gd)],
             env=_env, capture_output=True, text=True)
_until = re.search(r"until=(\d+)", _r.stdout or "")
check("a claim longer than the ticket is clamped to its cap",
      "rc=0" in _r.stdout and _until and int(_until.group(1)) <= int(_env["GPU_TICKET_HARDCAP"]),
      (_r.stdout + _r.stderr).strip()[:200])

# A reader must never guess which box it is reading. With GXCT unset the path used to be
# queue/gx.gate -- a file that never exists -- so a claimed box answered "free", and on 2026-11-02
# 03:1x a hand-run check reported GX2 free while a smoke held it. An unresolvable gate is an error
# that reaches the caller as a refusal (rc 5), never as "free" and never as a wait.
for _label, _gx in (("unset", None), ("unparsable", "bogus")):
    _e = dict(_env)
    _e["GPU_TICKET_HARDCAP"] = str(int(__import__("time").time()) + 600)
    _e.pop("GXCT", None)
    if _gx:
        _e["GXCT"] = _gx
    _r = _sp.run(["bash", "-c", '. "%s/pair_lib.sh"; pair_gx_gate_file 2>/dev/null >/dev/null; '
                  'echo "file_rc=$?"; echo "state=$(pair_gx_gate_state)"; '
                  'pair_gx_gate_up nobody 5 >/dev/null 2>&1; echo "up_rc=$?"' % HERE],
                 env=_e, capture_output=True, text=True)
    check("an unresolvable GXCT (%s) reads unresolved, never free, and refuses a claim" % _label,
          "file_rc=2" in _r.stdout and "state=gate_unresolved" in _r.stdout
          and "up_rc=5" in _r.stdout, (_r.stdout + _r.stderr).strip()[:200])
# ...and a resolvable one still names its own box, so the guard is not a blanket refusal. The claim
# is taken by the probing shell itself, so its pid is alive for the read that follows.
_e = dict(_env); _e["GPU_TICKET_HARDCAP"] = str(int(__import__("time").time()) + 600)
_e["GXCT"] = "seam-gx2-dev"
_r = _sp.run(["bash", "-c", '. "%s/pair_lib.sh"; pair_gx_gate_up keeper 5 >/dev/null 2>&1; '
              'echo "file=$(pair_gx_gate_file)"; echo "state=$(pair_gx_gate_state)"' % HERE],
             env=_e, capture_output=True, text=True)
check("a resolvable GXCT still reads its own box's claim",
      "file=%s/gx2.gate" % _gd in _r.stdout and "state=keeper until=" in _r.stdout,
      (_r.stdout + _r.stderr).strip()[:200])

# A server may only be stopped inside the claim that protects the box. At 07:53 on 2026-11-02 a
# foreign prologue TERMed dev's worker while GxPort2 held gx2, could not take the gate afterwards,
# and the pair served nothing until the next ticket. The guard is exercised in both directions:
# a claim held by somebody else must leave the worker alone, our own claim must stop it as before.
_sd = _tf.mkdtemp(prefix="stopgate")
_e = dict(_env); _e["GXCT"] = "seam-gx2-dev"; _e["PAIR"] = "dev"
_e["PAIR_STATE_DIR"] = _sd; _e["GPU_TICKET_HARDCAP"] = str(int(__import__("time").time()) + 600)
open(_sd + "/server-dev.pid", "w").write("pair=dev build=dev-deadbeef pid=999999 at=1970-01-01T00:00:00Z\n")
_stop_probe = (
    '. "%s/pair_lib.sh"; PAIR_DRY=0; '
    'pair_gxsh() { case "$*" in *cmdline*) echo "python3 -u qmesh/worker_seam_server.py --x";; '
    '*kill*) echo sent;; esac; }; '
    'pair_server_stop; echo "rc=$?"; test -f "%s/server-dev.pid" && echo pidfile_kept' % (HERE, _sd))
_holder = _sp.Popen(["sleep", "120"])
try:
    open(_gd + "/gx2.gate", "w").write("owner=GxPort2 pid=%d until=%d host=%s\n"
                                       % (_holder.pid, int(__import__("time").time()) + 600,
                                          _sp.run(["hostname", "-s"], capture_output=True,
                                                  text=True).stdout.strip()))
    _r = _sp.run(["bash", "-c", _stop_probe], env=_e, capture_output=True, text=True)
    check("a stop refuses while a live foreign claim owns the box",
          "SERVER_STOP_REFUSED" in _r.stdout and "rc=1" in _r.stdout
          and "pidfile_kept" in _r.stdout and "SERVER_STOP pid=" not in _r.stdout,
          (_r.stdout + _r.stderr).strip()[:200])
    open(_gd + "/gx2.gate", "w").write("owner=GxPort2 pid=%d until=%d host=elsewhere\n"
                                       % (_holder.pid, int(__import__("time").time()) + 600))
    _r = _sp.run(["bash", "-c", _stop_probe], env=_e, capture_output=True, text=True)
    check("a claim written on another host can never be read as ours",
          "SERVER_STOP_REFUSED" in _r.stdout and "rc=1" in _r.stdout,
          (_r.stdout + _r.stderr).strip()[:200])
finally:
    _holder.terminate(); _holder.wait()
os.unlink(_gd + "/gx2.gate")  # a remote-host claim stays claimed until it expires; clear it so the
                              # positive case tests our own claim rather than the previous file
_r = _sp.run(["bash", "-c", '. "%s/pair_lib.sh"; PAIR_DRY=0; '
              'pair_gxsh() { case "$*" in *cmdline*) echo "python3 -u qmesh/worker_seam_server.py --x";; '
              '*kill*) echo sent;; esac; }; '
              'pair_gx_gate_up mine 5 >/dev/null 2>&1; '
              'pair_server_stop; echo "rc=$?"' % HERE], env=_e, capture_output=True, text=True)
check("our own claim still stops the worker, and clears the pidfile",
      "SERVER_STOP pid=999999" in _r.stdout and "rc=0" in _r.stdout
      and not os.path.exists(_sd + "/server-dev.pid"), (_r.stdout + _r.stderr).strip()[:200])
_sh.rmtree(_sd, ignore_errors=True)
if os.path.exists(_gd + "/gx2.gate"):
    os.unlink(_gd + "/gx2.gate")
check("the wrapper hands its hard cap to the ticket's children",
      "export GPU_TICKET_HARDCAP=$hard" in src("gpu_run_pair.sh")
      and "hardcap=%s" in src("gpu_run_pair.sh"), "")
check("the release guard only holds tickets that name a pair container",
      'check_release "$@"' in src("gpu_run_pair.sh")
      and 'PAIR_CONTAINERS="seam-p7-dev seam-p7-prod seam-gx2-dev seam-gx1-prod"' in src("gpu_run_pair.sh")
      and "PAIR_CHECK_FORCE" in src("gpu_run_pair.sh"), "")
# The gate probes above set GPU_TICKET_HARDCAP to stand in for a wrapper; a wrapper actually run here is
# a fresh top-level ticket and must not look like one queued from inside a ticket.
def _nocap(e):
    e = dict(e); e.pop("GPU_TICKET_HARDCAP", None); return e
# Executed, not grepped: a ticket whose command is one quoted string must exit 2 before the GPU, because
# `setsid "$@"` would otherwise exec a filename with spaces and die as rc=127 in 0 s -- which read as a
# failed build twice on 2026-11-02 and cost a box window each time.
# A restore ticket is queued as a safety net behind an evicting one, so it usually finds the pair already
# serving the release it was going to serve. Reloading there costs 116 GiB and evicts the next ticket's
# measurement, so "already up on this build" has to be a success, not PAIR_SERVE_ALREADY. The build id is
# read from the running worker's environment -- neither the pidfile nor `current` says what is loaded.
_rb = _sp.run(["bash", "-c", '. "%s/pair_lib.sh"; GXCT=seam-gx2-dev; PAIR_DRY=0; '
               'pair_gxsh() { echo "dev-deadbeef"; }; echo "b=$(pair_gx_serving_build)"' % HERE],
              env=_nocap(_env), capture_output=True, text=True)
check("the serving-build reader returns the worker's own SEAM_BUILD_ID",
      "b=dev-deadbeef" in _rb.stdout and "SEAM_BUILD_ID=" not in _rb.stdout,
      (_rb.stdout + _rb.stderr).strip()[:200])
_ps = src("pair_serve.sh")
check("a restore that finds its own build already serving is a no-op, not a reload",
      "PAIR_SERVE_ALREADY_UP" in _ps and '[ "$serving" = "$BUILD_ID" ]' in _ps
      and _ps.index("PAIR_SERVE_ALREADY_UP") < _ps.index("pair_gx_gate_take"),
      "")
_r = _sp.run(["bash", "%s/gpu_run_pair.sh" % HERE, "T", "1", "scripts/pair_prepare.sh dev x --side gx"],
             env=_nocap(dict(_e, QBASE=os.path.join(_gd, "q"))),
             capture_output=True, text=True, timeout=60)
check("a ticket whose command is one quoted string is refused before the GPU",
      _r.returncode == 2 and "TICKET_BAD_CMD argv0=" in (_r.stdout + _r.stderr),
      "rc=%s %s" % (_r.returncode, (_r.stdout + _r.stderr).strip()[:200]))
# The guard resolves argv[0] the way the shell does, so a ticket that starts with a PATH command is
# accepted: `bash <script>` is a real form (GxPort2 documents `queue/run_dev.sh <Ag> 45 bash $0`).
_tmp = os.path.join(_gd, "t.sh")
with open(_tmp, "w") as _fh:
    _fh.write("#!/bin/sh\nexit 0\n")
_r = _sp.run(["bash", "%s/gpu_run_pair.sh" % HERE, "T", "1", "bash", _tmp],
             env=_nocap(dict(_e, QBASE=os.path.join(_gd, "q"))), capture_output=True, text=True, timeout=60)
check("a ticket that starts with a PATH command (bash <script>) is accepted",
      _r.returncode != 2 and "TICKET_BAD_CMD" not in (_r.stdout + _r.stderr),
      "rc=%s %s" % (_r.returncode, (_r.stdout + _r.stderr).strip()[:200]))
# The lock must not be held by anything but the wrapper's own head check. A ticket command inherits
# the wrapper's fds, so a ticket that slept (or an ssh under the release guard) while fd 9 was open
# kept the flock's open description alive after its wrapper died: at 09:03 on 2026-11-02 the queue
# stayed write-locked with no owner and three agents waited in `flock 9` until they were killed by
# hand. Executed, not grepped: start a ticket that tries to take the lock from inside its own body,
# and require it to succeed.
_qb = os.path.join(_gd, "q2")
os.makedirs(os.path.join(_qb, "gpu_queue"), exist_ok=True)
_bin = os.path.join(_gd, "bin"); os.makedirs(_bin, exist_ok=True)
with open(os.path.join(_bin, "nvidia-smi"), "w") as _fh:
    _fh.write("#!/bin/sh\nexit 0\n")   # empty compute-apps list: the box reads free
os.chmod(os.path.join(_bin, "nvidia-smi"), 0o755)
_lk = os.path.join(_gd, "lockprobe.sh")
with open(_lk, "w") as _fh:
    _fh.write("#!/bin/sh\nflock -n %s -c 'echo lock_free' || echo lock_held\n" % os.path.join(_qb, "gpu.lock"))
os.chmod(_lk, 0o755)
_r = _sp.run(["bash", "%s/gpu_run_pair.sh" % HERE, "LockProbe", "1", "sh", _lk],
             env=_nocap(dict(_e, QBASE=_qb, QGPUS="0", PATH=_bin + ":" + os.environ["PATH"])),
             capture_output=True, text=True, timeout=120)
check("a ticket command can take the queue lock its wrapper released",
      "lock_free" in (_r.stdout + _r.stderr), (_r.stdout + _r.stderr).strip()[:200])
_wrap = src("gpu_run_pair.sh")
_locked = _wrap.split('exec 9>"$LOCK"')[1].split("flock -u 9")[0]
check("the wrapper closes the lock descriptor before it starts a ticket",
      "exec 9>&-" in _wrap
      and _wrap.index('check_release "$@"') < _wrap.index('exec 9>"$LOCK"')
      and "check_release" not in _locked, _locked.strip()[:200])
# A ticket body that queues a ticket deadlocks the FIFO: the outer entry stays head while its own
# command waits for the inner one, and nothing behind either can run. On 2026-11-02 10:23-10:32 four
# waiters sat behind exactly that (an IDTrace ticket whose body called run_dev.sh) and the cure applied
# on the spot -- deleting a finished ticket file -- left the cause in place. GPU_TICKET_HARDCAP is
# exported into the ticket's children and nowhere else, so the wrapper can tell it is inside one.
_wrap = src("gpu_run_pair.sh")
check("a ticket cannot queue another ticket: the guard runs before any entry is written",
      "TICKET_NESTED" in _wrap
      and _wrap.index("TICKET_NESTED") < _wrap.index('ticket="$Q/'), "")
_qn = os.path.join(_gd, "q3")
_r = _sp.run(["bash", "%s/gpu_run_pair.sh" % HERE, "Nested", "1", "sh", _tmp],
             env=dict(_e, QBASE=_qn, QGPUS="0",
                      GPU_TICKET_HARDCAP=str(int(__import__("time").time()) + 600)),
             capture_output=True, text=True, timeout=60)
_qn_dir = os.path.join(_qn, "gpu_queue")
_qn_left = [f for f in os.listdir(_qn_dir) if not f.startswith("status-")] if os.path.isdir(_qn_dir) else []
check("a ticket queued from inside a ticket is refused and leaves no entry behind",
      _r.returncode == 2 and "TICKET_NESTED" in (_r.stdout + _r.stderr) and not _qn_left,
      "rc=%s left=%s %s" % (_r.returncode, _qn_left, (_r.stdout + _r.stderr).strip()[:160]))

# Rank orders the WAITERS; it must never preempt a running ticket. The head rule alone made the queue
# unsafe: a later rank-01 entry sorts below the running ticket's name and becomes head, and a ticket
# between phases keeps no P7 process, so gpu_busy() is false. Measured 2026-11-02: PairUpdate 11:20:43
# and CanaryB84 11:22:07 started beside G6Bound, SeamRobust's headroom 11:24:09 beside CanaryB84.
# This runs the wrapper for real (stub nvidia-smi, no ssh) around the exact shape: rank-99 ticket
# running a `sleep` with nothing on the GPU, then a rank-01 ticket queued behind it.
_qd = _tf.mkdtemp(prefix="queueserial")
_qbin = os.path.join(_qd, "bin"); os.makedirs(_qbin)
with open(os.path.join(_qbin, "nvidia-smi"), "w") as _f:
    _f.write("#!/bin/sh\nexit 0\n")
os.chmod(os.path.join(_qbin, "nvidia-smi"), 0o755)
_qbase = os.path.join(_qd, "q"); os.makedirs(os.path.join(_qbase, "gpu_queue"))
with open(os.path.join(_qbase, "gpu_priority.txt"), "w") as _f:
    _f.write("High\nLow\n")            # High = rank 01, so its ticket name sorts below Low's
_qenv = dict(os.environ, QBASE=_qbase, QGPUS="0", PAIR_CHECK_SKIP="1",
             PATH=_qbin + ":" + os.environ.get("PATH", ""))
for _k in ("GPU_TICKET_HARDCAP", "GPU_RUN_STATUS", "PAIR_CHECK_FORCE"):
    _qenv.pop(_k, None)
_qlog = os.path.join(_qbase, "gpu_queue.log")
_mark = os.path.join(_qd, "high-ran")
_low = _sp.Popen(["bash", os.path.join(HERE, "gpu_run_pair.sh"), "Low", "1", "sh", "-c", "sleep 20"],
                 env=_qenv, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
_t0 = __import__("time").time()
while __import__("time").time() - _t0 < 20:
    if os.path.exists(_qlog) and "start Low" in open(_qlog).read():
        break
    __import__("time").sleep(0.2)
_runfile = os.path.join(_qbase, "running")
_marker = open(_runfile).read() if os.path.exists(_runfile) else ""
check("a started ticket names itself in $BASE/running", "agent=Low pid=" in _marker, _marker.strip())
_high = _sp.Popen(["bash", os.path.join(HERE, "gpu_run_pair.sh"), "High", "1",
                   "sh", "-c", "echo ran > %s" % _mark],
                  env=_qenv, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
__import__("time").sleep(8)
_log = open(_qlog).read() if os.path.exists(_qlog) else ""
check("a rank-01 ticket does not start beside a live rank-99 ticket with an idle GPU",
      not os.path.exists(_mark) and "wait High head=rank1 running=agent=Low" in _log,
      "ran=%s log=%s" % (os.path.exists(_mark), _log.strip().splitlines()[-3:]))
_low.wait(timeout=60)
try:
    _high.wait(timeout=90)
except _sp.TimeoutExpired:
    _high.kill()
check("...and it starts as soon as that ticket's command exits",
      os.path.exists(_mark) and not os.path.exists(_runfile),
      "ran=%s running_file=%s" % (os.path.exists(_mark), os.path.exists(_runfile)))
check("a running marker is never left behind",
      not os.path.exists(_runfile) and not (os.path.exists(_runfile + ".tmp")), "")
_sh.rmtree(_qd, ignore_errors=True)

_sh.rmtree(_gd, ignore_errors=True)
check("the hook build finds a toolkit that carries cusparse.h",
      _pl.count("CPLUS_INCLUDE_PATH=$incdir") == 1
      and _pl.index("CPP_TOOLKIT") < _pl.rindex("CPP_BUILD")
      and "no_cusparse_h" in _pl
      and "$incenv bash build_moe.sh" in _pl, "")

check("the installer refuses a dst that is its own source tree",
      'dst_is_src' in _pi and _pi.index("dst_is_src") < _pi.rindex("INSTALL_OK"), "")
# ...and it does not have to reach that refusal to be safe: the default resolution runs from a checkout
# with "$HERE" = the source tree, which is how CanaryB84 installed over itself, printed INSTALL_OK
# dst=<repo>/qmesh/release and then CMP_FAIL on five files while artifacts/scripts stayed stale
# (2026-11-03 02:35Z). The source is canonicalised first and skipped as a candidate.
check("the installer never resolves its destination to its own source tree",
      _pi.index('SRC=$(cd "$SRC" && pwd)') < _pi.index("for cand in")
      and '[ "$(cd "$cand" && pwd)" = "$SRC" ] && continue' in _pi, "")
# Behavioural, and independent of the queue: the refusal comes before the ticket scan, so this runs
# wherever the sweep does. A dst that is the src must exit 2 and must not have replaced anything.
_wt = _tf.mkdtemp(prefix="pairsweep-inst.")
try:
    os.makedirs(f"{_wt}/rel"); os.makedirs(f"{_wt}/pairstate")
    open(f"{_wt}/rel/pair_lib.sh", "w").write("x\n")
    open(f"{_wt}/rel/pair_arm.sh", "w").write("#!/bin/bash\necho old\n")
    before = open(f"{_wt}/rel/pair_arm.sh").read()
    r = subprocess.run(["bash", os.path.join(HERE, "pair_install.sh"), "--src", f"{_wt}/rel",
                        "--dst", f"{_wt}/rel", "pair_arm.sh"], capture_output=True, text=True)
    check("...and a dst equal to the src is refused before anything is written",
          r.returncode == 2 and "INSTALL_FAIL dst_is_src" in r.stderr
          and open(f"{_wt}/rel/pair_arm.sh").read() == before
          and "INSTALL installed pair_arm.sh" not in r.stdout,
          "rc=%d err=%s" % (r.returncode, r.stderr.strip()[:160]))
finally:
    _sh.rmtree(_wt, ignore_errors=True)

# SIDE_SCRIPT is %-formatted as a whole, so a bare printf verb inside it is a format error, not a
# print: `printf '%s->%s '` in the EXTSRC line took every pair_check down with "not enough arguments
# for format string" (2026-11-03, PAIR_DRIFT on both halves). Render it here with a placeholder for
# every key the script names, so an unescaped verb fails the sweep instead of the pair.
_sc = open(os.path.join(HERE, "pair_sync_check.py")).read()
_lit = re.search(r'SIDE_SCRIPT = r"""(.*?)"""', _sc, re.S)
check("pair_sync_check's side script formats",
      _lit is not None
      and len(_lit.group(1) % {k: "x" for k in re.findall(r"%\((\w+)\)", _lit.group(1))}) > 0
      and "%%s->%%s" in _lit.group(1), "")

# The worker's argv differs by launch path (ensure starts it relative, the serve path absolute), so an
# argv-anchored match saw a resident pack as absent and the smoke loaded a second one beside it
# (2026-11-03 box_busy with 1005 MiB free). One selection by comm, used by the count, the box scan and
# every kill; and no `pkill -f` on a name a wrapper also carries.
_ens = open(os.path.join(HERE, "seam_server_ensure.sh")).read()
check("the worker is selected by comm, not by an argv prefix",
      _ens.count("pgrep -x python3 -a") >= 2
      and "worker_seam_server[.]py" in _ens
      and "pkill -f" not in _ens
      and 'kill ${1:+$1 }' in _ens, "")

# The ids shim forwards attribute writes to the wrapped generator; with two arguments it raised
# "setattr expected 3 arguments, got 2" right after ARM_START and every ids arm died (2026-11-03, both
# census shapes). It lived only in scripts/ -- versioned here so the installer can carry it.
_gap = open(os.path.join(HERE, "pair_arm_ids_gap.py")).read()
check("the ids gap shim forwards (obj, name, value)",
      'setattr(object.__getattribute__(self, "_g"), name, value)' in _gap
      and 'getattr(object.__getattribute__(self, "_g"), name)' in _gap, "")

# ---- the client half is named by the caller; the pack is only its cross-check ----------------------
# The dev pair's cold pack is switched by moving the GX container's /cold mount (4460 -> 4780p8,
# 2026-11-03), so the pack still decides WHICH half pairs -- but a lookup cannot name it: on 2026-11-05
# two exl3-p7hot-* dirs declared one placement sha, `auto` refused as ambiguous, and three callers that
# had no name of their own died. Every consumer names the half; pair_ckpt_resolve auto is printed beside
# it (PAIR_CKPT_XCHECK) and never decides.
_lib = src("pair_lib.sh")
check("pair_lib joins the two halves on the pack's placement sha, not on a directory name",
      'pair_cold_placement()' in _lib and 'placement_sha256' in _lib
      and 'pair_p7_placements()' in _lib and 'pair_ckpt_auto()' in _lib
      and 'pair_ckpt_resolve()' in _lib, "")
check("pair_ckpt_resolve refuses rather than guessing when the halves do not pair",
      'matches=$n' in _lib and 'return 1' in _lib, "")
check("the cold half's sha is read off the mount source, not out of a running container",
      _lib.index('PACK_MANIFEST.json') < _lib.index('docker exec $gx'), "")
check("the hot halves resolve before the pair's client container exists",
      "pair_p7_pool()" in _lib and 'docker inspect -f' in _lib, "")
check("the long-running scripts run from a private copy of themselves",
      'PAIR_SMOKE_SELF' in src("pair_smoke.sh") and 'PAIR_UPD_SELF' in src("pair_update_run.sh")
      and 'exec bash "$PAIR_SELF_COPY"' in src("pair_smoke.sh")
      and 'exec bash "$PAIR_SELF_COPY"' in src("pair_update_run.sh")
      and 'PAIR_SELF_HERE' in src("pair_smoke.sh") and 'PAIR_SELF_HERE' in src("pair_update_run.sh"), "")
check("a half-written pack is refused before it is loaded",
      "pair_p7_verify()" in _lib and "hashlib.sha256" in _lib and 'state=mismatch' in _lib
      and 'is not the pack its manifest describes' in src("pair_arm.sh")
      and 'is not the pack its manifest describes' in src("pair_server.sh"), "")
check("the resolver starts nothing inside the pair's container unless the host path is gone",
      _lib.index('pool=$(pair_p7_pool "$p7")') < _lib.index('docker exec "$p7"'), "")
for _f in ("pair_arm.sh", "pair_server.sh", "pair_smoke.sh", "pair_pack_disk.py"):
    check("no code names the hot half by hand: " + _f,
          "exl3-p7hot-4460" not in "".join(code_lines(src(_f))), "")
# RM10 2026-11-05: two /models/dsv41/exl3-p7hot-* dirs declared the same placement sha, `auto` refused,
# and the three callers that trusted it voided their runs. The half is NAMED now -- explicit
# (--checkpoint/CHECKPOINT), else PAIR_PACK_P7, else AB_CKPT -- and `pair_ckpt_resolve auto` survives only
# inside pair_ckpt_xcheck as a logged comparison, so ambiguity is loud instead of fatal-by-accident.
check("pair_lib refuses an unnamed half and prints the ambiguity it cannot settle",
      "PAIR_CKPT_UNNAMED" in _lib and "PAIR_CKPT_AMBIGUOUS" in _lib
      and "pair_ckpt_require()" in _lib and "pair_ckpt_xcheck()" in _lib, "")
check("pair_arm names the client half after the pair is loaded, never resolves it",
      src("pair_arm.sh").index('pair_load "$PAIR_ARG"') < src("pair_arm.sh").index("pair_ckpt_require")
      and 'pair_ckpt_require "$CHECKPOINT"' in src("pair_arm.sh")
      and '$(pair_ckpt_resolve' not in src("pair_arm.sh"), "")
check("pair_server names the client half and dies when it cannot be named",
      'CHECKPOINT=$(pair_ckpt_require "$CHECKPOINT")' in src("pair_server.sh")
      and 'CHECKPOINT=$(pair_ckpt_resolve' not in src("pair_server.sh"), "")
check("the smoke exports the named half it is actually serving",
      'PAIR_PACK_P7=$(pair_ckpt_require' in src("pair_smoke.sh")
      and "export PAIR_PACK_P7" in src("pair_smoke.sh")
      and 'PAIR_PACK_P7=$(pair_ckpt_resolve' not in src("pair_smoke.sh"), "")
check("prepare names the client half too, so its pack gate can judge",
      'PAIR_PACK_P7=$(pair_ckpt_require' in src("pair_prepare.sh")
      and src("pair_prepare.sh").index("pair_ckpt_require") < src("pair_prepare.sh").index("pair_pack_disk.py")
      and 'PAIR_PACK_P7=$(pair_ckpt_resolve' not in src("pair_prepare.sh"), "")
for _f in ("pair_ref.sh", "pair_server_smoke.sh", "pair_update_run.sh"):
    check("the ticket names the client half instead of deriving it: " + _f,
          "pair_ckpt_require" in src(_f) and '$(pair_ckpt_resolve' not in src(_f), "")
check("an unnamed hot half is refused, not checked against the previous pack's refs",
      'PACK_DISK_USAGE' in src("pair_pack_disk.py")
      and 'PAIR_PACK_P7", "")' in src("pair_pack_disk.py"), "")
check("the ensure gate has its own battery",
      os.path.exists(os.path.join(HERE, "tests", "gx_gate_ensure.sh")), "")
# 2026-11-03: seam_gxct pastes its payload into `ssh … "docker exec $CT bash -lc '<payload>'"`, so a single
# quote inside the payload closed the wrapper's argument. `seam_kill_pattern` then reached bash as
# `for p in $(pgrep -x python3 -a | grep ` -- unterminated, no signal ever sent -- and the six `grep -aq
# '^SEAM_SERVER_*'` probes ran as a bare `grep` with no file. Both are silent: the stop "failed" after a 30 s
# poll and the worker kept the pack. The escape in seam_gxct is the fix; this keeps payloads quote-free anyway.
import re as _re
import glob as _glob
_gxct_bad = []
for _f in sorted(_glob.glob(os.path.join(HERE, "*.sh"))):
    for _n, _line in enumerate(open(_f), 1):
        for _arg in _re.findall(r'seam_gxct\s+"((?:[^"\\]|\\.)*)"', _line):
            if "'" in _arg:
                _gxct_bad.append("%s:%d %s" % (os.path.basename(_f), _n, _arg.strip()[:70]))
check("no seam_gxct payload carries a single quote",
      not _gxct_bad, "; ".join(_gxct_bad)[:300])
_ens = src("seam_server_ensure.sh")
check("seam_gxct escapes a quote rather than trusting its callers",
      "${1//\\'/\\'\\\\\\'\\'}" in _ens
      and 'bash -lc \'$q\'' in _ens, "")
# Ask the function what it emits rather than reading it: the printf format carries quotes of its own, the
# payload must not. Both callers (plain stop and the -9 retry) are checked.
_kp = _re.search(r'^seam_kill_pattern\(\) \{.*$', _ens, _re.M)
_kp_body = _kp.group(0) if _kp else ""
_kp_out = {}
for _sig in ("", "-9"):
    _r = subprocess.run(["bash", "-c", 'eval "$1"; seam_kill_pattern "$2"', "_", _kp_body, _sig],
                        capture_output=True, text=True)
    _kp_out[_sig] = _r.stdout
check("the kill list is quote-free and selects by /proc cmdline",
      all("grep -qa worker_seam_server /proc/" in v and "'" not in v for v in _kp_out.values())
      and 'kill $p; done' in _kp_out[""] and "kill -9 $p; done" in _kp_out["-9"],
      repr(_kp_out)[:220])
print("\nRESULT %s fails=%d" % ("PASS" if not fails else "FAIL", len(fails)))
sys.exit(1 if fails else 0)
