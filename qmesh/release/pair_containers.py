#!/usr/bin/env python3
"""pair_containers.py -- keep each pair's two containers on the release layout, by construction.

The containers never mount a tree that can go stale: they mount the pair root at /pair and use a
symlink inside the container, /src/exllamav3 -> /pair/current/src. Flipping the host symlinks
therefore changes what the *next* job runs without recreating anything, and a running job keeps the
release it started with.

TORCH_EXTENSIONS_DIR is deliberately **not** /pair/extcache by default (2026-11-04): a bare `import
exllamav3` inside one of these containers used to land in whichever release the extcache link named,
and torch's rebuild decision is textual in build.ninja, so an import from a *different* release's src
rewrote that release's compiled cache -- it voided two release A/B arms and a flip smoke. The spec
default is /pair/scratch-ext, which costs a stray import a private JIT and nothing else. The paths
that mean the release's cache state it explicitly: pair_arm.sh, pair_server.sh and pair_build_ext.sh
pass `TORCH_EXTENSIONS_DIR=$EXT_DIR`, pair_smoke.sh's ext probe passes /pair/extcache, and
/pair/extcache stays the release's cache for the GX worker and for anything the pair flips to.

    pair_containers.py plan <pair>              # live containers vs qmesh/release/containers/<pair>-<side>.json
    pair_containers.py apply <pair> <sha> [--ts T]   # recreate what plan says is stale; keep the old one as <name>.bak-<ts>
    pair_containers.py rollback <pair> <ts>     # undo one apply: put <name>.bak-<ts> back
    pair_containers.py link <pair> <sha>        # host symlinks: current -> releases/<sha>, extcache -> releases/<sha>/ext
    pair_containers.py inner <pair>             # inside the containers: /src/exllamav3 -> /pair/current/src
    pair_containers.py ls <pair>                # container state, for pair_status.sh

Exit: 0 ok / nothing to do, 1 plan says recreate (plan only), 2 error.
Old containers are renamed, never removed: rollback needs them and a removal is not reversible.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = os.path.join(HERE, "containers")
ART = os.environ.get("PAIR_ART", os.path.expanduser("~/recipe-artifacts/gpu0-gx1"))
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def die(msg):
    print(f"pair_containers: {msg}", file=sys.stderr)
    sys.exit(2)


def load_conf(pair):
    conf = os.environ.get("PAIR_CONF", os.path.join(HERE, "pairs.conf"))
    for line in open(conf):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if f[0] != pair:
            continue
        keys = ["pair", "p7ct", "gpu", "seam_dev", "seam_gid", "gx_gid", "seam_peer", "gx_host",
                "gxct", "root", "port", "buckets", "p7_arch", "gx_arch",
                "load_margin", "ws_rows"]
        d = dict(zip(keys, f))
        d["root"] = os.environ.get("PAIR_ROOT_OVERRIDE") or d["root"]
        return d
    die(f"pair {pair} not in {conf}")


def sh(cmd, stdin=None):
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def side_host(cfg, side):
    return None if side == "p7" else cfg["gx_host"]


def ssh_bin():
    """The dry-run harness passes PAIR_DRY_SSH=stub_ssh so no command leaves the box."""
    if os.environ.get("PAIR_DRY") == "1" and os.environ.get("PAIR_DRY_SSH"):
        return [os.environ["PAIR_DRY_SSH"]]
    return ["ssh"]


def docker(cfg, side, args, stdin=None):
    # PAIR_DRY only ever runs behind the stub-ssh harness (PAIR_DRY_SSH): without it the dry run
    # would talk to a real docker, and a "dry" apply that creates containers is not a dry run.
    if os.environ.get("PAIR_DRY") == "1" and not os.environ.get("PAIR_DRY_SSH"):
        die("PAIR_DRY=1 needs PAIR_DRY_SSH (stub harness); refusing to touch real docker")
    cmd = ["docker"] + args
    host = side_host(cfg, side)
    if host:
        return sh(ssh_bin() + SSH_OPTS + [host, " ".join(_q(a) for a in cmd)], stdin=stdin)
    return sh(cmd, stdin=stdin)


def _q(s):
    if s and all(c.isalnum() or c in "-_./:=,@" for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"

def resolve_mounts(spec):
    replacements = {
        "__PAIR_COLD_PACK__": "PAIR_COLD_PACK",
        "__PAIR_ART_MOUNT__": "PAIR_ART_MOUNT",
    }
    for mount in spec.get("mounts") or []:
        if mount[0] in replacements:
            key = replacements[mount[0]]
            value = os.environ.get(key)
            if not value or not os.path.isabs(value):
                die(f"{key} must name an absolute host mount path")
            mount[0] = value
    return spec



def spec(cfg, side):
    ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
    path = os.path.join(SPEC_DIR, f"{cfg['pair']}-{side}.json")
    if not os.path.exists(path):
        die(f"missing container spec {path}")
    spec = json.load(open(path))
    if spec["name"] != ct:
        die(f"{path} says name={spec['name']} but pairs.conf says {ct}")
    spec = json.loads(json.dumps(spec).replace("__PAIR_ROOT__", cfg["root"]))
    spec = resolve_mounts(spec)
    # The owner's pack rule (2026-11-02): an expert pack lives on the local drive of the box that
    # serves it. /mnt is the network drive on P7 and NFS from P7 on the GX boxes, so a mount there
    # would have the pair read its experts across the wire (or, worse, a 12 TB drive that is not this
    # box's). The spec is the only place a mount can enter a container, so refuse it here: plan, apply
    # and the smoke all go through this function, and a stale spec can no longer reintroduce it.
    for src, dst in spec.get("mounts") or []:
        for p in (src, dst):
            if p == "/mnt" or p.startswith("/mnt/"):
                die(f"SPEC_REFUSED {path}: mount {dst}={src} is under /mnt; a pack lives on the "
                    f"serving box's own disk, so a container may not mount the network drive")
    return spec


def run_args(spec, name):
    argv = ["run", "-d", "--name", name, "--workdir", spec["workdir"]]
    for k, v in spec.get("env", {}).items():
        argv += ["-e", f"{k}={v}"]
    for src, dst in spec["mounts"]:
        argv += ["--mount", f"type=bind,source={src},target={dst}"]
    argv += spec.get("args", [])
    argv += ["--entrypoint", spec["entrypoint"], spec["image"]] + spec.get("cmd", [])
    return argv


def _ids_key(x):
    return (0, int(x)) if x.isdigit() else (1, x)


def gpu_req(reqs):
    """Normalised HostConfig.DeviceRequests: '' (no GPU), 'all', or 'device=N[,M]'.

    `--gpus all` and `--gpus device=1` both leave a non-empty DeviceRequests, so reading it as a
    boolean made them indistinguishable. That is how the P7 containers ran for a day with the host's
    whole GPU list (nvidia-smi inside listed three, one of them the A4000 that is not ours) while
    every plan/prepare read the spec as matched (2026-11-02).
    """
    ids, everything = set(), False
    for r in reqs or []:
        for d in r.get("DeviceIDs") or []:
            ids.add(str(d))
        if not (r.get("DeviceIDs") or []) and r.get("Count") in (-1, None):
            everything = True
    if ids:
        return "device=" + ",".join(sorted(ids, key=_ids_key))
    return "all" if everything else ""


def spec_gpus(spec):
    """The GPU scope a spec asks for, in the same form gpu_req() returns."""
    args = spec.get("args") or []
    if "--gpus" not in args:
        return ""
    raw = args[args.index("--gpus") + 1].strip().strip("'\"")
    if raw.lower() in ("all", "a"):
        return "all"
    if raw.lower().startswith("device="):
        ids = {x for x in raw.split("=", 1)[1].split(",") if x}
        return "device=" + ",".join(sorted(ids, key=_ids_key))
    return raw.lower()


def live(cfg, side):
    ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
    rc, out, err = docker(cfg, side, ["inspect", ct])
    if rc:
        return None
    d = json.loads(out)[0]
    mounts = {m["Destination"]: m["Source"] for m in (d.get("Mounts") or [])}
    env = {}
    for e in d["Config"].get("Env") or []:
        k, _, v = e.partition("=")
        env[k] = v
    ul = {u["Name"]: u.get("Soft", -1) for u in (d["HostConfig"].get("Ulimits") or [])}
    return {
        "running": bool(d["State"]["Running"]),
        "image": d["Config"]["Image"],
        "workdir": d["Config"].get("WorkingDir") or "",
        "entrypoint": d["Config"].get("Entrypoint") or [],
        "cmd": d["Config"].get("Cmd") or [],
        "mounts": mounts,
        "env": env,
        "network": d["HostConfig"].get("NetworkMode"),
        "ipc": d["HostConfig"].get("IpcMode"),
        "shm": d["HostConfig"].get("ShmSize"),
        "security": d["HostConfig"].get("SecurityOpt") or [],
        "ulimits": ul,
        "devices": [x.get("PathOnHost", "") for x in (d["HostConfig"].get("Devices") or [])],
        "gpus": gpu_req(d["HostConfig"].get("DeviceRequests")),
        "restart": (d["HostConfig"].get("RestartPolicy") or {}).get("Name"),
    }


def plan_side(cfg, side):
    """List of reasons the live container does not match the spec (empty = fine)."""
    ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
    want = spec(cfg, side)
    got = live(cfg, side)
    if got is None:
        return [f"missing:{ct}"]
    why = []
    if got["image"] != want["image"]:
        why.append(f"image={got['image']} want={want['image']}")
    if got["workdir"] != want["workdir"]:
        why.append(f"workdir={got['workdir']}")
    if list(got["entrypoint"]) != [want["entrypoint"]]:
        why.append(f"entrypoint={got['entrypoint']}")
    if list(got["cmd"]) != list(want["cmd"]):
        why.append(f"cmd={got['cmd']}")
    # A wanted mount is satisfied by an identical live mount, or by a live parent that carries it:
    # seam-p7-dev and seam-p7-prod were built with /models <- /models before the spec narrowed to the
    # pack's own directory, and /models/dsv41 is present through it. Calling that "absent" made every
    # dev prepare die on a RECREATE the owner forbids ("never recreate an existing container"), while
    # the container could do the job all along.
    def covering(src, dst):
        for gdst, gsrc in got["mounts"].items():
            if gdst == dst:
                continue
            if (dst == gdst or dst.startswith(gdst.rstrip("/") + "/")) and \
               (src == gsrc or src.startswith(gsrc.rstrip("/") + "/")):
                return gdst, gsrc
        return None

    for src, dst in want["mounts"]:
        seen = got["mounts"].get(dst)
        if seen == src:
            continue
        if seen:
            # `~` wrong source. Nothing here ever means "add a mount" for a mount the live container
            # has and the spec does not -- that is the next loop, and it removes. (A removed mount used
            # to print as `mount:+<dst>=<src>`, which read as an add.)
            why.append(f"mount:~{dst}={seen} want={src}")
            continue
        cover = covering(src, dst)
        if cover:
            why.append(f"advisory:mount:{dst}=covered_by {cover[0]}={cover[1]}")
        else:
            why.append(f"mount:+{dst}=absent want={src}")
    # A mount the spec no longer names is drift, not history. The P7 containers were once given the
    # network drive; the owner's pack rule (2026-11-02: packs live on the serving box's own
    # disk) drops it from the spec. Comparing only the wanted mounts made that removal invisible --
    # plan said "unchanged" and the next flip would have kept the drive in place. An extra live mount
    # is reported here as a removal (`mount:-`); the spec itself may not name /mnt at all
    # (SPEC_REFUSED in spec()).
    #
    # It is an advisory, not a verdict: a removal can only be applied by recreating the container, and
    # the owner forbids recreating one that exists ("never recreate or rename an existing container").
    # So the line stays visible (an operator reading plan sees the difference) while the pair keeps
    # running: nothing about it makes the release wrong, and the pack rule is enforced on the pack the
    # server actually loads (pair_pack_disk.py) and again by the smoke.
    wanted_dst = {dst for _src, dst in want["mounts"]}
    for dst, src in sorted(got["mounts"].items()):
        if dst not in wanted_dst:
            why.append(f"advisory:mount:-{dst}={src} (not in spec)")
    for k, v in want["env"].items():
        if got["env"].get(k) != v:
            why.append(f"env:{k}={got['env'].get(k) or '-'} want={v}")
    if got["network"] != "host" or got["ipc"] != "host":
        why.append(f"net={got['network']}/{got['ipc']}")
    if got["shm"] < 64 * 1024 ** 3:
        why.append(f"shm={got['shm']}")
    if got["security"] != ["label=disable"]:
        why.append(f"security={got['security']}")
    # The GPU scope is compared, not only its presence: a container that sees more than its own GPU
    # can run this pair's arm on the neighbour's card, and the whole point of pairs.conf's `gpu` is
    # that it cannot.
    want_gpu = spec_gpus(want)
    if got["gpus"] != want_gpu:
        why.append(f"gpus={got['gpus'] or 'none'} want={want_gpu or 'none'}")
    if "/dev/infiniband" not in got["devices"]:
        why.append("devices=missing_infiniband")
    if got["ulimits"].get("memlock") != -1:
        why.append(f"ulimit:memlock={got['ulimits'].get('memlock')}")
    ulimit_want = {}
    args = want.get("args", [])
    for i, a in enumerate(args):
        if a == "--ulimit":
            k, _, v = args[i + 1].partition("=")
            ulimit_want[k] = int(v)
    for k, v in ulimit_want.items():
        if got["ulimits"].get(k) != v:
            why.append(f"ulimit:{k}={got['ulimits'].get(k)} want={v}")
    if got["restart"] != "no":
        why.append(f"restart={got['restart']}")
    return why


QROOT = os.environ.get("PAIR_QUEUE_ROOT", os.path.join(ART, "queue"))


def _conf_rows():
    """(pair, p7ct, gxct) for every row of the configured pair inventory."""
    rows = []
    with open(os.environ.get("PAIR_CONF", os.path.join(HERE, "pairs.conf"))) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f_ = line.split()
            if len(f_) >= 9:
                rows.append((f_[0], f_[1], f_[8]))
    return rows


def _live_ticket_uses(ct):
    """True if a live queue runner in any pair's queue has `ct` on its command line."""
    for q in sorted(os.listdir(QROOT)) if os.path.isdir(QROOT) else []:
        qd = os.path.join(QROOT, q, "gpu_queue")
        if not os.path.isdir(qd):
            continue
        for t in os.listdir(qd):
            if not t[0].isdigit():
                continue
            pid = t.split("-")[2]
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    argv = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if ct in argv:
                return f"{q}/gpu_queue/{t} pid={pid}"
    return ""


def check_ownership(cfg):
    """A container belongs to one pair and is never touched while another queue's job runs in it."""
    for side in ("p7", "gx"):
        ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
        for pair, p7ct, gxct in _conf_rows():
            if pair != cfg["pair"] and ct in (p7ct, gxct):
                die(f"{ct} belongs to pair {pair} (pairs.conf); {cfg['pair']} may not touch it")
        who = _live_ticket_uses(ct)
        if who:
            die(f"{ct} is named by a live ticket ({who}); refusing to recreate it")


def real_reasons(why):
    """The reasons that mean "this container cannot do the job", i.e. not the advisories.

    An advisory names a difference only a recreation could remove (an extra mount, a narrower spec
    mount carried by a live parent) or one that is already satisfied. The owner's rule is that an
    existing container is never recreated outside a queued update, so an advisory may not turn into a
    refusal: it is printed for the operator and skipped by the verdict.
    """
    return [w for w in (why or []) if not w.startswith("advisory:")]


def cmd_plan(cfg, quiet=False):
    rc = 0
    for side in ("p7", "gx"):
        ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
        why = plan_side(cfg, side)
        real = real_reasons(why)
        adv = [w[len("advisory:"):] for w in why if w.startswith("advisory:")]
        if not real and not adv:
            if not quiet:
                print(f"PLAN {cfg['pair']} {side} {ct} OK")
            continue
        # "missing" is not "differs": a pair that has never run has no container to replace, and
        # printing RECREATE for it made a first rollout look like a swap of something running.
        exists = docker(cfg, side, ["inspect", "-f", "{{.Id}}", ct])[0] == 0
        if real:
            rc = 1
            print(f"PLAN {cfg['pair']} {side} {ct} {'RECREATE' if exists else 'CREATE'} {' '.join(real)}")
        else:
            print(f"PLAN {cfg['pair']} {side} {ct} OK advisory={' '.join(adv)}")
    return rc


WORKER_RE = "^python3 -u qmesh/worker_seam_server.py"


def stop_our_server(cfg):
    """Stop only OUR pair's server; never touch another container's processes.

    The pattern is anchored on the worker's own argv. The container starts it as
    `bash -lc 'cd /src/exllamav3 && stdbuf … python3 -u qmesh/worker_seam_server.py …'`, and while
    the wrapper exists its cmdline mentions the script too — an unanchored match stops the wrapper and
    leaves the worker holding the pack (2026-11-02: pid 3297 wrapper, 3305 worker).
    """
    ct = cfg["gxct"]
    rc, out, _ = docker(cfg, "gx", ["exec", ct, "pgrep", "-c", WORKER_RE])
    try:
        n = int(out.strip() or 0)
    except ValueError:
        n = 0
    if n:
        print(f"STOP_SERVER {ct} pids={n}")
        docker(cfg, "gx", ["exec", ct, "pkill", "-TERM", "-f", WORKER_RE])
        for _ in range(30):
            rc, out, _ = docker(cfg, "gx", ["exec", ct, "pgrep", "-c", WORKER_RE])
            if (out.strip() or "0") == "0":
                break
            subprocess.run(["sleep", "1"])
        else:
            die(f"server still running in {ct}; refusing to recreate")


def cmd_apply(cfg, sha, ts, force=False):
    if not sha:
        die("apply needs a release sha")
    if not os.path.isdir(os.path.join(cfg["root"], "releases", sha, "src")):
        die(f"release {sha[:8]} is not cut at {cfg['root']}")
    check_ownership(cfg)
    # Only a container that cannot do the job is recreated. An advisory difference (an extra mount, a
    # spec mount carried by a live parent) survives on purpose: applying it means swapping a container
    # that is serving, which is the owner's call, not a rollout's, so the flip must not fail over it.
    todo = [(s, real_reasons(plan_side(cfg, s))) for s in ("p7", "gx")]
    todo = [(s, w) for s, w in todo if w]
    if not todo and not force:
        print(f"APPLY {cfg['pair']} no_recreate_needed")
        return 0
    allow_recreate = os.environ.get("PAIR_ALLOW_RECREATE") == "1"
    for side, _why in todo:
        ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
        rc, _, _ = docker(cfg, side, ["inspect", "-f", "{{.Id}}", ct])
        if rc == 0 and not allow_recreate:
            # This rollout adds containers, it never swaps an existing one: the switch from an old
            # container to its release-mounted replacement happens only inside a queued PairUpdate
            # job that holds the pair's lock, after the owner has said when.
            die(f"{ct} already exists but its spec differs; refusing to recreate an existing "
                f"container (PAIR_ALLOW_RECREATE=1 does it inside a queued PairUpdate only)")
    _rc, _out, _ = docker(cfg, "gx", ["inspect", "-f", "{{.State.Running}}", cfg["gxct"]])
    if _rc == 0 and _out.strip() == "true":
        stop_our_server(cfg)
    for side, why in todo:
        ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
        rc, _, _ = docker(cfg, side, ["inspect", "-f", "{{.Id}}", ct])
        if rc != 0:
            print(f"CREATE {cfg['pair']} {side} {ct} {' '.join(why)}")
            argv = run_args(spec(cfg, side), ct)
            rc, out, err = docker(cfg, side, argv)
            if rc:
                die(f"docker run failed for {ct}: {err.strip()[:400]}")
            docker(cfg, side, ["start", ct])
            continue
        print(f"RECREATE {cfg['pair']} {side} {ct} {' '.join(why)}")
        new = f"{ct}.new-{ts}"
        docker(cfg, side, ["rm", "-f", new])
        argv = run_args(spec(cfg, side), new)
        rc, out, err = docker(cfg, side, argv)
        if rc:
            die(f"docker run failed for {new}: {err.strip()[:400]}")
        docker(cfg, side, ["stop", "--time", "20", ct])
        docker(cfg, side, ["rename", ct, f"{ct}.bak-{ts}"])
        rc, out, err = docker(cfg, side, ["rename", new, ct])
        if rc:
            docker(cfg, side, ["rename", f"{ct}.bak-{ts}", ct])
            die(f"rename {new} -> {ct} failed: {err.strip()[:400]}")
        docker(cfg, side, ["start", ct])
    # A pair's first containers have no current to be wired through, and creating one here would put
    # an unvalidated release in the place every client resolves: on 2026-11-02 `pair_prepare.sh prod`
    # left prod/current on the new sha at 00:00, before dev had passed, which is exactly the canary
    # rule the owner set. So the containers are wired straight at the release tree instead, and
    # `current` stays absent until a ticket promotes this sha (flip + smoke + verify). Such a
    # container is armed, not serving: its /src/exllamav3 names releases/<sha>/src, and nothing that
    # resolves through /pair/current can import.
    if not os.path.lexists(os.path.join(cfg["root"], "current")):
        print(f"LINK_UNPROMOTED {cfg['pair']} current=absent containers->releases/{sha[:12]} "
              f"(current moves only on a passed promote)")
    cmd_inner(cfg, sha, verify=False)
    # The post-check asks the same question the verdict asked: advisories are printed (they are the
    # operator's business) but they are not a failed apply.
    left = [(s, real_reasons(w)) for s, w in ((s, plan_side(cfg, s)) for s in ("p7", "gx"))]
    left = [(s, w) for s, w in left if w]
    if left:
        die("after recreate the containers still differ: " +
            " ".join(f"{s}:{';'.join(w)}" for s, w in left))
    print(f"APPLY {cfg['pair']} containers_ok ts={ts}")
    return 0


def cmd_rollback(cfg, ts):
    check_ownership(cfg)
    stop_our_server(cfg)
    for side in ("p7", "gx"):
        ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
        bak = f"{ct}.bak-{ts}"
        rc, out, _ = docker(cfg, side, ["inspect", "-f", "{{.State.Running}}", bak])
        if rc:
            print(f"ROLLBACK {cfg['pair']} {side} no {bak}")
            continue
        docker(cfg, side, ["stop", "--time", "20", ct])
        docker(cfg, side, ["rename", ct, f"{ct}.failed-{ts}"])
        docker(cfg, side, ["rename", bak, ct])
        docker(cfg, side, ["start", ct])
        print(f"ROLLBACK {cfg['pair']} {side} {ct} restored (new one kept as {ct}.failed-{ts})")
    return 0


def link_side_cmd(root, sha):
    """Atomic symlink flip with RELATIVE targets.

    The same directory is mounted at a different path inside each container (/pair), so an absolute
    target dangles there: on 2026-11-01 10:15 `readlink -f /src/exllamav3` returned nothing and the
    flip failed after both sides had built. Relative targets read correctly from both namespaces.
    """
    return (f"set -e; mkdir -p {root}/releases; cd {root}; "
            f"ln -sfn releases/{sha} .current.tmp && mv -T .current.tmp current; "
            f"ln -sfn releases/{sha}/ext .extcache.tmp && mv -T .extcache.tmp extcache; "
            f"readlink -f {root}/current; readlink -f {root}/extcache")


def cmd_link(cfg, sha):
    if not sha:
        die("link needs a release sha")
    rel = os.path.join(cfg["root"], "releases", sha)
    if not os.path.isdir(os.path.join(rel, "src")):
        die(f"release {sha[:8]} is not cut at {cfg['root']}")
    cmd = link_side_cmd(cfg["root"], sha)
    rc, out, err = sh(["bash", "-c", cmd])
    if rc:
        die(f"P7 symlink flip failed: {err.strip()[:300]}")
    print(f"LINK {cfg['pair']} p7 {out.split()[0] if out.split() else '-'}")
    ssh = ssh_bin() + SSH_OPTS + [cfg["gx_host"], cmd]
    rc, out, err = sh(ssh)
    if rc:
        die(f"GX symlink flip failed: {err.strip()[:300]}")
    print(f"LINK {cfg['pair']} gx {out.split()[0] if out.split() else '-'} "
          f"ext={out.split()[1] if len(out.split()) > 1 else '-'}")
    return 0


def cmd_inner(cfg, sha=None, verify=True):
    """Inside each container: /src/exllamav3 must resolve into this release.

    readlink -f resolves the whole chain, so the answer is /pair/releases/<sha>/src, not the
    intermediate /pair/current/src -- accept either, but it has to be THIS sha.

    With no current, a container being armed for its first release is wired straight at
    releases/<sha>/src: that is a container nobody may serve from yet (no current, so nothing that
    resolves through /pair/current can import), and it is what lets pair_prepare build inside it
    without promoting the release first. A *verified* wiring still needs a current, because there is
    nothing to verify a promotion against until one exists.
    """
    promoted = os.path.lexists(os.path.join(cfg["root"], "current"))
    if not promoted and (verify or not sha):
        die(f"{cfg['root']}/current is missing: refusing to wire a container to an unreleased tree")
    target = "/pair/current/src" if promoted else f"/pair/releases/{sha}/src"
    for side in ("p7", "gx"):
        ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
        rc, out, err = docker(cfg, side, ["inspect", "-f", "{{.State.Running}}", ct])
        if rc or out.strip() != "true":
            print(f"INNER {cfg['pair']} {side} {ct} not_running")
            continue
        cmd = ("set -e; mkdir -p /src /pair; "
               "if [ ! -L /src/exllamav3 ]; then rm -rf /src/exllamav3; fi; "
               f"ln -sfn {target} /src/exllamav3")
        # right after a recreate, /pair/current may not have moved yet: wire the symlink, verify later
        if verify:
            cmd += "; readlink -f /src/exllamav3"
        rc, out, err = docker(cfg, side, ["exec", ct, "bash", "-lc", cmd])
        if rc:
            die(f"inner symlink in {ct} failed: {err.strip()[:300]}")
        got = out.strip()
        if not verify:
            print(f"INNER {cfg['pair']} {side} {ct} linked{'' if promoted else ' target=' + target}")
            continue
        want_sha = sha
        if not want_sha:
            _, cur, _ = docker(cfg, side, ["exec", ct, "readlink", "-f", "/pair/current"])
            want_sha = os.path.basename(cur.strip())
        ok = {f"/pair/releases/{want_sha}/src", "/pair/current/src"}
        if got not in ok:
            die(f"inner symlink in {ct} resolved to {got!r}, want one of {sorted(ok)}")
        print(f"INNER {cfg['pair']} {side} {ct} {got}")
    return 0


def cmd_ls(cfg):
    for side in ("p7", "gx"):
        ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
        rc, out, _ = docker(cfg, side, ["inspect", "-f", "{{.State.Running}} {{.Config.Image}}", ct])
        print(f"CONTAINER {cfg['pair']} {side} {ct} "
              f"{'running ' + out.strip().split()[1] if rc == 0 else 'ABSENT'}")
    return 0


def main():
    if len(sys.argv) < 3:
        die(__doc__)
    what, pair = sys.argv[1], sys.argv[2]
    cfg = load_conf(pair)
    args = sys.argv[3:]
    if what == "plan":
        return cmd_plan(cfg, quiet="--quiet" in args)
    if what == "apply":
        ts = ""
        if "--ts" in args:
            i = args.index("--ts")
            ts = args[i + 1]
            del args[i:i + 2]
        ts = ts or subprocess.run(["date", "-u", "+%Y%m%dT%H%M%SZ"],
                                  capture_output=True, text=True).stdout.strip()
        return cmd_apply(cfg, args[0] if args else "", ts, force="--force" in args)
    if what == "rollback":
        return cmd_rollback(cfg, args[0] if args else "")
    if what == "guard":
        return check_ownership(cfg)
    if what == "link":
        return cmd_link(cfg, args[0] if args else "")
    if what == "inner":
        return cmd_inner(cfg)
    if what == "ls":
        return cmd_ls(cfg)
    die(f"unknown subcommand {what!r}")


if __name__ == "__main__":
    sys.exit(main())
