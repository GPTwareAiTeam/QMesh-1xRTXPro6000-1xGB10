#!/usr/bin/env python3
"""Stub docker for the pair-release dry sweep: remembers `run` argv and answers `inspect` from it."""
import json, os, sys
ST = "/tmp/stub/state.json"
st = json.load(open(ST)) if os.path.exists(ST) else {}
def save(): json.dump(st, open(ST, "w"))
a = sys.argv[1:]
cmd = a[0]
if cmd == "run":
    name = a[a.index("--name") + 1]
    d = {"State": {"Running": True}, "Config": {"Image": "", "WorkingDir": "", "Entrypoint": [], "Cmd": [], "Env": []},
         "HostConfig": {"NetworkMode": "default", "IpcMode": "private", "ShmSize": 67108864, "SecurityOpt": [],
                        "Ulimits": [], "Devices": [], "DeviceRequests": [], "RestartPolicy": {"Name": "no"}},
         "Mounts": []}
    i = 1
    while i < len(a):
        x = a[i]
        if x == "--name": i += 2
        elif x == "--workdir": d["Config"]["WorkingDir"] = a[i + 1]; i += 2
        elif x == "-e": d["Config"]["Env"].append(a[i + 1]); i += 2
        elif x == "--mount":
            kv = dict(p.split("=", 1) for p in a[i + 1].split(","))
            d["Mounts"].append({"Source": kv["source"], "Destination": kv["target"]}); i += 2
        elif x == "--ulimit":
            k, _, v = a[i + 1].partition("="); d["HostConfig"]["Ulimits"].append({"Name": k, "Soft": int(v)}); i += 2
        elif x == "--device": d["HostConfig"]["Devices"].append({"PathOnHost": a[i + 1].split(":")[0]}); i += 2
        elif x == "--security-opt": d["HostConfig"]["SecurityOpt"].append(a[i + 1]); i += 2
        elif x == "--network": d["HostConfig"]["NetworkMode"] = a[i + 1]; i += 2
        elif x == "--ipc": d["HostConfig"]["IpcMode"] = a[i + 1]; i += 2
        elif x == "--shm-size": d["HostConfig"]["ShmSize"] = int(a[i + 1]); i += 2
        elif x == "--gpus": d["HostConfig"]["DeviceRequests"] = [{}]; i += 2
        elif x == "--restart": d["HostConfig"]["RestartPolicy"] = {"Name": a[i + 1]}; i += 2
        elif x == "--entrypoint": d["Config"]["Entrypoint"] = [a[i + 1]]; i += 2
        elif not x.startswith("-") and not d["Config"]["Image"]:
            d["Config"]["Image"] = x; d["Config"]["Cmd"] = a[i + 1:]; break
        else: i += 1
    st[name] = d; save(); print("id-" + name); sys.exit(0)
if cmd == "inspect":
    ct = a[-1]
    if ct not in st and not ct.isdigit() if False else ct not in st:
        print(f"Error: No such object: {ct}", file=sys.stderr); sys.exit(1)
    if any("Running" in x for x in a[1:]):
        print("true"); sys.exit(0)
    print(json.dumps([st[ct]])); sys.exit(0)
if cmd == "exec":
    ct, rest = a[1], a[2:]
    open("/tmp/stub/exec.log", "a").write(" ".join(rest) + "\n")
    if ct not in st: print(f"no such container {ct}", file=sys.stderr); sys.exit(1)
    if rest[0] in ("pgrep", "pkill"): sys.exit(1)
    if rest[0] == "readlink": print("/pair/current/src"); sys.exit(0)
    if rest[0] == "ln": sys.stdout.write("LINK " + " ".join(rest) + "\n"); sys.exit(0)
    if rest[0] == "mkdir": sys.exit(0)
    if rest[0] == "sh": print("/pair/current/src"); sys.exit(0)
    if rest[0] in ("bash", "ln", "readlink", "sh", "mkdir"):
        if rest[1].startswith("-l") or rest[1] == "-c":
            print("/pair/current/src"); sys.exit(0)
        print("/pair/current/src"); sys.exit(0)
    print("exec_ok"); sys.exit(0)
if cmd in ("stop", "start", "rm"): save(); sys.exit(0)
if cmd == "rename":
    old, new = a[1], a[2]
    if old in st: st[new] = st.pop(old); save(); sys.exit(0)
    print("no such", file=sys.stderr); sys.exit(1)
print("docker_stub", " ".join(a)); sys.exit(0)
