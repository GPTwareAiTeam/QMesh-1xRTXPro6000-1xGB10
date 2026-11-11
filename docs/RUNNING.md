# Running the R5 pair

This is a site-configured deployment, not a one-command installer. The measured development pair used an RTX PRO 6000 Blackwell Max-Q with 96 GB and a GB10 unified-memory system over 100 GbE / RoCE. Different hardware, packs, network and memory pressure change the attainable capacity and performance.

## Prerequisites

- Obtain the [official DeepSeek-V4.1-Flash-1017 weights](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), including their `encoding` directory, and matching R5 hot and cold packs. Neither weights nor packs are included. A public pack download and site-independent pack-building recipe are not yet specified.
- Install compatible NVIDIA drivers, CUDA toolkits and CUDA-enabled PyTorch on both hosts. The project specifies Python >= 3.10.11 and PyTorch >= 2.6; Ninja and a C++/CUDA compiler are needed for native extensions. Install project dependencies from `requirements.txt` and `pyproject.toml` (for example, `pip install -e .` after installing the appropriate PyTorch build). Provision matching release code and native extensions for each architecture in prepared containers on both hosts.
- Provide enough GPU and host/unified memory for the packs, cache, runtime, prefill workspaces and network slabs. Do not assume historical memory measurements for a different pack apply to R5. Check that both pack manifests bind the same placement; the local and remote cold expert order must agree.
- Provide Docker with NVIDIA GPU access, management SSH access to GB10, and a working RoCE data path with appropriate devices, GIDs, peer address and ports. Protect management and API access. Container paths passed to the server must exist inside the P7 container.

## Configure and launch

1. Create a site-specific `PAIR_CONF` using the column header in `qmesh/release/pairs.conf`. Supply a `prod` row with unique P7 and GX container names, P7 GPU index, RoCE device and GIDs, GB10 data-plane peer address and management SSH target, release root, control port, row buckets, architectures, load margin and workspace rows. The template contains no site row. The same release root path is used on both hosts. Provision containers and matching release/extension artifacts using the tools in `qmesh/release/`; consult their usage and site configuration before applying them. Set `PAIR_COLD_PACK` and `PAIR_ART_MOUNT` to the site's cold-pack and artifact host mount paths when creating containers. Shared paths default under `HOME` and may need site overrides.
2. Ensure `pair_check.sh prod` reports both sides `IN_SYNC`, name the hot checkpoint, and identify the plain ColdSteal cold-pack directory in the P7 container. The scripts validate the pack and release before serving.
3. Start from the release host, substituting actual paths:

```sh
export PAIR_CONF=/path/to/site-pairs.conf
export PAIR_COLD_PACK=/path/to/cold-pack
export PAIR_ART_MOUNT=/path/to/artifacts
export CHECKPOINT=/path/in/p7-container/to/r5-hot-pack
qmesh/release/pair_server.sh prod start \
  -e ENC_DIR=/path/in/p7-container/to/official/encoding \
  -e QMESH_COLDSTEAL_SRC=/path/in/p7-container/to/plain-cold-pack
qmesh/release/pair_server.sh prod status
qmesh/release/pair_server.sh prod stop
```

Start ensures the GB10 worker first. `stop` stops the HTTP server only; the separately managed worker remains loaded. Direct invocation of `seam_server_ensure.sh` requires its `GX` SSH host and its gate/preflight prerequisites; the pair tools supply the management target from configuration. The API binds `127.0.0.1` by default, on port 8000 for prod and 8001 for dev (`SRV_HOST` and `SRV_PORT` control the launcher); `API_KEY` is forwarded when set. Do not expose an unauthenticated API to an untrusted network.

The launcher reads the active release's `current/src/qmesh/release/serving.env` automatically on start. Its measured R5 settings are:

| Key | Default | Purpose |
| --- | --- | --- |
| `BATCH` | `8` | Maximum concurrent batch width. |
| `QMESH_ALLOW_BATCH_UNSAFE` | `1` | Acknowledges that with BATCH>1 a request's greedy output can differ on near-tied tokens depending on what else is decoding; the server refuses BATCH>1 without it. |
| `EXL3_MOE_COOP_ROWS` | `48` | Cooperative MoE row limit. |
| `DRAFT_MAX_ROWS` | `48` | Draft row limit. |
| `QMESH_COLDSTEAL` | `1` | ColdSteal: the RTX computes cached cold experts itself while it waits for the GB10. |
| `QMESH_COLDSTEAL_SLOTS` | `8` | ColdSteal slots per layer. |
| `QMESH_COLDSTEAL_SMAX` | `8` | Maximum resident experts stolen per layer per call. |
| `QMESH_COLDSTEAL_MAX_ROWS` | `48` | Maximum call width eligible for ColdSteal. |
| `EXL3_DSV41_BATCH` | `1` | Enables batched attention. |
| `EXL3_DRAFT_WIDE_JOBS` | `4` | Draft-width cap threshold in concurrent jobs. |
| `EXL3_DRAFT_WIDE_WINDOW` | `2` | Draft-width cap window. |
| `EXL3_KV_GROW` | `1` | Enables growable KV cache, borrowing ColdSteal slot memory as needed. |
| `EXL3_KV_GROW_FREE_MIB` | `512` | KV growth budget in MiB before borrowing slots. |
| `EXL3_DSV4_SEQ_TOKENS` | `1048576` | Per-request sequence token limit. |
| `CACHE_TOKENS` | `5000192` | Shared KV cache token capacity. |

`QMESH_COLDSTEAL_SRC` is required site configuration, not a shipped path: pass it with `-e` as above, naming the plain cold-pack shards accessible inside the P7 container. Without it ColdSteal cannot arm. The caller's `CACHE_TOKENS`, `PAIR_SRV_EXTRA_ENVS` entries, and final `-e KEY=VALUE` arguments override the release defaults. If a release has no `serving.env`, the launcher's fallback is `CACHE_TOKENS=65536`, not the R5 value. Ensure enough memory for the configured workload; the shared cache capacity is not a per-request guarantee.

Other launcher defaults include `MAX_CHUNK=16384`, `DSPARK=1`, `NUM_DRAFT=5`, `KV_BITS=4`, `EXL3_WHOLE_GRAPH=1`, `PREFILL_ADAPTIVE=1`, `SEAM_ROWS=8192`, `SEAM_NS=1`, `SEAM_OVERLAP=1`, and `SEAM_COPY_STREAM=1`.

**DSpark drafting: Adaptive K (EV gate, K ∈ {3,5}).** Each round, the gate picks verify width by expected emitted tokens per measured round cost on this pair. `qmesh/release/pair_server.sh` sets `EXL3_DSPARK_GATE=ev`, `EXL3_DSPARK_CONF=0`, `EXL3_DSPARK_EV_KMIN=3`, and `EXL3_DSPARK_EV_COST=1:1e9,2:1e9,3:1e9,4:75.69,5:1e9,6:82.95`. With four or more concurrent jobs, `EXL3_DRAFT_WIDE_JOBS=4` and `EXL3_DRAFT_WIDE_WINDOW=2` narrow the draft window further. The prior TP=2 repository used the `auto` threshold gate instead.

See `qmesh/release/pair_server.sh` for other settings, [Architecture](ARCHITECTURE.md) for what each one does and [Results](RESULTS.md) for the measured effect of the release configuration.
