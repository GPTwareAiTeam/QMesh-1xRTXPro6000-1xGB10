#!/usr/bin/env bash
# pair_server_smoke.sh -- end-to-end checks of the pair's OpenAI-compatible API (P7 side).
#
#   pair_server_smoke.sh <prod|dev> [http port]
#
# One model instance on one card, so there is no rank agreement to check and no token-identity
# assertion here: what this proves is the wire and the two behaviours only a long-lived server can
# have -- the streaming shape, the drafter/prefix-cache path across two turns of one conversation,
# and the shutdown that leaves neither a process nor a listening port behind. The port defaults to
# PAIR_SRV_PORT/SRV_PORT/8000 -- whatever pair_server.sh would start on, since both boxes share the
# host network namespace and 127.0.0.1 here is the server's own loopback.
#
# When nothing is serving it starts the server (pair_server.sh start, which also brings the GX
# worker up) and stops it again on every exit path, including a failed check: a 60+ GiB server left
# running by a smoke is a resource leak with a bill. When a server is already serving it is used and
# left alone -- stopping somebody else's load to run a smoke is worse than the smoke.
#
# MODEL_ID is what /v1/models must report; it is forwarded to pair_server.sh, so a smoke that starts
# the server starts it under the same id it then requires.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

pair_load "${1:?usage: pair_server_smoke.sh <prod|dev> [http port]}"
# RM10 2026-11-05 (Main): the P7 client half is named, never derived (see pair_ref.sh:33).
PAIR_PACK_P7=$(pair_ckpt_require) || exit 1
export PAIR_PACK_P7
PORT=${2:-${PAIR_SRV_PORT:-${SRV_PORT:-8000}}}
# A dry run describes a launch; it cannot open a socket, and pair_server.sh status would report a
# server that does not exist. Refuse before anything pretends to check one.
if [ "$PAIR_DRY" = 1 ]; then
  pair_die "SMOKE_NEEDS_SERVER (PAIR_DRY=1 only describes; unset it to run the checks)"
fi
MODEL_ID=${MODEL_ID:-deepseek-v4.1-flash-pair}
export SRV_PORT=$PORT
# Both are forwarded by pair_server.sh from its environment, so exporting them here means a smoke
# that starts the server starts it under the same id/key this smoke then talks to.
export MODEL_ID
export API_KEY=${API_KEY:-}
LOG=$PAIR_STATE_DIR/server/pair-server.log
CT_LOG=$(pair_ct_path "$LOG")

own=0
stop_own() {
  [ "$own" = 1 ] || return 0
  own=0
  pair_step "SMOKE_SERVER_STOP pair=$PAIR (started by this smoke)"
  "$HERE/pair_server.sh" "$PAIR" stop
}
trap stop_own EXIT

if "$HERE/pair_server.sh" "$PAIR" status >/dev/null 2>&1; then
  pair_step "SMOKE_SERVER_REUSE pair=$PAIR port=$PORT (already serving; left running)"
else
  pair_step "SMOKE_SERVER_START pair=$PAIR port=$PORT (nothing was serving)"
  own=1
  "$HERE/pair_server.sh" "$PAIR" start || pair_die "SMOKE_SERVER_START_FAILED pair=$PAIR (see $LOG)"
fi

# Host-side urllib only: the checks have to see what a caller sees, not what the engine thinks.
if ! python3 - "$PORT" "$MODEL_ID" <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:%s" % sys.argv[1]
EXPECTED_MODEL = sys.argv[2]
AUTH = {"Authorization": "Bearer " + os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}


def call(path, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload).encode()
    headers = dict(AUTH)
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw or b"{}")
        except ValueError:
            return exc.code, {"raw": raw.decode(errors="replace")}


def stream(path, payload, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={**AUTH, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode()


def ok(name, detail=""):
    print("PASS %s%s" % (name, (" " + detail) if detail else ""), flush=True)


def fail(name, detail):
    print("FAIL %s: %s" % (name, detail), flush=True)
    sys.exit(1)


# ---- the model list a client discovers the endpoint with -------------------------------------
# A 401 here is not "the wire is wrong": either a leftover server is enforcing a key this shell does not
# have, or this smoke started one and the key did not travel. Say which, because the bare "status=401"
# reads like a broken endpoint and costs a debugging round on a server that is in fact healthy.
status, body = call("/v1/models")
if status == 401:
    fail("auth" if os.environ.get("API_KEY") else "auth_missing_key",
         "401 on /v1/models with %s: the serving process enforces an API_KEY this smoke %s "
         "(a reused server started under a different key, or a start that did not carry it); "
         "re-run with the server's key or stop it so this smoke starts its own"
         % ("API_KEY set" if os.environ.get("API_KEY") else "no API_KEY",
            "does not match" if os.environ.get("API_KEY") else "was not given"))
models = body.get("data") or []
if status != 200 or not models:
    fail("models", "status=%s body=%s" % (status, body))
model = models[0].get("id")
if model != EXPECTED_MODEL:
    fail("models", "id=%r expected %r" % (model, EXPECTED_MODEL))
ok("models", "id=%s n=%d" % (model, len(models)))

# ---- plain completion: text out, usage counted ------------------------------------------------
status, body = call("/v1/completions", {"model": model, "prompt": "The capital of France is",
                                        "max_tokens": 8, "temperature": 0})
try:
    text = body["choices"][0]["text"]
    used = body["usage"]["completion_tokens"]
except (KeyError, IndexError, TypeError):
    fail("completions", "status=%s body=%s" % (status, body))
if status != 200 or not text.strip() or used <= 0:
    fail("completions", "status=%s text=%r completion_tokens=%s" % (status, text, used))
ok("completions", "text=%r completion_tokens=%s" % (text, used))

# ---- chat: the streamed deltas must reproduce the non-streaming text, when the engine repeats ------
MESSAGES = [{"role": "user", "content": "Say hello in one short sentence."}]
status, plain = call("/v1/chat/completions",
                     {"model": model, "messages": MESSAGES, "max_tokens": 24, "temperature": 0})
try:
    plain_text = plain["choices"][0]["message"]["content"] or ""
except (KeyError, IndexError, TypeError):
    fail("chat", "status=%s body=%s" % (status, plain))
if status != 200 or not plain_text.strip():
    fail("chat", "status=%s text=%r" % (status, plain_text))
# Same request again, non-streamed. The MoE epilogue retires blocks through fp32 atomics, so greedy
# decode is not run-to-run reproducible on this engine (notes/GraphsPort.md:218); if two identical
# plain requests already disagree, an equality assertion against the streamed text proves nothing, so
# the divergence is reported and not failed. Determinism has its own arm: SEAM_MOE_CPP_DET=1.
status, again = call("/v1/chat/completions",
                     {"model": model, "messages": MESSAGES, "max_tokens": 24, "temperature": 0})
try:
    again_text = (again["choices"][0]["message"]["content"] or "") if status == 200 else ""
except (KeyError, IndexError, TypeError):
    again_text = ""
repeatable = again_text == plain_text
raw = stream("/v1/chat/completions",
             {"model": model, "messages": MESSAGES, "max_tokens": 24, "temperature": 0,
              "stream": True, "stream_options": {"include_usage": True}})
deltas, done, usage_event = [], False, None
for line in raw.splitlines():
    if not line.startswith("data: "):
        continue
    data = line[6:]
    if data == "[DONE]":
        done = True
        continue
    event = json.loads(data)
    for choice in event.get("choices") or []:
        content = (choice.get("delta") or {}).get("content")
        if content:
            deltas.append(content)
    if "usage" in event:
        usage_event = event["usage"]
if not done:
    fail("chat", "stream ended without [DONE]")
stream_text = "".join(deltas)
if not repeatable:
    # The engine is not repeatable on this prompt, so equality with the streamed text is not evidence.
    # Still require that the stream itself is well formed, and say the divergence out loud.
    if not stream_text.strip():
        fail("chat", "stream text empty (plain=%r repeat=%r)" % (plain_text, again_text))
    if usage_event is None or usage_event.get("completion_tokens", 0) <= 0:
        fail("chat", "stream usage event missing or empty: %s" % (usage_event,))
    ok("chat", "text=%r nondeterministic=1 repeat=%r stream_matches=%d stream_completion_tokens=%s"
       % (plain_text, again_text, 0 if stream_text != plain_text else 1,
          usage_event["completion_tokens"]))
elif stream_text != plain_text:
    fail("chat", "stream=%r plain=%r repeat=%r" % (stream_text, plain_text, again_text))
elif usage_event is None or usage_event.get("completion_tokens", 0) <= 0:
    fail("chat", "stream usage event missing or empty: %s" % (usage_event,))
else:
    ok("chat", "text=%r identical=1 repeatable=1 stream_completion_tokens=%s"
       % (plain_text, usage_event["completion_tokens"]))

# ---- two turns of one conversation: the shared prefix must come back from the prefix cache -----
LONG = "The quick brown fox jumps over the lazy dog. " * 400 + "Summarize this in one sentence."
status, first = call("/v1/chat/completions",
                     {"model": model, "messages": [{"role": "user", "content": LONG}],
                      "max_tokens": 24, "temperature": 0})
try:
    first_text = first["choices"][0]["message"]["content"] or ""
    first_prompt = first["usage"]["prompt_tokens"]
except (KeyError, IndexError, TypeError):
    fail("prefix-cache", "turn 1 status=%s body=%s" % (status, first))
status, second = call("/v1/chat/completions",
                      {"model": model, "max_tokens": 8, "temperature": 0,
                       "messages": [{"role": "user", "content": LONG},
                                    {"role": "assistant", "content": first_text},
                                    {"role": "user", "content": "Now say only the number of words."}]})
try:
    usage2 = second["usage"]
    cached = (usage2.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    token_ids = second["choices"][0].get("token_ids") or []
except (KeyError, IndexError, TypeError):
    fail("prefix-cache", "turn 2 status=%s body=%s" % (status, second))
if status != 200 or cached <= 0:
    fail("prefix-cache", "turn 1 prompt_tokens=%s turn 2 prompt_tokens=%s cached_tokens=%s"
         % (first_prompt, usage2.get("prompt_tokens"), cached))
if not token_ids:
    fail("prefix-cache", "turn 2 returned no token_ids")
ok("prefix-cache", "turn1_prompt_tokens=%s turn2_prompt_tokens=%s cached_tokens=%s token_ids=%d"
   % (first_prompt, usage2.get("prompt_tokens"), cached, len(token_ids)))

# ---- tools on chat: the parse path, not just the refusal --------------------------------------
# The reference compared both ranks' token ids here. There is one rank, so the check is the wire
# contract: a finish_reason the client can dispatch on, an assistant message, and tool calls whose
# ids the server minted (a missing id makes an agent loop unable to answer the call).
TOOLS = [{"type": "function",
          "function": {"name": "get_weather",
                       "parameters": {"type": "object",
                                      "properties": {"city": {"type": "string"}},
                                      "required": ["city"]}}}]
status, tooled = call("/v1/chat/completions",
                      {"model": model, "max_tokens": 48, "temperature": 0, "tools": TOOLS,
                       "messages": [{"role": "user",
                                     "content": "What is the weather in Paris? Use the tool."}]})
try:
    tooled_choice = tooled["choices"][0]
    tooled_message = tooled_choice["message"]
except (KeyError, IndexError, TypeError):
    fail("chat-tools", "status=%s body=%s" % (status, tooled))
if (status != 200 or tooled_choice.get("finish_reason") not in ("tool_calls", "stop")
        or tooled_message.get("role") != "assistant"):
    fail("chat-tools", "status=%s finish_reason=%s message=%s"
         % (status, tooled_choice.get("finish_reason"), tooled_message))
tooled_calls = tooled_message.get("tool_calls") or []
for invocation in tooled_calls:
    if not str(invocation.get("id", "")).startswith("call_") \
            or not (invocation.get("function") or {}).get("name"):
        fail("chat-tools", "malformed tool call %s" % (invocation,))
ok("chat-tools", "finish_reason=%s tool_calls=%d content=%r"
   % (tooled_choice["finish_reason"], len(tooled_calls),
      (tooled_message.get("content") or "")[:40]))

# ---- tools on a completion must be refused, not silently ignored ------------------------------
status, body = call("/v1/completions",
                    {"model": model, "prompt": "x", "max_tokens": 4,
                     "tools": [{"type": "function",
                                "function": {"name": "lookup", "parameters": {"type": "object"}}}]})
if status != 400:
    fail("tools", "status=%s (expected 400) body=%s" % (status, body))
ok("tools", "HTTP 400 type=%s" % ((body.get("error") or {}).get("type"),))
PY
then
  pair_step "SMOKE_FAILED pair=$PAIR port=$PORT"
  exit 1
fi

if [ "$own" = 1 ]; then
  stop_own
  # The socket must be gone with the process, and the log must not hold a traceback: a server that
  # died on an exception can still have answered every check above.
  if python3 - "$PORT" <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(1)
try:
    still_listening = sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0
finally:
    sock.close()
sys.exit(1 if still_listening else 0)
PY
  then
    pair_step "PASS shutdown port=$PORT closed pid=- log=$LOG"
  else
    pair_die "SMOKE_SHUTDOWN_FAILED port=$PORT still listening after pair_server.sh stop"
  fi
  if docker exec "$P7CT" grep -q Traceback "$CT_LOG" 2>/dev/null; then
    pair_die "SMOKE_TRACEBACK pair=$PAIR (the server log holds a traceback: $LOG)"
  fi
else
  pair_step "SKIP shutdown (not this smoke's server to stop)"
fi

pair_step "SMOKE_OK pair=$PAIR port=$PORT"
