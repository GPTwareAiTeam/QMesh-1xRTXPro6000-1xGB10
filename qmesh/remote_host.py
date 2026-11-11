"""P7-side client for the GX1 EXL3 cold-expert worker."""
from __future__ import annotations

import pickle
import socket
import struct

import torch


def recv_msg(conn):
    header = conn.recv(8)
    if not header:
        raise EOFError
    size = struct.unpack("<Q", header)[0]
    data = bytearray()
    while len(data) < size:
        block = conn.recv(size - len(data))
        if not block:
            raise EOFError
        data += block
    return pickle.loads(data)


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("<Q", len(data)) + data)


class RemoteExpertHost:
    """Minimal synchronous host with the MoeCpuHost methods the split uses.

    Correctness path: one round trip per layer. It is intentionally not
    graph-capturable yet; the graph-safe RDMA ring replaces it after parity.
    """

    @classmethod
    def for_config(cls, infer_params):
        host = getattr(infer_params, "remote_expert_host", None)
        if host is None:
            raise RuntimeError("remote_expert_host is not configured")
        hosts = getattr(infer_params, "remote_expert_hosts", None)
        if hosts is None:
            hosts = {}
            infer_params.remote_expert_hosts = hosts
        key = (host["host"], host["port"])
        if key not in hosts:
            hosts[key] = cls(host["host"], host["port"])
        return hosts[key]

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = socket.create_connection((host, port), timeout=30)
        self.sock.settimeout(None)
        # The split path uses the issue/collect branch for every batch shape.
        self.stream_min_rows = 10**9
        self.layer_index = {}

    def register_layer(self, key, *args, **kwargs):
        idx = int(key.split(".")[1])
        self.layer_index[key] = idx
        return idx

    def _forward(self, layer_idx, y, selected, weights):
        y = y.detach().half().cpu()
        selected = selected.detach().long().cpu()
        weights = weights.detach().float().cpu()
        send_msg(self.sock, {
            "cmd": "forward",
            "layer": layer_idx,
            "x": y,
            "ids": selected,
            "weights": weights,
        })
        reply = recv_msg(self.sock)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "remote worker failed"))
        return reply["out"].to(y.device if y.is_cuda else "cpu")

    def submit_issue(self, layer_idx, y, selected, weights):
        # Synchronous correctness path: do the round trip now and return the
        # device result as the pending handle. cpu_split_combine adds it.
        out = self._forward(layer_idx, y, selected, weights)
        return out.to(y.device, non_blocking=False)

    def submit_collect(self, handle):
        return handle

    def submit_prefill(self, layer_idx, y, selected, weights):
        return self._forward(layer_idx, y, selected, weights).to(y.device)

    def submit_issue_fused(self, *args, **kwargs):
        raise NotImplementedError("remote path uses EXL3_MOE_SPLIT_FUSED=0")

    def unregister(self):
        pass

    def shutdown(self):
        try:
            self.sock.close()
        except Exception:
            pass
