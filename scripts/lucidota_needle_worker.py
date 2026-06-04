#!/usr/bin/env python3
"""Tiny HTTP worker for resident Needle tool-call router lanes.

Default target for the edge topology is one process with six lanes. That shares
one Needle weight load and exposes batched generation over identical rolling
500-token chunks. Current Needle code batches encoder/decode work; exact
tensor-level K/V de-duplication for identical prefixes is the next runner
optimization, not something the upstream checkpoint API exposes directly.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_REPOS" / "needle"))

from needle import SimpleAttentionNetwork, generate, generate_batch, get_tokenizer, load_checkpoint  # noqa: E402

MODEL = None
PARAMS = None
TOKENIZER = None
STARTED = time.time()
INSTANCE = "needle-0"
CHECKPOINT = ""
SLOTS = 1
KV_POLICY = "rolling_window_500_token_chunks"
KV_PROBE_RECEIPT = "05_OUTPUTS/runtime/needle_kv_probe_latest.json"

class Handler(BaseHTTPRequestHandler):
    server_version = "LucidotaNeedle/0.1"

    def log_message(self, fmt, *args):
        return

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload, sort_keys=True).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            self._json(200, {
                "ok": True,
                "instance": INSTANCE,
                "model": "needle-26m",
                "checkpoint": CHECKPOINT,
                "shared_server": SLOTS > 1,
                "slots": SLOTS,
                "kv_policy": KV_POLICY,
                "kv_truth": "single process/shared weights with batched identical-prefix work; exact tensor KV de-dup is runner-extension target",
                "kv_probe_receipt": KV_PROBE_RECEIPT,
                "exact_tensor_kv_pointer_sharing_currently_proven": False,
                "prefix_reuse_for_different_lane_tools_requires_refactor": True,
                "uptime_s": round(time.time() - STARTED, 3),
                "endpoints": ["/health", "/generate", "/generate_batch"],
            })
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path not in ("/generate", "/generate_batch"):
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            n = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/generate_batch":
                lanes = payload.get("lanes")
                if lanes is not None:
                    if not isinstance(lanes, list):
                        raise ValueError("lanes must be a list")
                    queries = [str(item.get("query") or "") for item in lanes if isinstance(item, dict)]
                    tools_list = []
                    for item in lanes:
                        tools = item.get("tools", "[]") if isinstance(item, dict) else "[]"
                        tools_list.append(tools if isinstance(tools, str) else json.dumps(tools, separators=(",", ":")))
                else:
                    queries = [str(q) for q in payload.get("queries", [])]
                    raw_tools = payload.get("tools_list", ["[]"] * len(queries))
                    tools_list = [t if isinstance(t, str) else json.dumps(t, separators=(",", ":")) for t in raw_tools]
                if not queries:
                    raise ValueError("generate_batch requires lanes or queries")
                if len(queries) > SLOTS:
                    raise ValueError(f"requested {len(queries)} lanes exceeds configured slots={SLOTS}")
                max_gen_len = int(payload.get("max_gen_len") or 128)
                out = generate_batch(
                    MODEL,
                    PARAMS,
                    TOKENIZER,
                    queries,
                    tools_list,
                    max_gen_len=max_gen_len,
                )
                self._json(200, {
                    "ok": True,
                    "instance": INSTANCE,
                    "model": "needle-26m",
                    "shared_server": SLOTS > 1,
                    "slots": SLOTS,
                    "kv_policy": KV_POLICY,
                    "outputs": out,
                })
                return
            query = str(payload.get("query") or "")
            tools = payload.get("tools", "[]")
            if not isinstance(tools, str):
                tools = json.dumps(tools, separators=(",", ":"))
            max_gen_len = int(payload.get("max_gen_len") or 128)
            out = generate(MODEL, PARAMS, TOKENIZER, query=query, tools=tools, max_gen_len=max_gen_len, stream=False)
            self._json(200, {"ok": True, "instance": INSTANCE, "model": "needle-26m", "output": out})
        except Exception as exc:
            self._json(500, {"ok": False, "instance": INSTANCE, "error": str(exc)[:500]})


def main() -> int:
    global MODEL, PARAMS, TOKENIZER, INSTANCE, CHECKPOINT, SLOTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--instance", default=None)
    ap.add_argument("--checkpoint", default=str(ROOT / "03_VAULT" / "models" / "needle" / "needle.pkl"))
    ap.add_argument("--slots", type=int, default=1)
    args = ap.parse_args()
    INSTANCE = args.instance or f"needle-{args.port}"
    CHECKPOINT = str(Path(args.checkpoint))
    SLOTS = max(1, int(args.slots))
    PARAMS, config = load_checkpoint(CHECKPOINT)
    MODEL = SimpleAttentionNetwork(config)
    TOKENIZER = get_tokenizer()
    print(json.dumps({"ok": True, "event": "loaded", "instance": INSTANCE, "port": args.port, "slots": SLOTS, "checkpoint": CHECKPOINT}), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
