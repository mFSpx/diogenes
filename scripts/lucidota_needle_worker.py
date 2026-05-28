#!/usr/bin/env python3
"""Tiny HTTP worker for one resident Needle tool-call router instance."""
from __future__ import annotations
import argparse, json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_REPOS" / "needle"))

from needle import SimpleAttentionNetwork, generate, get_tokenizer, load_checkpoint  # noqa: E402

MODEL = None
PARAMS = None
TOKENIZER = None
STARTED = time.time()
INSTANCE = "needle-0"
CHECKPOINT = ""

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
                "uptime_s": round(time.time() - STARTED, 3),
                "endpoints": ["/health", "/generate"],
            })
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            n = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
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
    global MODEL, PARAMS, TOKENIZER, INSTANCE, CHECKPOINT
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--instance", default=None)
    ap.add_argument("--checkpoint", default=str(ROOT / "03_VAULT" / "models" / "needle" / "needle.pkl"))
    args = ap.parse_args()
    INSTANCE = args.instance or f"needle-{args.port}"
    CHECKPOINT = str(Path(args.checkpoint))
    PARAMS, config = load_checkpoint(CHECKPOINT)
    MODEL = SimpleAttentionNetwork(config)
    TOKENIZER = get_tokenizer()
    print(json.dumps({"ok": True, "event": "loaded", "instance": INSTANCE, "port": args.port, "checkpoint": CHECKPOINT}), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
