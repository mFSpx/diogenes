#!/usr/bin/env python3
"""ModernBERT-base NER/encoding server via ONNX runtime on :8084.

Exposes POST /v1/embeddings (OpenAI-compat) using the int8 ONNX variant.
Run: python scripts/lucidota_modernbert_onnx_server.py [--port 8084]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "04_RUNTIME" / "models" / "modernbert-base"
ONNX_PATH = MODEL_DIR / "onnx" / "model_int8.onnx"

_session: ort.InferenceSession | None = None
_tokenizer = None


def load_model() -> None:
    global _session, _tokenizer
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    _session = ort.InferenceSession(str(ONNX_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    print(f"[modernbert] loaded {ONNX_PATH}", flush=True)


def cls_embed(texts: list[str]) -> list[list[float]]:
    enc = _tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    inputs = {k: enc[k].astype(np.int64) for k in [i.name for i in _session.get_inputs()]}
    out = _session.run(None, inputs)[0]  # (batch, seq, hidden)
    cls = out[:, 0, :]  # CLS token
    norms = np.linalg.norm(cls, axis=1, keepdims=True)
    return (cls / np.maximum(norms, 1e-9)).tolist()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass  # silence access logs

    def do_POST(self):
        if self.path not in ("/v1/embeddings", "/embeddings"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        texts = body.get("input", [])
        if isinstance(texts, str):
            texts = [texts]
        vecs = cls_embed(texts)
        resp = {
            "object": "list",
            "model": "modernbert-base-int8",
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
            "usage": {"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": sum(len(t.split()) for t in texts)},
        }
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8084)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    load_model()
    print(f"[modernbert] serving on {args.host}:{args.port}", flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
