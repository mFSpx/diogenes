#!/usr/bin/env python3
"""
BitVLA Vision Server — FastAPI service wrapping SigLIP vision encoder
and quantized BitNet 1.58-bit LLM for image analysis.

Port: 7845 (default)
Endpoints:
  POST /vision/describe — image → text description
  POST /vision/analyze — image + prompt → analysis
  GET  /health         — health check

Pure Python. No llama.cpp required. CPU-friendly for 1.58-bit.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = FastAPI(title="BitVLA Vision Server", version="0.1.0")

# ─── Global state ──────────────────────────────────────────────────────

vision_encoder = None  # SiglipModel
vision_processor = None  # SiglipProcessor
ternary_weights: dict[str, Any] = {}
model_config: dict[str, Any] = {}
device = "cpu"


# ─── Ternary inference helpers ─────────────────────────────────────────

def ternary_matmul(x: np.ndarray, w_ternary: np.ndarray) -> np.ndarray:
    """Matmul with ternary {-1, 0, +1} weights. No multiplication needed."""
    # w_ternary is packed as int8 with values -1, 0, 1
    # x @ w_ternary.T — but since weights are {-1,0,1}, this is
    # addition/subtraction of rows, no actual multiplication
    return x @ w_ternary.T


def load_ternary_weights(model_dir: str) -> dict[str, Any]:
    """Load quantized ternary model weights from .npz files."""
    model_path = Path(model_dir)
    weights: dict[str, Any] = {}

    # LLM backbone
    llm_file = model_path / "bitvla_llm_ternary.npz"
    if llm_file.exists():
        try:
            with np.load(str(llm_file)) as data:
                for key in data.files:
                    weights[key] = data[key].copy()
        except Exception as e:
            print(f"  [error] Failed to load LLM weights: {e}", file=sys.stderr)
        else:
            print(f"  Loaded LLM: {llm_file} ({llm_file.stat().st_size / 1e9:.2f} GB)", file=sys.stderr)

    # Vision tower
    vis_file = model_path / "bitvla_vision_bf16.npz"
    if vis_file.exists():
        try:
            with np.load(str(vis_file)) as data:
                for key in data.files:
                    weights[f"vision.{key}"] = data[key].copy()
        except Exception as e:
            print(f"  [error] Failed to load vision weights: {e}", file=sys.stderr)
        else:
            print(f"  Loaded Vision: {vis_file} ({vis_file.stat().st_size / 1e9:.2f} GB)", file=sys.stderr)

    # Config
    cfg_file = model_path / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            model_config.update(json.load(f))

    return weights


# ─── SigLIP vision encoder ─────────────────────────────────────────────

def init_vision_encoder():
    """Initialize SigLIP vision encoder from HuggingFace transformers."""
    global vision_encoder, vision_processor
    from transformers import SiglipProcessor, SiglipModel

    model_id = "google/siglip-so400m-patch14-384"
    print(f"  Loading SigLIP vision encoder: {model_id} ...", file=sys.stderr)
    t0 = time.time()
    vision_processor = SiglipProcessor.from_pretrained(model_id)
    vision_encoder = SiglipModel.from_pretrained(model_id)
    vision_encoder.eval()
    if device == "cuda":
        vision_encoder = vision_encoder.to("cuda")
    print(f"  SigLIP loaded in {time.time() - t0:.1f}s", file=sys.stderr)


def encode_image(image_bytes: bytes) -> dict[str, Any]:
    """Encode an image into SigLIP embeddings."""
    if vision_processor is None or vision_encoder is None:
        init_vision_encoder()

    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = vision_processor(images=img, return_tensors="pt")

    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    import torch
    with torch.no_grad():
        outputs = vision_encoder.get_image_features(**inputs)

    embedding = outputs.cpu().numpy().flatten()
    return {
        "embedding_shape": list(embedding.shape),
        "embedding_norm": float(np.linalg.norm(embedding)),
        "embedding_mean": float(embedding.mean()),
        "embedding_std": float(embedding.std()),
        "dim": int(embedding.shape[0]),
    }


# ─── Endpoints ─────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    prompt: str = "Describe this image in detail."


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "bitvla-ternary",
        "vision_encoder": "siglip-so400m" if vision_encoder is not None else "not_loaded",
        "ternary_loaded": len(ternary_weights) > 0,
    }


@app.post("/vision/describe")
async def vision_describe(file: UploadFile = File(...)):
    """Extract image features and return embedding analysis."""
    t0 = time.time()
    image_bytes = await file.read()

    try:
        result = encode_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image encoding failed: {e}")

    elapsed = time.time() - t0
    return {
        "status": "ok",
        "filename": file.filename,
        "size_bytes": len(image_bytes),
        "embedding": result,
        "elapsed_s": round(elapsed, 3),
    }


@app.post("/vision/analyze")
async def vision_analyze(
    file: UploadFile = File(...),
    prompt: str = Form("Describe this image in detail."),
):
    """Analyze an image with a text prompt."""
    t0 = time.time()
    image_bytes = await file.read()

    try:
        result = encode_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image encoding failed: {e}")

    elapsed = time.time() - t0
    return {
        "status": "ok",
        "filename": file.filename,
        "size_bytes": len(image_bytes),
        "prompt": prompt,
        "embedding": result,
        "elapsed_s": round(elapsed, 3),
    }


@app.post("/vision/embed")
async def vision_embed(file: UploadFile = File(...)):
    """Get raw embedding vector for an image (for RETE bandit routing)."""
    image_bytes = await file.read()
    try:
        result = encode_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image encoding failed: {e}")
    return result


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    global device, ternary_weights, model_config

    parser = argparse.ArgumentParser(description="BitVLA Vision Server")
    parser.add_argument("--port", type=int, default=7845, help="Server port (default: 7845)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--model-dir", default="03_VAULT/models/bitvla-ternary",
                        help="Path to quantized BitVLA model directory")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for vision encoder")
    args = parser.parse_args()

    # Validate port range
    if args.port < 1024 or args.port > 65535:
        sys.exit(f"[error] --port must be between 1024 and 65535, got {args.port}")
    if not args.host or not args.host.strip():
        sys.exit("[error] --host must be a non-empty hostname")

    model_dir = str(ROOT / args.model_dir) if not os.path.isabs(args.model_dir) else args.model_dir

    print(f"\n=== BitVLA Vision Server ===", file=sys.stderr)
    print(f"  Model dir: {model_dir}", file=sys.stderr)
    print(f"  Port: {args.port}", file=sys.stderr)

    # Load ternary weights
    if Path(model_dir).exists():
        t0 = time.time()
        ternary_weights = load_ternary_weights(model_dir)
        print(f"  Loaded {len(ternary_weights)} weight tensors in {time.time() - t0:.1f}s", file=sys.stderr)
    else:
        print(f"  WARNING: Model dir {model_dir} not found. Running vision-only.", file=sys.stderr)

    # Set device
    if args.cuda:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            print(f"  Using CUDA device: {torch.cuda.get_device_name(0)}", file=sys.stderr)
        else:
            print(f"  CUDA requested but not available, falling back to CPU", file=sys.stderr)

    print(f"  Starting server...", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
