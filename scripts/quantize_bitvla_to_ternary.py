#!/usr/bin/env python3
"""
Quantize BitVLA BF16 master weights to 1.58-bit ternary.

Splits the model into two files matching the paper's 1.4 GB memory figure:
  bitvla_llm_ternary.npz  — LLM backbone at 1.58-bit (target: ~0.7 GB)
  bitvla_vision_bf16.npz  — Vision encoder + projector at BF16 (~0.6 GB)

The paper's 1.4 GB = LLM backbone (0.55 GB packed) + embeddings/lm_head
(Q8_0 at ~0.16 GB) + activations/KV cache overhead at runtime. The vision
tower runs separately and isn't counted in the 1.4 GB inference budget.

Usage:
    python scripts/quantize_bitvla_to_ternary.py \
        --model-dir 03_VAULT/models/bitvla \
        --output-dir 03_VAULT/models/bitvla-ternary
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from safetensors.torch import load_file


# ─── Quantization functions ────────────────────────────────────────

def absmean_quantize(weight: torch.Tensor):
    """Ternary {-1, 0, +1} via absmean scaling. One step per tensor."""
    w = weight.float()
    step = w.abs().mean().clamp(min=1e-5)
    q = (w / step).round().clamp(-1, 1)
    return q, step


def pack_ternary_to_int2(q_weight: torch.Tensor):
    """Pack ternary {-1,0,+1} → 2-bit packed uint8. 4 values per byte."""
    q = q_weight.to(torch.int8) + 1
    q_flat = q.flatten().to(torch.uint8)
    n_elems = q_flat.numel()
    remainder = n_elems % 4
    if remainder:
        q_flat = torch.cat([q_flat, torch.zeros(4 - remainder, dtype=torch.uint8)])
    q_flat = q_flat.view(-1, 4)
    packed = q_flat[:, 0] | (q_flat[:, 1] << 2) | (q_flat[:, 2] << 4) | (q_flat[:, 3] << 6)
    return packed, n_elems, q_weight.shape


def quantize_to_q8_0(weight: torch.Tensor):
    """Block-wise Q8_0 quantization: 32-weight blocks, fp16 scale per block."""
    w = weight.float()
    w_flat = w.flatten()
    n = w_flat.numel()
    block_size = 32
    num_blocks = (n + block_size - 1) // block_size
    pad = num_blocks * block_size - n
    if pad:
        w_flat = torch.cat([w_flat, torch.zeros(pad)])
    blocks = w_flat.view(num_blocks, block_size)
    scales = blocks.abs().max(dim=1).values.half()
    q = (blocks / scales.float().unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    return q.numpy(), scales.numpy(), n


# ─── Categorization ─────────────────────────────────────────────────

BITLINEAR_PATTERNS = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
VISION_PATTERNS = ["vision_tower", "vision_model", "siglip", "mm_projector",
                   "multi_modal_projector"]


def categorize(key: str) -> str:
    kl = key.lower()
    if any(p in kl for p in VISION_PATTERNS):
        return "vision"
    if "embed_tokens" in kl or "token_embed" in kl:
        return "embedding"
    if "lm_head" in kl:
        return "lm_head"
    if any(p in kl for p in BITLINEAR_PATTERNS):
        return "bitlinear"
    return "other"


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quantize BitVLA to 1.58-bit ternary")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir
    output_dir = args.output_dir

    print(f"Loading: {model_dir}/model.safetensors")
    t0 = time.time()
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    print(f"  Loaded {len(sd)} tensors in {time.time() - t0:.1f}s")

    # ── Categorize ──
    llm_tensors = {}
    vision_tensors = {}
    stats = {"bitlinear": 0, "embedding": 0, "lm_head": 0, "other": 0, "vision": 0}

    for key, tensor in sd.items():
        cat = categorize(key)
        stats[cat] += 1
        if cat == "vision":
            vision_tensors[key] = tensor
        else:
            llm_tensors[key] = tensor

    print(f"\nCategorization:")
    print(f"  LLM backbone:  {stats['bitlinear']} BitLinear mats + {stats['embedding']} embeds + {stats['lm_head']} lm_head + {stats['other']} other")
    print(f"  Vision tower:  {stats['vision']} tensors")

    # ── Quantize LLM ──
    print(f"\nQuantizing LLM backbone...")
    q_llm = {}
    llm_original = 0
    llm_packed = 0

    for key, tensor in llm_tensors.items():
        llm_original += tensor.numel() * tensor.element_size()
        cat = categorize(key)

        if cat == "bitlinear":
            q_vals, step = absmean_quantize(tensor)
            packed, n_elems, orig_shape = pack_ternary_to_int2(q_vals)
            q_llm[key + ".q_weight"] = packed.numpy()
            q_llm[key + ".w_step"] = np.float32(step)
            q_llm[key + ".orig_shape"] = np.array(orig_shape, dtype=np.int64)
            q_llm[key + ".n_elems"] = np.int64(n_elems)
            llm_packed += packed.numel() + 4

        elif cat in ("embedding", "lm_head"):
            q_int8, scales, n = quantize_to_q8_0(tensor)
            q_llm[key + ".q8_weight"] = q_int8
            q_llm[key + ".q8_scales"] = scales
            q_llm[key + ".q8_n"] = np.int64(n)
            llm_packed += q_int8.nbytes + scales.nbytes + 8

        else:
            q_llm[key] = tensor
            llm_packed += tensor.numel() * tensor.element_size()

    print(f"  LLM original: {llm_original/1e9:.2f} GB")
    print(f"  LLM packed:   {llm_packed/1e9:.2f} GB")
    print(f"  Compression:  {llm_original/max(llm_packed,1):.1f}x")

    # ── Vision (keep BF16) ──
    vision_original = sum(t.numel() * t.element_size() for t in vision_tensors.values())
    print(f"\nVision tower (kept BF16): {vision_original/1e9:.2f} GB")

    # ── Combined ──
    total = llm_packed + vision_original
    print(f"\nTotal on disk: {total/1e9:.2f} GB  (LLM {llm_packed/1e9:.2f} + Vision {vision_original/1e9:.2f})")
    print(f"Paper's 1.4 GB figure = LLM backbone + runtime overhead (activations/KV cache)")

    if args.dry_run:
        print("\n[Dry run — no files written]")
        return

    # ── Save ──
    os.makedirs(output_dir, exist_ok=True)

    def _to_np(v):
        if hasattr(v, 'numpy'):
            return v.float().numpy() if hasattr(v, 'dtype') and v.dtype == torch.bfloat16 else v.numpy()
        return v

    # LLM core
    llm_path = os.path.join(output_dir, "bitvla_llm_ternary.npz")
    np.savez_compressed(llm_path, **{k: _to_np(v) for k, v in q_llm.items()})
    llm_size = os.path.getsize(llm_path)
    print(f"\nSaved: {llm_path}  ({llm_size/1e9:.2f} GB)")

    # Vision tower (separate file)
    vis_path = os.path.join(output_dir, "bitvla_vision_bf16.npz")
    np.savez_compressed(vis_path, **{k: _to_np(v) for k, v in vision_tensors.items()})
    vis_size = os.path.getsize(vis_path)
    print(f"Saved: {vis_path}  ({vis_size/1e9:.2f} GB)")

    # Copy config files
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "preprocessor_config.json",
                  "processor_config.json", "generation_config.json",
                  "dataset_statistics.json"]:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(output_dir, fname))

    metadata = {
        "schema": "lucidota.bitvla_ternary.v1",
        "original_model": "lxsy/bitvla-bf16",
        "quantization": "1.58-bit ternary (absmean) for BitLinear, Q8_0 for embeddings/lm_head",
        "llm_file": os.path.basename(llm_path),
        "llm_size_bytes": llm_size,
        "vision_file": os.path.basename(vis_path),
        "vision_size_bytes": vis_size,
        "total_size_bytes": llm_size + vis_size,
        "llm_original_bytes": llm_original,
        "llm_packed_bytes": llm_packed,
        "vision_original_bytes": vision_original,
        "paper_1_4gb_rationale": "LLM backbone (packed) + runtime KV cache/activations; vision tower runs separately",
    }
    meta_path = os.path.join(output_dir, "quantization_info.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"LLM core:  {llm_path}  ({llm_size/1e9:.2f} GB)")
    print(f"Vision:    {vis_path}  ({vis_size/1e9:.2f} GB)")
    print(f"Combined:  {(llm_size+vis_size)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
