#!/usr/bin/env python3
"""KORPUS low-level text math helpers: minhash, entropy, CKDOG vector literals."""
from __future__ import annotations

import re

from ALGOS.minhash import shingles, signature
from ALGOS.shannon_entropy import shannon_entropy
from kernel.mini_embeddings import INT16_MAX, hash_quantized_embedding


def minhash_for_text(text: str, k: int = 64) -> list[int]:
    return signature(shingles(re.sub(r"\s+", " ", text or "").strip().lower(), width=5), k=k)


def entropy_for_text(text: str) -> float:
    return float(shannon_entropy(list((text or "")[:10000]))) if text else 0.0


def vector_literal(text: str) -> str:
    return "[" + ",".join(f"{float(v) / float(INT16_MAX):.8f}" for v in hash_quantized_embedding(text or "")) + "]"
