#!/usr/bin/env python3
"""Sparse winner-take-all tags for high-dimensional similarity."""
from __future__ import annotations
import hashlib


def expand(values: list[float], m: int, salt: str = '') -> list[float]:
    if m <= 0:
        raise ValueError('m must be positive')
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f'{salt}:{i}:{r}'.encode()).digest()
            j = int.from_bytes(h[:8], 'big') % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: list[float], k: int) -> list[int]:
    k = max(0, min(k, len(values)))
    winners = {i for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]}
    return [1 if i in winners else 0 for i in range(len(values))]


def hamming(a: list[int], b: list[int]) -> int:
    if len(a) != len(b):
        raise ValueError('vectors must be same length')
    return sum(x != y for x, y in zip(a, b))
