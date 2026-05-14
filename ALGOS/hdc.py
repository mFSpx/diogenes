#!/usr/bin/env python3
"""Hyperdimensional computing primitives using bipolar vectors."""
from __future__ import annotations
import hashlib, random
from typing import Iterable

Vector = list[int]

def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(hashlib.sha256(symbol.encode('utf-8')).digest()[:8], 'big')
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError('vectors must have equal length')
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: Iterable[Vector]) -> Vector:
    vecs = list(vectors)
    if not vecs:
        raise ValueError('at least one vector is required')
    dim = len(vecs[0])
    if any(len(v) != dim for v in vecs):
        raise ValueError('vectors must have equal length')
    sums = [0] * dim
    for v in vecs:
        for i, x in enumerate(v):
            sums[i] += x
    return [1 if x >= 0 else -1 for x in sums]

def permute(v: Vector, shifts: int = 1) -> Vector:
    if not v:
        return []
    s = shifts % len(v)
    return v[-s:] + v[:-s] if s else list(v)

def similarity(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError('vectors must have equal length')
    if not a:
        raise ValueError('vectors must not be empty')
    return sum(x * y for x, y in zip(a, b)) / len(a)
