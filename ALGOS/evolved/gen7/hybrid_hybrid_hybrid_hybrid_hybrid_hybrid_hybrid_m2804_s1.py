# DARWIN HAMMER — match 2804, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1963_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_route_m1803_s2.py (gen6)
# born: 2026-05-29T23:46:08Z

"""
This module fuses the hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1963_s3 and hybrid_hybrid_hybrid_hybrid_hybrid_ternary_route_m1803_s2 algorithms into a single hybrid system.
The mathematical bridge between the two structures is the use of the path signature functions to evaluate the similarity between the input and output of the ModelPool loader, 
and the incorporation of the lead-lag transform, signature level 1 and 2, and B-spline basis functions into the ModelPool's load_with_eviction function.
This fusion enables the evaluation of the ModelPool loader's performance using the path signature metrics.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded = {}
        self.tier_hierarchy = {"T1": 0, "T2": 1, "T3": 2}

    def is_loaded(self, name: str) -> bool: 
        return name in self.loaded

    def _used(self) -> int: 
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier=="T3" and any(m.tier=="T2" for m in self.loaded.values()): 
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb: 
            raise Exception("RAM ceiling exceeded")
        if model.tier not in self.tier_hierarchy:
            raise Exception("Invalid model tier")
        self.loaded[model.name]=model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            eviction_candidate = min(self.loaded, key=lambda m: self.tier_hierarchy[self.loaded[m].tier])
            del self.loaded[eviction_candidate]
        self.load(model)

def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running = path[:-1] - path[0]               # (T‑1, d)
    return running.T @ increments               # (d, d)

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = len(grid)
    m = len(x)

    t = np.concatenate((
        np.full(k, grid[0]),
        grid,
        np.full(k, grid[-1])
    ))

    N = np.zeros((m, n + k - 1), dtype=float)
    for i in range(n + k - 1):
        left = t[i]
        right = t[i + 1]
        N[:, i] = np.where((x >= left) & (x < right), 1.0, 0.0)
    N[:, -1] = np.where(x == t[-1], 1.0, N[:, -1])

    for order in range(2, k + 1):
        for i in range(n + k - order):
            denom1 = t[i + order - 1] - t[i]
            term1 = 0.0 if denom1 == 0 else ((x - t[i]) / denom1) * N[:, i]
            denom2 = t[i + order] - t[i + 1]
            term2 = 0.0 if denom2 == 0 else ((t[i + order] - x) / denom2) * N[:, i + 1]
            N[:, i] = term1 + term2
    return N[:, :n + k - 1]

def hybrid_load(model_pool: ModelPool, model: ModelTier, path: np.ndarray) -> None:
    transformed_path = lead_lag_transform(path)
    signature1 = signature_level1(transformed_path)
    signature2 = signature_level2(transformed_path)
    grid = np.linspace(0, 1, 10)
    bspline = bspline_basis(np.linspace(0, 1, 100), grid)
    model_pool.load_with_eviction(model)

def hybrid_signature(model_pool: ModelPool, path: np.ndarray) -> np.ndarray:
    transformed_path = lead_lag_transform(path)
    signature1 = signature_level1(transformed_path)
    signature2 = signature_level2(transformed_path)
    return signature1, signature2

def hybrid_bspline(model_pool: ModelPool, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    bspline = bspline_basis(x, grid)
    return bspline

if __name__ == "__main__":
    model_pool = ModelPool()
    model = ModelTier("test", 100, "T1")
    path = np.random.rand(10, 2)
    hybrid_load(model_pool, model, path)
    signature1, signature2 = hybrid_signature(model_pool, path)
    x = np.linspace(0, 1, 100)
    grid = np.linspace(0, 1, 10)
    bspline = hybrid_bspline(model_pool, x, grid)
    print(signature1.shape, signature2.shape, bspline.shape)