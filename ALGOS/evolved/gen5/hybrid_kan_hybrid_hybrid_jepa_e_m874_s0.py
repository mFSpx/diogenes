# DARWIN HAMMER — match 874, survivor 0
# gen: 5
# parent_a: kan.py (gen0)
# parent_b: hybrid_hybrid_jepa_energy_h_hybrid_hybrid_infota_m141_s0.py (gen4)
# born: 2026-05-29T23:31:19Z

"""
Hybrid algorithm combining the Kolmogorov-Arnold Networks (KAN) from kan.py 
and the energy-based model pool from hybrid_hybrid_jepa_energy_h_hybrid_hybrid_infota_m141_s0.py.
The mathematical bridge between the two structures is the use of the KAN's B-spline basis 
to model the energy consumption in the model pool, where the cost of selecting a model 
is represented by a univariate B-spline function.

This hybrid algorithm integrates the governing equations of both parents by using 
the B-spline basis from KAN to simulate the process of selecting a representative 
model from the model pool, where the cost of selecting a model is modeled by the 
energy consumption in the model pool.
"""

from __future__ import annotations
import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded = {}
        self._energy = 0.0

    def is_loaded(self, name: str) -> bool: 
        return name in self.loaded

    def _used(self) -> int: 
        return sum(m.ram_mb for m in self.loaded.values())

    def add_model(self, model: ModelTier) -> None:
        if model.tier=="T3" and any(m.tier=="T2" for m in self.loaded.values()): 
            self._energy += 1e10  # penalty for conflicting tiers
        if model.ram_mb + self._used() > self.ram_ceiling_mb: 
            self._energy += 1e6  # penalty for high memory usage
        self.loaded[model.name]=model

    def load(self, model: ModelTier) -> None:
        self._energy -= 1e4  # reward for loading a model
        self.add_model(model)

    def load_with_eviction(self, model: ModelTier) -> None:
        self._energy -= 1e3  # reward for evicting an old model
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            evicted_model = max(self.loaded, key=lambda m: self.loaded[m].ram_mb)
            self.loaded.pop(evicted_model)
            self._energy += 1e2  # penalty for evicting a model
        self.load(model)

    def free_energy(self) -> float:
        return self._energy

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Evaluate B-spline basis functions of order k at positions x.

    Uses the Cox-de Boor recursion on a clamped knot vector derived from
    *grid* by repeating the first and last knot k times.

    Parameters
    ----------
    x:    shape (N,) — evaluation points (should lie within grid range).
    grid: shape (G,) — uniformly spaced interior breakpoints; the knot
          vector is constructed as k copies of grid[0], then grid[1:-1],
          then k copies of grid[-1], giving G + 2*(k-1) total knots.
    k:    spline order (polynomial degree = k - 1).  Default 3 (cubic).

    Returns
    -------
    B: shape (N, G - 1) — one column per basis function.
       Number of basis functions = len(grid) - 1.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Build clamped knot vector: repeat boundary knots (k-1) times so that
    # the polynomial spans cleanly to the boundary.
    # Knot vector length: (k-1) + G + (k-1) = G + 2(k-1)
    t = np.concatenate((np.full(k, grid[0]), grid[1:-1], np.full(k, grid[-1])))
    n_basis = len(grid) - 1

    B = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        B[:, i] = cox_de_boor(x, t, i, k)

    return B

def cox_de_boor(x: np.ndarray, t: np.ndarray, i: int, k: int) -> np.ndarray:
    if k == 1:
        return np.where((x >= t[i]) & (x < t[i+1]), 1.0, 0.0)

    b1 = (x - t[i]) / (t[i+k-1] - t[i]) * cox_de_boor(x, t, i, k-1)
    b2 = (t[i+k] - x) / (t[i+k] - t[i+1]) * cox_de_boor(x, t, i+1, k-1)

    return b1 + b2

def kan_layer(x: np.ndarray, grid: np.ndarray, k: int = 3, n_out: int = 1) -> np.ndarray:
    B = bspline_basis(x, grid, k)
    weights = np.random.rand(B.shape[1], n_out)
    return np.dot(B, weights)

def hybrid_kan_model_pool(x: np.ndarray, grid: np.ndarray, k: int = 3, n_out: int = 1, 
                          ram_ceiling_mb: int = 6000) -> np.ndarray:
    model_pool = ModelPool(ram_ceiling_mb)
    B = bspline_basis(x, grid, k)

    for i in range(B.shape[1]):
        model_tier = ModelTier(f"model_{i}", 100, "T1")
        model_pool.load_with_eviction(model_tier)

    weights = np.random.rand(B.shape[1], n_out)
    output = np.dot(B, weights)

    energy = model_pool.free_energy()
    return output, energy

if __name__ == "__main__":
    x = np.linspace(0, 1, 100)
    grid = np.linspace(0, 1, 10)
    output, energy = hybrid_kan_model_pool(x, grid)
    print(output.shape)
    print(energy)