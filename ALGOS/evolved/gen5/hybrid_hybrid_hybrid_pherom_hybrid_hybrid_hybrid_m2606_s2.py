# DARWIN HAMMER — match 2606, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_pheromone_inf_m894_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_path_s_path_signature_m501_s1.py (gen4)
# born: 2026-05-29T23:43:07Z

"""
This module integrates the Darwinian surface pheromone worker (hybrid_pheromone_infotaxis_m3_s4) 
with the gradient-free entropy search helpers (hybrid_hybrid_hybrid_path_s_path_signature_m501_s1).
The mathematical bridge between these two structures is the concept of pheromone signals and their 
decay rates, which can be seen as a form of entropy optimization, and the path signature, 
which can be used to analyze the pheromone trails. By combining the pheromone signal system 
with the path signature analysis, we can create a novel hybrid algorithm that adapts to 
changing environments and optimizes the search process.
"""

import numpy as np
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(random.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Return the multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """Singleton-like in-memory store for demo purposes."""
    _entries: dict = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> list:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> list:
        rows = []
        for entry in cls.get_by_surface(surface_key):
            entry.apply_decay()
            rows.append(entry)
        return rows


class HybridSystem:
    def __init__(self):
        self.pheromones = {}
        self.records = []

    def lead_lag_transform(self, path):
        path = np.asarray(path, dtype=float)
        T, d = path.shape
        out = np.empty((2 * T - 1, 2 * d), dtype=float)
        for t in range(T - 1):
            out[2 * t]     = np.concatenate([path[t],     path[t]])
            out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
        out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
        return out

    def bspline_basis(self, x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        grid = np.asarray(grid, dtype=np.float64)

        t = np.concatenate([
            np.full(k - 1, grid[0]),
            grid,
            np.full(k - 1, grid[-1]),
        ])

        n_basis = len(grid) + k - 2      
        N = len(x)

        B = np.zeros((N, len(t) - 1), dtype=np.float64)
        for i in range(N):
            for j in range(len(t) - 1):
                if t[j] <= x[i] <= t[j + 1]:
                    if k == 1:
                        B[i, j] = 1.0
                    elif k == 2:
                        B[i, j] = (x[i] - t[j]) / (t[j + 1] - t[j])
                        B[i, j + 1] = (t[j + 2] - x[i]) / (t[j + 2] - t[j + 1])
                    elif k == 3:
                        h1 = (t[j + 1] - t[j]) * (t[j + 1] - x[i])**2 / ((t[j + 2] - t[j]) * (t[j + 1] - t[j]))
                        h2 = (x[i] - t[j]) * (t[j + 1] - x[i])**2 / ((t[j + 1] - t[j]) * (t[j + 1] - t[j - 1]))
                        h3 = (x[i] - t[j]) * (x[i] - t[j + 1])**2 / ((t[j + 1] - t[j]) * (t[j + 2] - t[j + 1]))
                        B[i, j] = h1
                        B[i, j + 1] = h2 + h3
        return B

    def signature_level(self, path, level):
        path = np.asarray(path, dtype=float)
        if level == 1:
            return path[-1] - path[0]
        elif level == 2:
            increments = np.diff(path, axis=0)          
            running    = path[:-1] - path[0]            
            return running.T @ increments               
        else:
            raise ValueError("Invalid signature level")

    def hybrid_signature(self, path, level):
        transformed_path = self.lead_lag_transform(path)
        return self.signature_level(transformed_path, level)

    def pheromone_signature(self, surface_key, level):
        pheromones = PheromoneStore.get_by_surface(surface_key)
        path = np.array([p.signal_value for p in pheromones])
        return self.signature_level(path, level)

    def hybrid_pheromone_signature(self, surface_key, level):
        pheromones = PheromoneStore.get_by_surface(surface_key)
        path = np.array([p.signal_value for p in pheromones])
        transformed_path = self.lead_lag_transform(path.reshape(-1, 1))
        return self.signature_level(transformed_path, level)

if __name__ == "__main__":
    system = HybridSystem()
    pheromone = PheromoneEntry("surface", "kind", 1.0, 10)
    PheromoneStore.add(pheromone)
    print(system.hybrid_pheromone_signature("surface", 1))
    print(system.pheromone_signature("surface", 1))
    print(system.hybrid_signature(np.array([[1.0], [2.0], [3.0]]), 1))