# DARWIN HAMMER — match 2606, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_pheromone_inf_m894_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_path_s_path_signature_m501_s1.py (gen4)
# born: 2026-05-29T23:43:07Z

"""
This module fuses the pheromone-infused optimization (hybrid_hybrid_pheromone_inf_hybrid_pheromone_inf_m894_s0.py) 
with the path-signature-based transformation (hybrid_hybrid_hybrid_path_s_path_signature_m501_s1.py). 
The mathematical bridge between these two structures lies in the concept of signal decay and path transformations. 
By integrating pheromone signals with path signatures, we create a novel hybrid algorithm that adapts to changing environments 
and optimizes the search process through transformed paths.

The interface between the two algorithms is established through the use of pheromone signals to weight the path transformations. 
The pheromone signals are used to compute a decay factor, which is then applied to the path signature to obtain a weighted transformation.
"""

import numpy as np
import random
import math
from datetime import datetime, timezone

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
            before = entry.signal_value
            entry.apply_decay()
            rows.append((entry.uuid, before, entry.signal_value))
        return rows


class HybridSystem:
    def __init__(self):
        self.pheromones = {}

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

    def hybrid_signature(self, path, pheromone_entry: PheromoneEntry):
        transformed_path = self.lead_lag_transform(path)
        decay_factor = pheromone_entry.decay_factor()
        weighted_path = transformed_path * decay_factor
        return weighted_path

    def compute_pheromone_signal(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int):
        pheromone_entry = PheromoneEntry(surface_key, signal_kind, signal_value, half_life_seconds)
        PheromoneStore.add(pheromone_entry)
        return pheromone_entry

    def get_weighted_path(self, path, surface_key: str):
        pheromone_entries = PheromoneStore.get_by_surface(surface_key)
        if not pheromone_entries:
            return None
        pheromone_entry = random.choice(pheromone_entries)
        return self.hybrid_signature(path, pheromone_entry)


if __name__ == "__main__":
    hybrid_system = HybridSystem()
    path = np.random.rand(10, 2)
    surface_key = "example_surface"
    signal_kind = "example_signal"
    signal_value = 1.0
    half_life_seconds = 10

    pheromone_entry = hybrid_system.compute_pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds)
    weighted_path = hybrid_system.get_weighted_path(path, surface_key)
    print(weighted_path)