# DARWIN HAMMER — match 2606, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_pheromone_inf_m894_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_path_s_path_signature_m501_s1.py (gen4)
# born: 2026-05-29T23:43:07Z

"""
This module integrates the concepts of pheromone signals and decay rates from the hybrid_hybrid_pheromone_inf_hybrid_pheromone_inf_m894_s0 algorithm with the path signature and lead-lag transformation from the hybrid_hybrid_hybrid_path_s_path_signature_m501_s1 algorithm.
The mathematical bridge between these two structures is the application of the lead-lag transformation to the pheromone signals, allowing the creation of a novel hybrid algorithm that adapts to changing environments and optimizes the search process.
"""

import numpy as np
import math
import random
import sys
import pathlib
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

    def apply_lead_lag_to_pheromones(self):
        for pheromone in self.pheromones.values():
            transformed_path = self.lead_lag_transform(np.array([pheromone.signal_value]))
            pheromone.signal_value = self.signature_level(transformed_path, 1)

    def update_pheromones(self):
        for pheromone in self.pheromones.values():
            pheromone.apply_decay()

    def calculate_pheromone_signature(self):
        pheromone_values = [pheromone.signal_value for pheromone in self.pheromones.values()]
        return self.signature_level(np.array(pheromone_values), 1)


if __name__ == "__main__":
    hybrid_system = HybridSystem()
    pheromone1 = PheromoneEntry("surface1", "signal1", 10.0, 100)
    pheromone2 = PheromoneEntry("surface2", "signal2", 20.0, 200)
    hybrid_system.pheromones[pheromone1.uuid] = pheromone1
    hybrid_system.pheromones[pheromone2.uuid] = pheromone2
    hybrid_system.apply_lead_lag_to_pheromones()
    hybrid_system.update_pheromones()
    print(hybrid_system.calculate_pheromone_signature())