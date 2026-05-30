# DARWIN HAMMER — match 5649, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hybrid_pherom_m411_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_bayes__m2655_s1.py (gen4)
# born: 2026-05-30T00:03:46Z

import math
import numpy as np
import random
import sys
from pathlib import Path
from datetime import datetime, timezone

Vector = np.ndarray

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    return np.linalg.norm(a - b)

def solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(b)
    m = np.hstack((a, b[:, None]))
    for col in range(n):
        pivot = np.argmax(np.abs(m[:, col]))
        if np.abs(m[pivot, col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[[pivot, col]] = m[[col, pivot]]
        m[col] /= m[col, col]
        for row in range(n):
            if row == col:
                continue
            m[row] -= m[row, col] * m[col]
    return m[:, -1]

@dataclass(frozen=True)
class RBFSurrogate:
    centers: np.ndarray
    weights: np.ndarray
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        return np.dot(self.weights, gaussian(euclidean(x, self.centers), self.epsilon))

def fit(points: np.ndarray, values: np.ndarray, epsilon: float = 1.0, ridge: float = 1e-9) -> RBFSurrogate:
    centers = points
    y = values
    if len(centers) != len(y):
        raise ValueError("points and values must be same length")
    k = np.array([[gaussian(euclidean(a, b), epsilon) + (ridge if i == j else 0.0) for j, b in enumerate(centers)] for i, a in enumerate(centers)])
    return RBFSurrogate(centers, solve_linear(k, y), epsilon)

class PheromoneSystem:
    def __init__(self) -> None:
        self.pheromones: Dict[str, Dict[str, Any]] = {}

    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        now = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {}
        if signal_kind not in self.pheromones[surface_key]:
            self.pheromones[surface_key][signal_kind] = {
                'value': signal_value,
                'timestamp': now,
                'half_life_seconds': half_life_seconds
            }
        else:
            elapsed_seconds = (now - self.pheromones[surface_key][signal_kind]['timestamp']).total_seconds()
            decay_factor = 0.5 ** (elapsed_seconds / self.pheromones[surface_key][signal_kind]['half_life_seconds'])
            self.pheromones[surface_key][signal_kind]['value'] *= decay_factor
            self.pheromones[surface_key][signal_kind]['timestamp'] = now
        return self.pheromones[surface_key][signal_kind]['value']

class HybridSystem:
    def __init__(self):
        self.pheromones = {}
        self.records = []
        self.rbf_surrogates = {}

    def calculate_pheromone_signal_from_rbf(self, surface_key, signal_kind, signal_value, half_life_seconds):
        if surface_key not in self.rbf_surrogates:
            points = np.random.rand(100, 2)
            values = np.random.rand(100)
            self.rbf_surrogates[surface_key] = fit(points, values)
        rbf_value = self.rbf_surrogates[surface_key].predict(np.array([0.5, 0.5]))
        pheromone_value = PheromoneSystem().calculate_pheromone_signal(
            surface_key,
            signal_kind,
            signal_value * rbf_value,
            half_life_seconds
        )
        return pheromone_value

    def calculate_pheromone_signal_from_pheromone(self, surface_key, signal_kind, signal_value, half_life_seconds):
        return PheromoneSystem().calculate_pheromone_signal(
            surface_key,
            signal_kind,
            signal_value,
            half_life_seconds
        )

    def calculate_pheromone_signal_from_entropy(self, surface_key, signal_kind, signal_value, half_life_seconds):
        if surface_key not in self.rbf_surrogates:
            points = np.random.rand(100, 2)
            values = np.random.rand(100)
            self.rbf_surrogates[surface_key] = fit(points, values)
        rbf_value = self.rbf_surrogates[surface_key].predict(np.array([0.5, 0.5]))
        entropy = 0.5 * (1 - rbf_value)
        return PheromoneSystem().calculate_pheromone_signal(
            surface_key,
            signal_kind,
            signal_value * entropy,
            half_life_seconds
        )

if __name__ == "__main__":
    hybrid_system = HybridSystem()
    pheromone_value = hybrid_system.calculate_pheromone_signal_from_rbf("surface_key", "signal_kind", 1.0, 3600)
    print(pheromone_value)
    pheromone_value = hybrid_system.calculate_pheromone_signal_from_pheromone("surface_key", "signal_kind", 1.0, 3600)
    print(pheromone_value)
    pheromone_value = hybrid_system.calculate_pheromone_signal_from_entropy("surface_key", "signal_kind", 1.0, 3600)
    print(pheromone_value)