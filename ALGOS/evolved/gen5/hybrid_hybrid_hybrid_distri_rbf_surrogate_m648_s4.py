# DARWIN HAMMER — match 648, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_distributed_l_hybrid_physarum_netw_m110_s4.py (gen4)
# parent_b: rbf_surrogate.py (gen0)
# born: 2026-05-29T23:30:25Z

import math
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Callable, Sequence, Any
import numpy as np

Vector = Sequence[float]

def _gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def _euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def _solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular system in surrogate")
        m[col], m[pivot] = m[pivot], m[col]

        div = m[col][col]
        m[col] = [v / div for v in m[col]]

        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]

    return [row[-1] for row in m]

@dataclass(frozen=True)
class RBFModel:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        return sum(
            w * _gaussian(_euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

def fit_rbf(points: Iterable[Vector],
            values: Iterable[float],
            epsilon: float = 1.0,
            ridge: float = 1e-9) -> RBFModel:
    centers = [tuple(map(float, p)) for p in points]
    y = [float(v) for v in values]

    if not centers or len(centers) != len(y):
        raise ValueError("points and values must be non-empty and of equal length")

    n = len(centers)
    K = [
        [
            _gaussian(_euclidean(a, b), epsilon) + (ridge if i == j else 0.0)
            for j, b in enumerate(centers)
        ]
        for i, a in enumerate(centers)
    ]

    weights = _solve_linear(K, y)
    return RBFModel(centers, weights, epsilon)

def broadcast_probability(phases: int, phase: int) -> float:
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    if k < 0 or t0 < 0 or not (0.0 <= alpha <= 1.0):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)

def hybrid_temperature(phases: int,
                       phase: int,
                       k: int,
                       conductance: float,
                       pressure_a: float,
                       pressure_b: float,
                       t0: float = 1.0,
                       alpha: float = 0.95,
                       eps: float = 1e-12) -> float:
    p_decay = broadcast_probability(phases, phase)
    cool = cooling_temperature(k, t0, alpha)
    scaling = conductance / (abs(pressure_a - pressure_b) + eps)
    return p_decay * cool * scaling

class HybridLeaderPhysarumNetwork:
    def __init__(self,
                 num_nodes: int,
                 initial_conductance: float = 1.0,
                 epsilon_rbf: float = 1.0):
        self.n = num_nodes
        self.G = np.full((num_nodes, num_nodes), initial_conductance, dtype=float)
        np.fill_diagonal(self.G, 0.0)

        self.pressures = np.random.rand(num_nodes)

        self.surrogate: RBFModel | None = None
        self.epsilon_rbf = epsilon_rbf

        self.sampled_points: List[Tuple[float, ...]] = []
        self.sampled_conductances: List[float] = []

    def physarum_update(self, flow_rate: float = 0.1) -> None:
        eta = 0.05
        for i in range(self.n):
            for j in range(i + 1, self.n):
                delta = abs(self.pressures[i] - self.pressures[j]) - self.G[i, j]
                self.G[i, j] += eta * delta * flow_rate
                self.G[j, i] = self.G[i, j]

        np.clip(self.G, 1e-6, None, out=self.G)

    def train_surrogate(self, sample_fraction: float = 0.3) -> None:
        num_samples = int(sample_fraction * self.n * (self.n - 1) / 2)
        for _ in range(num_samples):
            i = random.randint(0, self.n - 1)
            j = random.randint(0, self.n - 1)
            if i == j:
                continue

            point = (self.pressures[i], self.pressures[j])
            conductance = self.G[i, j]

            self.sampled_points.append(point)
            self.sampled_conductances.append(conductance)

        self.surrogate = fit_rbf(self.sampled_points, self.sampled_conductances, self.epsilon_rbf)

    def predict_conductance(self, pressure_a: float, pressure_b: float) -> float:
        if self.surrogate is None:
            raise ValueError("Surrogate not trained")

        point = (pressure_a, pressure_b)
        return self.surrogate.predict(point)

    def hybrid_leading(self, phases: int, phase: int, k: int, t0: float = 1.0, alpha: float = 0.95, eps: float = 1e-12) -> float:
        predicted_conductance = self.predict_conductance(self.pressures[0], self.pressures[1])
        return hybrid_temperature(phases, phase, k, predicted_conductance, self.pressures[0], self.pressures[1], t0, alpha, eps)