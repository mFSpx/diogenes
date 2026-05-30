# DARWIN HAMMER — match 648, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_distributed_l_hybrid_physarum_netw_m110_s4.py (gen4)
# parent_b: rbf_surrogate.py (gen0)
# born: 2026-05-29T23:30:25Z

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable, Callable, Sequence, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent B – RBF surrogate utilities (re‑implemented)
# ----------------------------------------------------------------------
Vector = Sequence[float]

def _gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def _euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def _solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b with Gaussian elimination (no external libs)."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for col in range(n):
        # Pivot
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular system in surrogate")
        m[col], m[pivot] = m[pivot], m[col]

        # Normalize pivot row
        div = m[col][col]
        m[col] = [v / div for v in m[col]]

        # Eliminate other rows
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]

    return [row[-1] for row in m]

@dataclass(frozen=True)
class RBFModel:
    """Radial‑basis‑function surrogate."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Evaluate surrogate at point x."""
        return sum(
            w * _gaussian(_euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

def fit_rbf(points: Iterable[Vector],
            values: Iterable[float],
            epsilon: float = 1.0,
            ridge: float = 1e-9) -> RBFModel:
    """Fit an RBF surrogate to (points, values)."""
    centers = [tuple(map(float, p)) for p in points]
    y = [float(v) for v in values]

    if not centers or len(centers) != len(y):
        raise ValueError("points and values must be non‑empty and of equal length")

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


# ----------------------------------------------------------------------
# Parent A – Simulated annealing + Physarum primitives (re‑implemented)
# ----------------------------------------------------------------------
def broadcast_probability(phases: int, phase: int) -> float:
    """Probability decay p = 1 / 2^{phases‑phase}, clamped to [0,1]."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Exponential cooling schedule."""
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
    """
    Combined temperature:
        T = broadcast_prob * cooling_temp * f(conductance, pressures)

    f(g, pₐ, p_b) = g / (|pₐ - p_b| + eps)
    """
    p_decay = broadcast_probability(phases, phase)
    cool = cooling_temperature(k, t0, alpha)
    scaling = conductance / (abs(pressure_a - pressure_b) + eps)
    return p_decay * cool * scaling

# ----------------------------------------------------------------------
# Improved Hybrid system integrating both parents
# ----------------------------------------------------------------------
class HybridLeaderPhysarumNetwork:
    """
    Holds a Physarum‑inspired conductance matrix, node pressures,
    and an RBF surrogate that predicts conductances from node features.
    """

    def __init__(self,
                 num_nodes: int,
                 initial_conductance: float = 1.0,
                 epsilon_rbf: float = 1.0,
                 learning_rate: float = 0.05,
                 flow_rate: float = 0.1):
        self.n = num_nodes
        # Symmetric conductance matrix G_{ij}
        self.G = np.full((num_nodes, num_nodes), initial_conductance, dtype=float)
        np.fill_diagonal(self.G, 0.0)

        # Random pressures in (0,1)
        self.pressures = np.random.rand(num_nodes)

        # RBF surrogate (initially empty – will be trained on demand)
        self.surrogate: RBFModel | None = None
        self.epsilon_rbf = epsilon_rbf
        self.eta = learning_rate
        self.flow_rate = flow_rate

    # ------------------------------------------------------------------
    # Physarum‑like update (very lightweight)
    # ------------------------------------------------------------------
    def physarum_update(self) -> None:
        """
        Perform a single update of conductances based on pressure differences.
        Conductance update rule (simplified):
            g_{ij} ← g_{ij} + η * (|p_i - p_j| - g_{ij}) * flow_rate
        where η is a small learning rate.
        """
        for i in range(self.n):
            for j in range(i + 1, self.n):
                delta = abs(self.pressures[i] - self.pressures[j]) - self.G[i, j]
                self.G[i, j] += self.eta * delta * self.flow_rate
                self.G[j, i] = self.G[i, j]  # keep symmetric

        # Keep conductances positive
        np.clip(self.G, 1e-6, None, out=self.G)

    # ------------------------------------------------------------------
    # Surrogate handling
    # ------------------------------------------------------------------
    def train_surrogate(self, sample_fraction: float = 0.3) -> None:
        """
        Sample a subset of edges, collect (feature, conductance) pairs
        """
        if self.surrogate is not None:
            return

        num_samples = int(sample_fraction * self.n * (self.n - 1) / 2)
        indices = np.random.choice(self.n, size=(num_samples, 2), replace=False)
        features = [(self.pressures[i], self.pressures[j]) for i, j in indices]
        conductances = [self.G[i, j] for i, j in indices]

        self.surrogate = fit_rbf(features, conductances, epsilon=self.epsilon_rbf)

    def predict_conductance(self, pressure_a: float, pressure_b: float) -> float:
        if self.surrogate is None:
            raise ValueError("surrogate not trained")
        return self.surrogate.predict((pressure_a, pressure_b))

    # ------------------------------------------------------------------
    # Leader election step
    # ------------------------------------------------------------------
    def leader_election_step(self,
                            phases: int,
                            phase: int,
                            k: int,
                            t0: float = 1.0,
                            alpha: float = 0.95,
                            eps: float = 1e-12) -> float:
        self.train_surrogate()
        conductance = self.predict_conductance(self.pressures[0], self.pressures[1])
        temperature = hybrid_temperature(phases, phase, k, conductance, self.pressures[0], self.pressures[1], t0, alpha, eps)
        return temperature