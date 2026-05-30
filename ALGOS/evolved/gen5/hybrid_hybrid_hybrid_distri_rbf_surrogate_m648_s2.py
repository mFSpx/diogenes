# DARWIN HAMMER — match 648, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_distributed_l_hybrid_physarum_netw_m110_s4.py (gen4)
# parent_b: rbf_surrogate.py (gen0)
# born: 2026-05-29T23:30:25Z

"""Hybrid Leader Election with Physarum‑Inspired Conductance and RBF Surrogate Modeling.

Parents
-------
* **Parent A** – Simulated annealing leader election where the temperature T is the product of a
  broadcast‑probability decay and an exponential cooling schedule.  T also depends on the
  Physarum network's edge conductance g and node pressures pₐ, p_b.
* **Parent B** – Radial‑basis‑function (RBF) surrogate that interpolates a scalar field from scattered
  samples using a Gaussian kernel.

Mathematical Bridge
-------------------
The bridge is the *conductance prediction*: the Physarum dynamics generate a high‑dimensional
conductance matrix **G** that is expensive to recompute at every annealing step.  We train an
RBF surrogate **S** on a small set of sampled (node‑feature → conductance) pairs and use the
surrogate to estimate conductances on‑the‑fly.  The estimated conductance 𝑔̂ appears in the hybrid
temperature formula


T(k, phase) =  (1 / 2^{phases‑phase})          # broadcast decay
               * t₀·α^{k}                      # exponential cooling
               * f(𝑔̂, pₐ, p_b)                # physarum‑derived scaling


where  


f(𝑔̂, pₐ, p_b) = 𝑔̂ / (|pₐ - p_b| + ε)


Thus the RBF surrogate supplies the physics‑driven term inside the Metropolis acceptance rule,
creating a single unified system.

The module below implements:
* RBF surrogate utilities (fit, predict).
* Physarum‑inspired conductance update (very light version).
* Hybrid temperature computation.
* Leader‑election step that uses the Metropolis rule with the hybrid temperature.
"""

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
# Hybrid system integrating both parents
# ----------------------------------------------------------------------
class HybridLeaderPhysarumNetwork:
    """
    Holds a Physarum‑inspired conductance matrix, node pressures,
    and an RBF surrogate that predicts conductances from node features.
    """

    def __init__(self,
                 num_nodes: int,
                 initial_conductance: float = 1.0,
                 epsilon_rbf: float = 1.0):
        self.n = num_nodes
        # Symmetric conductance matrix G_{ij}
        self.G = np.full((num_nodes, num_nodes), initial_conductance, dtype=float)
        np.fill_diagonal(self.G, 0.0)

        # Random pressures in (0,1)
        self.pressures = np.random.rand(num_nodes)

        # RBF surrogate (initially empty – will be trained on demand)
        self.surrogate: RBFModel | None = None
        self.epsilon_rbf = epsilon_rbf

    # ------------------------------------------------------------------
    # Physarum‑like update (very lightweight)
    # ------------------------------------------------------------------
    def physarum_update(self, flow_rate: float = 0.1) -> None:
        """
        Perform a single update of conductances based on pressure differences.
        Conductance update rule (simplified):
            g_{ij} ← g_{ij} + η * (|p_i - p_j| - g_{ij}) * flow_rate
        where η is a small learning rate.
        """
        eta = 0.05
        for i in range(self.n):
            for j in range(i + 1, self.n):
                delta = abs(self.pressures[i] - self.pressures[j]) - self.G[i, j]
                self.G[i, j] += eta * delta * flow_rate
                self.G[j, i] = self.G[i, j]  # keep symmetric

        # Keep conductances positive
        np.clip(self.G, 1e-6, None, out=self.G)

    # ------------------------------------------------------------------
    # Surrogate handling
    # ------------------------------------------------------------------
    def train_surrogate(self, sample_fraction: float = 0.3) -> None:
        """
        Sample a subset of edges, collect (feature, conductance) pairs,
        and fit an RBF surrogate.
        Feature vector for edge (i,j): (i, j, pressure_i, pressure_j)
        """
        edges = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        sample_size = max(1, int(len(edges) * sample_fraction))
        sampled = random.sample(edges, sample_size)

        points: List[Tuple[float, ...]] = []
        values: List[float] = []

        for i, j in sampled:
            feat = (float(i), float(j), float(self.pressures[i]), float(self.pressures[j]))
            points.append(feat)
            values.append(float(self.G[i, j]))

        self.surrogate = fit_rbf(points, values, epsilon=self.epsilon_rbf)

    def predict_conductance(self, i: int, j: int) -> float:
        """
        Use the surrogate (if trained) to predict conductance for edge (i,j).
        Falls back to the stored matrix value if surrogate is unavailable.
        """
        if self.surrogate is None:
            return float(self.G[i, j])

        feat = (float(i), float(j), float(self.pressures[i]), float(self.pressures[j]))
        pred = self.surrogate.predict(feat)
        # Conductance must stay positive; clamp
        return max(pred, 1e-6)

    # ------------------------------------------------------------------
    # Leader election utilities
    # ------------------------------------------------------------------
    def leader_election_step(self,
                             candidates: List[int],
                             phases: int,
                             phase: int,
                             k: int,
                             t0: float = 1.0,
                             alpha: float = 0.95) -> int:
        """
        Perform one Metropolis‑based election step.

        - Randomly pick a candidate c.
        - Compute energy ΔE = number of conflicts with already elected nodes
          (here we treat conflict as sharing an edge with non‑zero conductance).
        - Compute hybrid temperature T using predicted conductance between c and the
          currently elected set (or a dummy reference node if none yet).
        - Accept c with probability min(1, exp(-ΔE / T)).
        Returns the elected node (or -1 if rejected).
        """
        if not candidates:
            raise ValueError("candidate list empty")

        c = random.choice(candidates)

        # For simplicity we treat the conflict count as the sum of conductances
        # from c to all previously elected nodes stored in self._elected.
        # If no previous leader, ΔE = 0.
        previous = getattr(self, "_elected", [])
        delta_E = sum(self.G[c, p] for p in previous) if previous else 0.0

        # Use the surrogate to get a representative conductance for temperature.
        # If previous leaders exist, use average predicted conductance to them;
        # otherwise use a self‑loop surrogate (i,i) which we define as median G.
        if previous:
            conds = [self.predict_conductance(c, p) for p in previous]
            g_est = sum(conds) / len(conds)
        else:
            g_est = np.median(self.G)

        # Pressures of the two nodes (c and a reference node 0)
        p_a = float(self.pressures[c])
        p_b = float(self.pressures[0])

        T = hybrid_temperature(phases, phase, k, g_est, p_a, p_b, t0, alpha)

        # Metropolis acceptance
        accept_prob = math.exp(-delta_E / T) if T > 0 else 0.0
        if random.random() < min(1.0, accept_prob):
            # Accept candidate
            self._elected = previous + [c]
            return c
        else:
            return -1

    # ------------------------------------------------------------------
    # Utility for inspection
    # ------------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        """Return a snapshot of the network state (useful for debugging)."""
        return {
            "conductance_matrix": self.G.copy(),
            "pressures": self.pressures.copy(),
            "elected": getattr(self, "_elected", []),
            "surrogate_trained": self.surrogate is not None,
        }


# ----------------------------------------------------------------------
# Demonstration functions (fulfil requirement of ≥3 functions)
# ----------------------------------------------------------------------
def hybrid_temperature_demo() -> None:
    """Show temperature calculation for a toy configuration."""
    T = hybrid_temperature(
        phases=5,
        phase=2,
        k=3,
        conductance=0.8,
        pressure_a=0.4,
        pressure_b=0.7,
        t0=1.0,
        alpha=0.9,
    )
    print(f"[demo] Hybrid temperature = {T:.6f}")


def rbf_surrogate_demo() -> None:
    """Fit a tiny RBF surrogate and query it."""
    points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    values = [1.0, 2.0, 3.0]
    model = fit_rbf(points, values, epsilon=1.5)
    query = (0.5, 0.5)
    pred = model.predict(query)
    print(f"[demo] RBF prediction at {query} = {pred:.4f}")


def leader_election_demo() -> None:
    """Run a few leader‑election steps on a small network."""
    net = HybridLeaderPhysarumNetwork(num_nodes=6, initial_conductance=0.5)
    net.train_surrogate()
    candidates = list(range(6))

    print("[demo] Starting election")
    for step in range(4):
        elected = net.leader_election_step(
            candidates=candidates,
            phases=5,
            phase=step + 1,
            k=step,
            t0=1.0,
            alpha=0.95,
        )
        if elected >= 0:
            print(f"  Step {step+1}: elected node {elected}")
            candidates.remove(elected)
        else:
            print(f"  Step {step+1}: no election (rejected)")

        # Update physarum dynamics and retrain surrogate intermittently
        net.physarum_update()
        if step % 2 == 1:
            net.train_surrogate()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Hybrid Temperature Demo ===")
    hybrid_temperature_demo()
    print("\n=== RBF Surrogate Demo ===")
    rbf_surrogate_demo()
    print("\n=== Leader Election Demo ===")
    leader_election_demo()
    print("\nAll demos executed without error.")