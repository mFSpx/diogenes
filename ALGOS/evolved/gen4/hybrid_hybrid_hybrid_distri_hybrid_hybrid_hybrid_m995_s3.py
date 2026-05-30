# DARWIN HAMMER — match 995, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_distributed_l_hybrid_model_vram_sc_m95_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_ternar_m374_s1.py (gen3)
# born: 2026-05-29T23:32:11Z

"""
HybridGraphPheromoneSystem
--------------------------

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – builds a perceptual‑hash based similarity graph and treats the
  adjacency matrix as a weight matrix that can be updated by gradient‑style
  operations while taking VRAM constraints into account.

* **Parent B** – maintains pheromone signals that decay over time and are used
  to bias a multi‑armed bandit (MAB) exploration/exploitation process.  Entropy
  of the probability distribution over actions is also computed.

**Mathematical bridge**

The bridge is the adjacency matrix **A** of the similarity graph.  Each edge
(i, j) is associated with a pheromone value ϕᵢⱼ.  The matrix of pheromones **Φ**
is element‑wise multiplied with **A** to obtain a *pheromone‑weighted* graph
**W = A ∘ Φ**.  Bandit arm values are derived from the row‑sums of **W**
(i.e. pheromone‑biased degrees).  Updates to **Φ** follow the exponential decay
model of Parent B, while a simple gradient‑descent step on **W** (as in Parent A)
is used to steer the leader election toward nodes with higher VRAM‑aware
broadcast probability.

The resulting system can:
1. Build the perceptual similarity graph.
2. Maintain and decay pheromones on edges.
3. Use a bandit‑style selection of a leader node based on pheromone‑weighted
   degrees, constrained by an estimated VRAM budget.

All operations are expressed with NumPy matrices and pure‑Python logic.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
import numpy as np
from datetime import datetime, timezone
from collections.abc import Mapping, Hashable

# ----------------------------------------------------------------------
# Helper functions – shared by both parent topologies
# ----------------------------------------------------------------------
def compute_phash(values: list[float]) -> int:
    """Perceptual hash: 1 bit per element indicating >= average (max 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two integers."""
    return (a ^ b).bit_count()


def broadcast_probability(phase: int, step: int) -> float:
    """Probability p = 1 / 2^{phase-step}, clamped to [0, 1]."""
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


# ----------------------------------------------------------------------
# Graph construction (Parent A)
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]


def build_graph(elements: list[list[float]]) -> Graph:
    """
    Build an undirected similarity graph.
    Nodes are stringified indices; an edge exists when the perceptual hash
    Hamming distance ≤ 4.
    """
    n = len(elements)
    hashes = {str(i): compute_phash(el) for i, el in enumerate(elements)}
    graph: dict[str, set[str]] = {str(i): set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(hashes[str(i)], hashes[str(j)]) <= 4:
                graph[str(i)].add(str(j))
                graph[str(j)].add(str(i))
    return graph


def adjacency_matrix(graph: Graph) -> np.ndarray:
    """Return the adjacency matrix A (shape N×N) for a graph with string keys."""
    nodes = sorted(graph.keys(), key=int)
    index = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=np.float64)
    for node, neighbors in graph.items():
        i = index[node]
        for nb in neighbors:
            j = index[nb]
            A[i, j] = 1.0
    return A, nodes


# ----------------------------------------------------------------------
# Pheromone handling (Parent B)
# ----------------------------------------------------------------------
class PheromoneStore:
    """Maintain pheromone values on graph edges with exponential decay."""

    def __init__(self, half_life_seconds: float = 60.0):
        self.half_life = half_life_seconds
        self._store: dict[tuple[int, int], float] = {}
        self._timestamps: dict[tuple[int, int], datetime] = {}

    def _key(self, i: int, j: int) -> tuple[int, int]:
        return (min(i, j), max(i, j))

    def add(self, i: int, j: int, value: float) -> None:
        """Add (or reset) pheromone on edge (i, j) to `value`."""
        k = self._key(i, j)
        now = datetime.now(timezone.utc)
        self._store[k] = value
        self._timestamps[k] = now

    def get(self, i: int, j: int) -> float:
        """Return decayed pheromone value for edge (i, j)."""
        k = self._key(i, j)
        if k not in self._store:
            return 0.0
        elapsed = (datetime.now(timezone.utc) - self._timestamps[k]).total_seconds()
        decay_factor = 0.5 ** (elapsed / self.half_life)
        return self._store[k] * decay_factor

    def matrix(self, size: int) -> np.ndarray:
        """Return the full N×N pheromone matrix Φ (symmetric, zeros on diagonal)."""
        Φ = np.zeros((size, size), dtype=np.float64)
        for (i, j), val in self._store.items():
            Φ[i, j] = Φ[j, i] = self.get(i, j)
        return Φ


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def pheromone_weighted_graph(A: np.ndarray, Φ: np.ndarray) -> np.ndarray:
    """
    Element‑wise multiplication of adjacency matrix A with pheromone matrix Φ.
    The result W = A ∘ Φ is the pheromone‑biased connectivity matrix.
    """
    return np.multiply(A, Φ)


def gradient_step(W: np.ndarray, target: np.ndarray, lr: float = 0.01) -> np.ndarray:
    """
    Simple gradient descent step to move W toward a target matrix.
    Loss L = ||W - target||_F^2, gradient = 2*(W - target).
    """
    grad = 2.0 * (W - target)
    return W - lr * grad


def estimate_vram_usage(matrix: np.ndarray) -> float:
    """Estimate VRAM in megabytes for a float64 matrix."""
    bytes_per_elem = matrix.itemsize
    total_bytes = matrix.size * bytes_per_elem
    return total_bytes / (1024.0 ** 2)  # MB


def select_leader_bandit(W: np.ndarray, phase: int, step: int) -> int:
    """
    Multi‑armed bandit selection of a leader node.
    - Arm values are the pheromone‑biased degrees (row sums of W).
    - Probabilities are proportional to values raised to an exploration factor
      derived from broadcast_probability().
    - Returns the index of the selected node.
    """
    degrees = W.sum(axis=1)
    if np.allclose(degrees, 0):
        # fallback to uniform random
        return random.randrange(W.shape[0])

    exploration = broadcast_probability(phase, step)
    # Apply exploration factor (softmax‑like)
    probs = np.power(degrees, exploration)
    total = probs.sum()
    if total == 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / total

    # Sample according to probs
    cumulative = np.cumsum(probs)
    r = random.random()
    return int(np.searchsorted(cumulative, r))


# ----------------------------------------------------------------------
# High‑level hybrid system
# ----------------------------------------------------------------------
class HybridGraphPheromoneSystem:
    """
    Combines:
    * Graph built from perceptual hashes (Parent A)
    * Edge pheromones with decay (Parent B)
    * Bandit‑driven leader election using pheromone‑weighted degrees
    * VRAM‑aware gradient adjustments
    """

    def __init__(self, elements: list[list[float]], half_life_seconds: float = 60.0):
        self.graph = build_graph(elements)
        self.A, self.node_order = adjacency_matrix(self.graph)  # adjacency matrix
        self.N = self.A.shape[0]
        self.pheromones = PheromoneStore(half_life_seconds)
        # initialise pheromones on existing edges with a small random seed
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.A[i, j] > 0:
                    self.pheromones.add(i, j, random.uniform(0.1, 0.5))

    def step(self, phase: int, step: int) -> str:
        """
        Perform one hybrid iteration:
        1. Build pheromone matrix Φ.
        2. Compute weighted graph W = A ∘ Φ.
        3. Apply a gradient step toward a VRAM‑constrained target matrix.
        4. Select a leader node via bandit logic.
        5. Return the leader's original identifier (string).
        """
        Φ = self.pheromones.matrix(self.N)
        W = pheromone_weighted_graph(self.A, Φ)

        # VRAM‑aware target: keep total memory below a threshold (e.g., 10 MB)
        vram_mb = estimate_vram_usage(W)
        target = np.copy(W)
        if vram_mb > 10.0:
            # Scale down uniformly to fit the budget
            scale = math.sqrt(10.0 / vram_mb)
            target *= scale

        W_updated = gradient_step(W, target, lr=0.05)

        # Update pheromones based on the new weighted strengths
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.A[i, j] > 0:
                    reinforcement = W_updated[i, j] - W[i, j]  # positive = reward
                    if reinforcement > 0:
                        # increase pheromone proportionally
                        new_val = self.pheromones.get(i, j) + reinforcement * 0.1
                        self.pheromones.add(i, j, new_val)

        leader_idx = select_leader_bandit(W_updated, phase, step)
        return self.node_order[leader_idx]

    def current_state(self) -> dict:
        """Return a snapshot useful for debugging / testing."""
        Φ = self.pheromones.matrix(self.N)
        W = pheromone_weighted_graph(self.A, Φ)
        return {
            "adjacency": self.A.tolist(),
            "pheromones": Φ.tolist(),
            "weighted": W.tolist(),
            "vram_mb": estimate_vram_usage(W),
        }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic data: 20 elements, each a 128‑dim float vector
    random.seed(42)
    np.random.seed(42)
    elements = [np.random.rand(128).tolist() for _ in range(20)]

    system = HybridGraphPheromoneSystem(elements, half_life_seconds=30.0)

    # Run a few hybrid steps
    for phase in range(1, 5):
        leader = system.step(phase=phase, step=phase)  # using same value for simplicity
        state = system.current_state()
        print(f"Phase {phase}: selected leader node {leader}, VRAM ≈ {state['vram_mb']:.2f} MB")
    sys.exit(0)