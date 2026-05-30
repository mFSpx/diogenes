# DARWIN HAMMER — match 995, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_distributed_l_hybrid_model_vram_sc_m95_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_ternar_m374_s1.py (gen3)
# born: 2026-05-29T23:32:11Z

"""
Hybrid Graph-Pheromone Model
============================

This module fuses the core mathematics of the two parent algorithms:

* **Parent A** – builds a perceptual‑hash based similarity graph and updates an
  adjacency weight matrix *W* with a simple gradient descent that is driven by
  a broadcast probability function.

* **Parent B** – maintains a pheromone dictionary whose signals decay over time
  and are used to modulate the exploration intensity of a multi‑armed bandit
  (entropy‑based) decision process.

**Mathematical bridge**

The bridge is the *edge weight* of the similarity graph.  In the original
Parent A the weight matrix *W* is updated only by the broadcast probability.
In the hybrid we let the pheromone signal associated with a node act as an
additive bias on the corresponding rows/columns of *W*.  Concretely, for an
edge *(i, j)* we compute


ΔW_ij = - η * (A_ij - σ(W_ij)) * p_bc(phase, step) + λ * Φ_i_j


where

* `A` is the binary adjacency matrix derived from perceptual hashes,
* `σ` is the logistic sigmoid,
* `p_bc` is the broadcast probability from Parent A,
* `Φ_i_j` is the averaged pheromone signal of the two incident nodes,
* `η` (learning rate) and `λ` (pheromone influence) are hyper‑parameters.

The three public functions below expose the hybrid operation:
`build_graph`, `update_weights_with_pheromones`, and `select_leader`.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Mapping, Hashable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Utility functions taken from Parent A
# ----------------------------------------------------------------------
def compute_phash(values: list[float]) -> int:
    """Perceptual hash: 1 bit per value indicating >= mean (max 64 bits)."""
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
    """Return p = 1 / 2^(phase‑step) clamped to [0, 1]."""
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


# ----------------------------------------------------------------------
# Pheromone system taken from Parent B (trimmed to essentials)
# ----------------------------------------------------------------------
class PheromoneSystem:
    """Manages decaying pheromone signals for arbitrary surface keys."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        """Create or update a pheromone entry and return the current value."""
        now = datetime.now(timezone.utc)
        entry = self._store.get(surface_key)
        if entry is None:
            # first appearance
            self._store[surface_key] = {
                "kind": signal_kind,
                "value": signal_value,
                "half_life": half_life_seconds,
                "timestamp": now,
            }
            return signal_value

        # decay the previous value
        elapsed = (now - entry["timestamp"]).total_seconds()
        decayed = entry["value"] * math.pow(
            0.5, elapsed / entry["half_life"]
        )
        # replace with the new raw value (could also blend)
        entry.update(
            {
                "kind": signal_kind,
                "value": signal_value,
                "half_life": half_life_seconds,
                "timestamp": now,
            }
        )
        return signal_value, decayed

    def current(self, surface_key: str) -> float:
        """Return the presently decayed pheromone value (0 if absent)."""
        entry = self._store.get(surface_key)
        if entry is None:
            return 0.0
        elapsed = (datetime.now(timezone.utc) - entry["timestamp"]).total_seconds()
        return entry["value"] * math.pow(0.5, elapsed / entry["half_life"])


# ----------------------------------------------------------------------
# Core hybrid model
# ----------------------------------------------------------------------
class HybridGraphPheromoneModel:
    """Encapsulates the fused graph‑pheromone dynamics."""

    def __init__(
        self,
        elements: list[list[float]],
        learning_rate: float = 0.1,
        pheromone_factor: float = 0.5,
    ):
        """
        Parameters
        ----------
        elements : list of feature vectors (float lists)
            Input data from which the similarity graph is derived.
        learning_rate : float
            Gradient descent step size (η).
        pheromone_factor : float
            Influence weight of pheromones on edge updates (λ).
        """
        self.elements = elements
        self.node_ids = [str(i) for i in range(len(elements))]
        self.adj_matrix = self._build_adjacency(elements)  # binary A
        self.weights = self._init_weights(len(elements))   # real‑valued W
        self.pheromones = PheromoneSystem()
        self.lr = learning_rate
        self.lam = pheromone_factor

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_adjacency(elements: list[list[float]]) -> np.ndarray:
        """Create binary adjacency matrix using perceptual hash similarity."""
        n = len(elements)
        hashes = [compute_phash(el) for el in elements]
        A = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                if hamming_distance(hashes[i], hashes[j]) <= 4:
                    A[i, j] = A[j, i] = 1.0
        return A

    @staticmethod
    def _init_weights(n: int) -> np.ndarray:
        """Random symmetric weight matrix with small values."""
        rng = np.random.default_rng()
        W = rng.normal(loc=0.0, scale=0.1, size=(n, n)).astype(np.float32)
        # enforce symmetry and zero diagonal
        W = (W + W.T) / 2.0
        np.fill_diagonal(W, 0.0)
        return W

    # ------------------------------------------------------------------
    # Hybrid update step
    # ------------------------------------------------------------------
    def update_weights(
        self,
        phase: int,
        step: int,
    ) -> None:
        """
        Perform a single hybrid gradient‑descent update of the weight matrix.

        The update mixes:
        * the broadcast probability from Parent A,
        * the sigmoid‑error term (A - σ(W)),
        * the averaged pheromone signal of the two incident nodes.
        """
        p_bc = broadcast_probability(phase, step)
        sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))

        n = self.weights.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                # error term based on current adjacency
                error = self.adj_matrix[i, j] - sigmoid(self.weights[i, j])

                # pheromone bias – use node identifiers as surface keys
                phi_i = self.pheromones.current(f"node_{i}")
                phi_j = self.pheromones.current(f"node_{j}")
                phi_avg = (phi_i + phi_j) / 2.0

                # hybrid gradient step
                delta = -self.lr * error * p_bc + self.lam * phi_avg
                self.weights[i, j] += delta
                self.weights[j, i] = self.weights[i, j]  # keep symmetry

    # ------------------------------------------------------------------
    # Pheromone interaction
    # ------------------------------------------------------------------
    def deposit_pheromone(
        self,
        node_index: int,
        signal_kind: str = "visit",
        signal_value: float = 1.0,
        half_life_seconds: float = 30.0,
    ) -> None:
        """Add (or refresh) a pheromone entry for a given node."""
        key = f"node_{node_index}"
        self.pheromones.signal(key, signal_kind, signal_value, half_life_seconds)

    # ------------------------------------------------------------------
    # Leader selection (eigenvector centrality approximation)
    # ------------------------------------------------------------------
    def select_leader(self, iterations: int = 20) -> str:
        """
        Approximate the principal eigenvector of the current weight matrix
        using the power method and return the node id with maximal centrality.
        """
        n = self.weights.shape[0]
        vec = np.random.rand(n).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-12
        for _ in range(iterations):
            vec = self.weights @ vec
            norm = np.linalg.norm(vec) + 1e-12
            vec /= norm
        leader_idx = int(np.argmax(vec))
        return self.node_ids[leader_idx]

    # ------------------------------------------------------------------
    # Exposed high‑level operation
    # ------------------------------------------------------------------
    def run_step(self, phase: int, step: int) -> str:
        """
        Execute a full hybrid iteration:
        1. Deposit a random pheromone on a random node (simulating activity).
        2. Update the weight matrix with the hybrid rule.
        3. Return the current leader identifier.
        """
        random_node = random.randint(0, len(self.node_ids) - 1)
        self.deposit_pheromone(random_node)
        self.update_weights(phase, step)
        return self.select_leader()


# ----------------------------------------------------------------------
# Stand‑alone helper functions (required by the prompt)
# ----------------------------------------------------------------------
def build_graph(elements: list[list[float]]) -> np.ndarray:
    """Public wrapper that returns the binary adjacency matrix."""
    return HybridGraphPheromoneModel._build_adjacency(elements)


def update_weights_with_pheromones(
    weight_matrix: np.ndarray,
    adjacency: np.ndarray,
    pheromone_vals: np.ndarray,
    phase: int,
    step: int,
    lr: float = 0.1,
    lam: float = 0.5,
) -> np.ndarray:
    """
    Vectorised version of the hybrid update rule for external use.

    Parameters
    ----------
    weight_matrix : np.ndarray
        Current symmetric weight matrix W.
    adjacency : np.ndarray
        Binary adjacency matrix A.
    pheromone_vals : np.ndarray
        1‑D array with a pheromone value per node.
    phase, step : int
        Broadcast probability parameters.
    lr, lam : float
        Learning‑rate and pheromone influence.

    Returns
    -------
    np.ndarray
        Updated weight matrix.
    """
    p_bc = broadcast_probability(phase, step)
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

    # error term (A - σ(W))
    error = adjacency - sigmoid(weight_matrix)

    # pheromone bias: average of the two incident nodes
    phi_matrix = (pheromone_vals[:, None] + pheromone_vals[None, :]) / 2.0

    delta = -lr * error * p_bc + lam * phi_matrix
    new_W = weight_matrix + delta

    # enforce symmetry and zero diagonal
    new_W = (new_W + new_W.T) / 2.0
    np.fill_diagonal(new_W, 0.0)
    return new_W.astype(np.float32)


def select_leader(weights: np.ndarray, iterations: int = 20) -> int:
    """
    Power‑method leader extraction on an arbitrary weight matrix.
    Returns the index of the node with maximal eigenvector centrality.
    """
    n = weights.shape[0]
    vec = np.random.rand(n).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-12
    for _ in range(iterations):
        vec = weights @ vec
        vec_norm = np.linalg.norm(vec) + 1e-12
        vec /= vec_norm
    return int(np.argmax(vec))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # generate synthetic data: 15 nodes, each with 20 random float features
    rng = np.random.default_rng(seed=42)
    elements = rng.random((15, 20)).tolist()

    model = HybridGraphPheromoneModel(elements, learning_rate=0.05, pheromone_factor=0.2)

    # run a few hybrid iterations
    for phase in range(1, 4):
        for step in range(1, 4):
            leader = model.run_step(phase, step)
            print(f"Phase {phase}, Step {step} -> Leader: {leader}")

    # demonstrate the vectorised helper
    pher_vals = np.array([model.pheromones.current(f"node_{i}") for i in range(len(elements))],
                         dtype=np.float32)
    new_W = update_weights_with_pheromones(
        model.weights,
        model.adj_matrix,
        pher_vals,
        phase=2,
        step=2,
        lr=0.05,
        lam=0.2,
    )
    print("Updated weight matrix (sample entry):", new_W[0, 1])
    print("Selected leader via helper:", select_leader(new_W))