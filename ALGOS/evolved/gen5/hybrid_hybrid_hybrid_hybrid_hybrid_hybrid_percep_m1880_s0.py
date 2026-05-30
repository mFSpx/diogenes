# DARWIN HAMMER — match 1880, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m217_s2.py (gen4)
# parent_b: hybrid_hybrid_perceptual_de_hybrid_hybrid_hybrid_m1188_s1.py (gen4)
# born: 2026-05-29T23:39:25Z

"""
Hybrid Fusion Model
===================

This module fuses the core topologies of the two parent algorithms:

* **Parent A** – Sheaf + Dense Associative Memory (DAM) + Tensor‑Train (TTT) transformation.
* **Parent B** – Radial‑Basis Function (RBF) surrogate model + Perceptual hashing + Pheromone‑based decay.

**Mathematical Bridge**

The bridge is the *feature vector* produced by the Tensor‑Train transformation of a sheaf
section.  This vector is used

1. as the input to the Dense Associative Memory energy function (Parent A), and
2. as a centre for a Gaussian RBF kernel whose contribution is weighted by a
   pheromone value indexed by a perceptual hash of the same vector (Parent B).

Thus a single vector simultaneously participates in an energy computation,
a kernel‑based surrogate prediction, and a pheromone‑driven adaptive weighting
scheme.  The hybrid model therefore couples the associative‑memory energy
regulariser with a hash‑guided, pheromone‑scaled RBF surrogate.

The implementation below provides a minimal yet functional realisation of this
fusion, with three public functions demonstrating the hybrid operation.
"""

import math
import random
import sys
import pathlib
import numpy as np

Vector = np.ndarray


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def compute_phash(values: list[float]) -> int:
    """Simple perceptual hash: compare each value to the mean of the first 64."""
    if not values:
        return 0
    avg = sum(values[:64]) / min(64, len(values))
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


class Sheaf:
    """Very small sheaf implementation: stores a section per node."""

    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims
        self.edges = edges
        self.sections: dict = {}

    def set_section(self, node, value: Vector) -> None:
        expected = self.node_dims.get(node)
        if expected is None:
            raise KeyError(f"Node {node} unknown")
        if value.shape[0] != expected:
            raise ValueError(f"Section dimension mismatch for node {node}")
        self.sections[node] = value

    def get_section(self, node):
        return self.sections.get(node)


class DenseAssociativeMemory:
    """Simplified DAM: energy = -beta * sum(patterns * input)."""

    def __init__(self, patterns: np.ndarray, beta: float = 1.0):
        self.patterns = patterns  # shape (n_patterns, dim)
        self.beta = beta

    def _compute_energy(self, x: Vector) -> float:
        """Negative inner‑product energy (the lower, the better)."""
        return -self.beta * np.sum(self.patterns * x)


class HybridFusionModel:
    """
    Unified model that:
      * stores sheaf sections,
      * projects them through a Tensor‑Train matrix,
      * evaluates DAM energy,
      * builds a pheromone‑weighted RBF surrogate,
      * clusters vectors via perceptual hashing.
    """

    def __init__(self, node_dims: dict, edges: list, patterns: np.ndarray, beta: float = 1.0):
        self.sheaf = Sheaf(node_dims, edges)
        self.dam = DenseAssociativeMemory(patterns, beta)
        self.ttt_matrices: dict = {}          # node -> TTT matrix
        self.rbf_centers: dict = {}           # node -> list of centre vectors
        self.pheromones: dict = {}            # hash -> weight (decays over time)

    # ------------------------------------------------------------------
    # 1. Tensor‑Train projection (Parent A)
    # ------------------------------------------------------------------
    def _init_ttt(self, d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if d_out is None:
            d_out = d_in
        return rng.standard_normal((d_out, d_in)) * scale

    def compute_ttt(self, node) -> Vector | None:
        """Project the sheaf section of *node* through a fresh TTT matrix."""
        section = self.sheaf.get_section(node)
        if section is None:
            return None
        ttt = self._init_ttt(section.shape[0])
        self.ttt_matrices[node] = ttt
        return ttt @ section

    # ------------------------------------------------------------------
    # 2. DAM energy (Parent A)
    # ------------------------------------------------------------------
    def compute_energy(self, node) -> float | None:
        """Energy of the TTT‑projected vector under the dense associative memory."""
        proj = self.compute_ttt(node)
        if proj is None:
            return None
        return self.dam._compute_energy(proj)

    # ------------------------------------------------------------------
    # 3. Pheromone‑weighted RBF surrogate (Parent B)
    # ------------------------------------------------------------------
    def _hash_vector(self, v: Vector) -> int:
        return compute_phash(v.tolist())

    def _ensure_pheromone(self, h: int) -> None:
        """Create a pheromone entry if missing (initial weight = 1.0)."""
        if h not in self.pheromones:
            self.pheromones[h] = 1.0

    def rbf_predict(self, node, query: Vector, epsilon: float = 1.0) -> float | None:
        """
        Predict a scalar for *query* using all stored RBF centres of *node*.
        Each centre contributes gaussian(euclidean(query, centre)) weighted
        by its pheromone value (indexed by the centre's hash).
        """
        centres = self.rbf_centers.get(node)
        if not centres:
            return None

        total = 0.0
        weight_sum = 0.0
        for c in centres:
            h = self._hash_vector(c)
            self._ensure_pheromone(h)
            w = self.pheromones[h]
            r = euclidean(query, c)
            total += w * gaussian(r, epsilon)
            weight_sum += w
        return total / weight_sum if weight_sum != 0 else 0.0

    # ------------------------------------------------------------------
    # 4. Hybrid update (demonstrates the bridge)
    # ------------------------------------------------------------------
    def hybrid_update(self, node, target: Vector, decay: float = 0.95) -> None:
        """
        Perform a hybrid update:
          * project the current section via TTT,
          * compute residual to *target*,
          * store the projected vector as a new RBF centre,
          * decay the pheromone of the centre's hash,
          * optionally adjust the DAM gradient (omitted for brevity).
        """
        proj = self.compute_ttt(node)
        if proj is None:
            raise RuntimeError(f"No section set for node {node}")

        # 1. Store centre for future RBF predictions
        self.rbf_centers.setdefault(node, []).append(proj.copy())

        # 2. Update pheromone (increase for good match, decay otherwise)
        h = self._hash_vector(proj)
        self._ensure_pheromone(h)
        residual = np.linalg.norm(proj - target)
        # Smaller residual → stronger reinforcement
        reinforcement = math.exp(-residual)
        self.pheromones[h] = self.pheromones[h] * decay + reinforcement

    # ------------------------------------------------------------------
    # 5. Utility: cluster current centres by perceptual hash (Parent B)
    # ------------------------------------------------------------------
    def cluster_centres(self, node, max_hamming: int = 4) -> list[list[int]]:
        """
        Cluster stored centres of *node* using perceptual hash hamming distance.
        Returns a list of lists of centre indices.
        """
        centres = self.rbf_centers.get(node, [])
        hashes = [self._hash_vector(c) for c in centres]

        clusters: list[list[int]] = []
        for idx, h in enumerate(hashes):
            placed = False
            for cl in clusters:
                if bin(h ^ hashes[cl[0]]).count("1") <= max_hamming:
                    cl.append(idx)
                    placed = True
                    break
            if not placed:
                clusters.append([idx])
        return clusters


# ----------------------------------------------------------------------
# Public demonstration functions
# ----------------------------------------------------------------------
def hybrid_feature_vector(model: HybridFusionModel, node) -> Vector | None:
    """Return the Tensor‑Train projected feature vector for *node*."""
    return model.compute_ttt(node)


def hybrid_rbf_score(model: HybridFusionModel, node, query: Vector) -> float | None:
    """Return the pheromone‑weighted RBF surrogate score for *query*."""
    return model.rbf_predict(node, query)


def hybrid_training_step(model: HybridFusionModel, node, target: Vector) -> None:
    """
    Execute one hybrid training step:
      * update the model with *target*,
      * print energy and current pheromone for the latest centre.
    """
    model.hybrid_update(node, target)
    energy = model.compute_energy(node)
    latest_centre = model.rbf_centers[node][-1]
    ph = model._hash_vector(latest_centre)
    pher = model.pheromones[ph]
    print(f"Node {node}: energy={energy:.4f}, pheromone={pher:.4f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph with two nodes
    node_dims = {"A": 5, "B": 5}
    edges = []  # not used in this minimal example

    # Random patterns for DAM (3 patterns, dimension 5)
    rng = np.random.default_rng(42)
    patterns = rng.standard_normal((3, 5))

    model = HybridFusionModel(node_dims, edges, patterns, beta=0.5)

    # Initialise sections with random vectors
    for n in node_dims:
        model.sheaf.set_section(n, rng.standard_normal(node_dims[n]))

    # Perform a few hybrid steps
    for step in range(3):
        node = "A"
        target = rng.standard_normal(node_dims[node])
        hybrid_training_step(model, node, target)

        # Query the surrogate with a new random vector
        query = rng.standard_normal(node_dims[node])
        score = hybrid_rbf_score(model, node, query)
        print(f"RBF surrogate score (step {step}): {score:.4f}")

    # Show clustering of centres
    clusters = model.cluster_centres("A")
    print(f"Centre clusters for node A: {clusters}")