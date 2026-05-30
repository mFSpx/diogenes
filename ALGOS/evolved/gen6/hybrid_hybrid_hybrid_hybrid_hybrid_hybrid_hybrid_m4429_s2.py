# DARWIN HAMMER — match 4429, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_rbf_su_m2481_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1599_s5.py (gen5)
# born: 2026-05-29T23:55:35Z

"""Hybrid algorithm combining Bayesian RBF surrogate scaling with Physarum‑inspired network dynamics.

Parents:
- hybrid_hybrid_hybrid_bayes__hybrid_hybrid_rbf_su_m2481_s2.py (RBF surrogate, perceptual‑hash similarity)
- hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1599_s5.py (Physarum flux, conductance update, information density)

Mathematical bridge:
The similarity matrix derived from perceptual hashes defines a weighted graph.
Physarum dynamics are run on this graph, producing node pressures and edge
conductances.  The resulting information density (log‑pressure) is used as a
multiplicative factor that modulates the Bayesian likelihood computed by an
RBF surrogate model.  Thus the Bayesian update is “physically” informed by the
network flow that encodes feature similarity."""
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Mapping, Hashable, List, Dict, Tuple, Set

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Vector = Sequence[float]
Node = Hashable
FeatureVec = Sequence[float]
Graph = Mapping[Node, Set[Node]]

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash based on average threshold."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def similarity_matrix(hashes: Dict[Node, int]) -> Tuple[np.ndarray, List[Node]]:
    """Pairwise similarity (1 - normalized Hamming distance) between hashes."""
    nodes = list(hashes.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = hashes[ni]
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                hj = hashes[nj]
                d = hamming_distance(hi, hj)
                S[i, j] = 1.0 - d / 64.0
    return S, nodes


# ----------------------------------------------------------------------
# Utilities from Parent B (Physarum)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on an edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 1.0, gain: float = 1.0,
                       decay: float = 0.05,
                       gain_cap: float = 10.0) -> float:
    """Conductance dynamics with hard gain cap."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    effective_gain = min(gain, gain_cap)
    new_val = conductance + dt * (effective_gain * abs(q) - decay * conductance)
    return max(0.0, new_val)


def information_density(pressure: float) -> float:
    """Entropy‑style information density derived from pressure."""
    return math.log(pressure + 1.0)


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RBFSurrogate:
    """RBF surrogate model defined by centres and a scale parameter."""
    centers: np.ndarray          # shape (m, d)
    sigma: float = 1.0           # epsilon in the Gaussian kernel

    def evaluate(self, x: Vector) -> float:
        """Return the average RBF response of x to all centres."""
        diffs = self.centers - np.asarray(x)
        dists = np.linalg.norm(diffs, axis=1)
        responses = np.exp(-((self.sigma * dists) ** 2))
        return float(responses.mean() if responses.size else 0.0)


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def build_similarity_graph(features: Dict[Node, FeatureVec]) -> Tuple[np.ndarray, List[Node]]:
    """
    Build a weighted adjacency matrix from feature vectors.
    Steps:
    1. Compute a perceptual hash for each node (using the raw feature values).
    2. Convert hashes to a similarity matrix S (values in [0,1]).
    The matrix S is interpreted as edge lengths = 1 - similarity for Physarum.
    """
    hashes = {n: compute_phash(list(vec)) for n, vec in features.items()}
    S, nodes = similarity_matrix(hashes)
    return S, nodes


def physarum_step(S: np.ndarray,
                  pressures: np.ndarray,
                  conductances: np.ndarray,
                  dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one Physarum iteration on a fully‑connected graph whose
    edge lengths are derived from similarity matrix S.

    Parameters
    ----------
    S : (n, n) similarity matrix (0..1)
    pressures : (n,) current node pressures
    conductances : (n, n) symmetric matrix of edge conductances
    dt : time step for conductance update

    Returns
    -------
    new_pressures : (n,) updated pressures (here we keep them unchanged;
                    a full pressure solve would require solving a linear system;
                    for the hybrid demo we simply return the input).
    new_conductances : (n, n) updated conductance matrix
    """
    n = S.shape[0]
    new_conductances = conductances.copy()
    # Edge length is inversely related to similarity: higher similarity → shorter path
    edge_lengths = 1.0 - S
    for i in range(n):
        for j in range(i + 1, n):
            q = flux(conductances[i, j], edge_lengths[i, j],
                     pressures[i], pressures[j])
            updated = update_conductance(conductances[i, j], q, dt=dt)
            new_conductances[i, j] = new_conductances[j, i] = updated
    return pressures, new_conductances


def hybrid_bayes_score(prior: float,
                       evidence: FeatureVec,
                       surrogate: RBFSurrogate,
                       conductances: np.ndarray,
                       pressures: np.ndarray) -> float:
    """
    Compute a Bayesian posterior that is scaled by both an RBF surrogate
    and Physarum‑derived information density.

    posterior ∝ prior × L(evidence) × RBF(evidence) × ρ,
    where
        L(evidence)   – simple Gaussian likelihood based on distance to surrogate centres,
        RBF(evidence) – surrogate evaluation,
        ρ             – mean information density derived from node pressures.
    """
    # 1. Gaussian likelihood using distance to surrogate centres
    centre_dists = np.linalg.norm(surrogate.centers - np.asarray(evidence), axis=1)
    likelihoods = np.exp(-((surrogate.sigma * centre_dists) ** 2))
    L = float(likelihoods.mean() if likelihoods.size else 0.0)

    # 2. RBF surrogate output
    R = surrogate.evaluate(evidence)

    # 3. Information density factor: average over all nodes
    rho = float(np.mean([information_density(p) for p in pressures]))

    posterior = prior * L * R * rho
    return posterior


def hybrid_iteration(features: Dict[Node, FeatureVec],
                     prior: float,
                     surrogate: RBFSurrogate,
                     dt: float = 1.0) -> Tuple[Dict[Node, float], np.ndarray]:
    """
    One full hybrid iteration:
    * Build similarity graph from features.
    * Initialise uniform conductances.
    * Run a Physarum step to obtain updated conductances.
    * Compute a posterior score for each node using the hybrid Bayesian rule.
    Returns a mapping node → posterior score and the final conductance matrix.
    """
    S, node_order = build_similarity_graph(features)
    n = len(node_order)

    # Initialise pressures randomly (positive) and conductances uniformly.
    pressures = np.abs(np.random.randn(n)) + 0.1
    conductances = np.ones((n, n), dtype=np.float64) * 0.5
    np.fill_diagonal(conductances, 0.0)

    # Physarum dynamics
    pressures, conductances = physarum_step(S, pressures, conductances, dt=dt)

    # Hybrid Bayesian scores per node (using node's own feature vector as evidence)
    scores: Dict[Node, float] = {}
    for idx, node in enumerate(node_order):
        evidence = features[node]
        scores[node] = hybrid_bayes_score(prior, evidence, surrogate,
                                          conductances, pressures)
    return scores, conductances


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic dataset of 5 nodes with 3‑dimensional features
    random.seed(42)
    np.random.seed(42)
    nodes = [f"node_{i}" for i in range(5)]
    features = {n: np.random.rand(3).tolist() for n in nodes}

    # Define an RBF surrogate with 2 random centres
    centre_array = np.random.rand(2, 3)
    surrogate = RBFSurrogate(centers=centre_array, sigma=1.0)

    # Prior probability (arbitrary)
    prior = 0.2

    # Run a single hybrid iteration
    scores, final_conductances = hybrid_iteration(features, prior, surrogate, dt=0.5)

    # Print results
    print("Hybrid posterior scores:")
    for node, score in scores.items():
        print(f"  {node}: {score:.6f}")

    print("\nFinal conductance matrix (rounded):")
    print(np.round(final_conductances, 3))