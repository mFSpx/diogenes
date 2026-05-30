# DARWIN HAMMER — match 5197, survivor 4
# gen: 7
# parent_a: hybrid_gini_coefficient_hybrid_hybrid_infota_m740_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m2382_s0.py (gen6)
# born: 2026-05-30T00:00:32Z

"""Hybrid Entropy‑Gini / Bayesian Sheaf Algorithm

Parents:
- **hybrid_gini_coefficient_hybrid_hybrid_infota_m740_s2.py** – provides
  normalised probability handling, Shannon entropy, Gini index and a
  weighted MinHash signature.
- **hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m2382_s0.py** – defines a
  *HybridSheaf* data structure, Bayesian‑evidence weighting and a temporal
  burst generator.

Mathematical Bridge
-------------------
The bridge is the **Bayesian evidence scalar `e`** produced from the sheaf
patterns.  In the original parents the evidence modulated burst signals;
here it also scales the combined entropy‑Gini metric:


M(p, e) = e · ( α·Ĥ(p) + (1‑α)·G(p) )


where `Ĥ` is the normalised Shannon entropy, `G` the Gini index and `α∈[0,1]`
a configurable weight.  This metric feeds the weighted MinHash
construction, giving higher‑probability (and higher‑evidence) tokens a larger
influence on the signature.  The resulting signature can be compared with
a deterministic Jaccard‑like similarity, while the metric `M` drives a
temporal burst signal.

The module therefore fuses:
1. probability‑space statistics (entropy, Gini),
2. Bayesian‑scaled inequality (evidence),
3. sheaf‑based pattern generation,
4. weighted MinHash and burst computation.

All components are implemented with NumPy and the Python standard library.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Utilities from Parent A
# --------------------------------------------------------------------------- #


def _validate_and_normalise(probs: Sequence[float]) -> np.ndarray:
    """Return a normalised probability vector as a NumPy array.

    Raises:
        ValueError: If the input contains negative values or sums to zero.
    """
    arr = np.asarray(probs, dtype=float)
    if np.any(arr < 0):
        raise ValueError("Probabilities must be non‑negative.")
    total = arr.sum()
    if total <= 0.0:
        raise ValueError("Sum of probabilities must be positive.")
    return arr / total


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash using BLAKE2b."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


# --------------------------------------------------------------------------- #
# Core mathematical primitives (entropy, gini, Bayesian evidence)
# --------------------------------------------------------------------------- #


def shannon_entropy_norm(probs: np.ndarray) -> float:
    """Normalised Shannon entropy in [0, 1].

    Ĥ = -∑ p_i log(p_i) / log(N) ,  where N = len(probs).
    """
    # Guard against log(0) by selecting only positive entries
    positive = probs[probs > 0]
    if positive.size == 0:
        return 0.0
    ent = -np.sum(positive * np.log(positive))
    norm = ent / math.log(probs.size) if probs.size > 1 else 0.0
    return float(norm)


def gini_index(probs: np.ndarray) -> float:
    """Classic Gini index for a probability distribution."""
    return 1.0 - float(np.sum(probs ** 2))


def combined_entropy_gini(
    probs: Sequence[float],
    alpha: float = 0.5,
) -> float:
    """Weighted combination α·Ĥ + (1‑α)·G, where Ĥ is normalised entropy."""
    p = _validate_and_normalise(probs)
    h = shannon_entropy_norm(p)
    g = gini_index(p)
    return float(alpha * h + (1.0 - alpha) * g)


def bayesian_evidence(patterns: np.ndarray) -> float:
    """A lightweight Bayesian evidence estimator.

    Treat each pattern vector as a Gaussian observation with unit variance.
    The log‑likelihood of the whole pattern matrix is proportional to
    -0.5·||patterns||².  The evidence scalar is a sigmoid‑scaled version
    of that log‑likelihood, yielding a value in (0, 1).
    """
    if patterns.size == 0:
        return 0.0
    log_likelihood = -0.5 * np.linalg.norm(patterns) ** 2
    # Sigmoid to map to (0,1)
    return 1.0 / (1.0 + math.exp(-log_likelihood))


# --------------------------------------------------------------------------- #
# HybridSheaf – simplified version of Parent B
# --------------------------------------------------------------------------- #


class HybridSheaf:
    """Hybrid data structure merging a cellular sheaf with dense associative memory.

    For the purposes of this fusion the sheaf stores a pattern matrix
    `patterns` of shape (n_nodes, dim).  The matrix is interpreted as a set of
    high‑dimensional embeddings that can be queried per node.
    """

    def __init__(self, node_dims: Dict[int, int], edges: List[Tuple[int, int]], patterns: np.ndarray):
        self.node_dims = node_dims          # mapping node → feature dimension
        self.edges = edges                  # list of (src, dst) tuples
        self.patterns = patterns.astype(float)
        self._evidence: float | None = None

    @property
    def evidence(self) -> float:
        """Cached Bayesian evidence; computed on first access."""
        if self._evidence is None:
            self._evidence = bayesian_evidence(self.patterns)
        return self._evidence

    def node_pattern(self, node: int) -> np.ndarray:
        """Return the pattern vector associated with *node*."""
        if node not in self.node_dims:
            raise KeyError(f"Node {node} not defined in node_dims.")
        dim = self.node_dims[node]
        vec = self.patterns[node, :dim]
        return vec

    def aggregate(self) -> np.ndarray:
        """Aggregate patterns over all nodes (mean vector)."""
        if self.patterns.shape[0] == 0:
            return np.array([], dtype=float)
        return np.mean(self.patterns, axis=0)


# --------------------------------------------------------------------------- #
# Hybrid operations – three public functions
# --------------------------------------------------------------------------- #


def weighted_minhash_signature(
    probs: Sequence[float],
    sheaf: HybridSheaf,
    seed: int = 0,
    num_hashes: int = 5,
    alpha: float = 0.5,
) -> List[int]:
    """Create a weighted MinHash signature.

    The weight for token *i* is `w_i = p_i * e`, where `e` is the sheaf evidence.
    The hash value for token *i* under hash function *k* is
    `h_{k,i} = _hash(seed + k, str(i)) / w_i`.  The minimum across tokens is the
    k‑th component of the signature.
    """
    p = _validate_and_normalise(probs)
    e = sheaf.evidence
    eps = 1e-12  # avoid division by zero

    signature: List[int] = []
    for k in range(num_hashes):
        min_val = float("inf")
        min_tok = -1
        for idx, prob in enumerate(p):
            weight = prob * e + eps
            h = _hash(seed + k, str(idx)) / weight
            if h < min_val:
                min_val = h
                min_tok = idx
        signature.append(min_tok)
    return signature


def hybrid_metric(
    probs: Sequence[float],
    sheaf: HybridSheaf,
    alpha: float = 0.5,
) -> float:
    """Combined entropy‑Gini metric scaled by Bayesian evidence.

    M(p, e) = e · ( α·Ĥ(p) + (1‑α)·G(p) )
    """
    base = combined_entropy_gini(probs, alpha=alpha)
    return float(sheaf.evidence * base)


def burst_signal(
    metric: float,
    time_step: int,
    frequency: float = 0.1,
    decay: float = 0.01,
) -> float:
    """Temporal burst signal driven by the hybrid metric.

    The signal follows a damped sinusoid:
        s(t) = metric · exp(-decay·t) · sin(2π·frequency·t)
    """
    envelope = math.exp(-decay * time_step)
    oscillation = math.sin(2.0 * math.pi * frequency * time_step)
    return float(metric * envelope * oscillation)


# --------------------------------------------------------------------------- #
# Helper to generate synthetic sheaf patterns (used in the smoke test)
# --------------------------------------------------------------------------- #


def generate_random_sheaf(
    num_nodes: int,
    dim: int,
    seed: int = 42,
) -> HybridSheaf:
    """Generate a random HybridSheaf with Gaussian patterns."""
    random.seed(seed)
    np.random.seed(seed)

    node_dims = {i: dim for i in range(num_nodes)}
    # Random edges (directed) – simple ring topology for determinism
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]

    # Patterns: each node gets a dim‑dimensional Gaussian vector
    patterns = np.random.normal(loc=0.0, scale=1.0, size=(num_nodes, dim))

    return HybridSheaf(node_dims=node_dims, edges=edges, patterns=patterns)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Example probability vector (unnormalised)
    probs = [0.2, 0.5, 0.3]

    # Generate a synthetic sheaf
    sheaf = generate_random_sheaf(num_nodes=3, dim=8, seed=123)

    # Compute hybrid metric
    metric = hybrid_metric(probs, sheaf, alpha=0.6)
    print(f"Hybrid metric (scaled entropy‑Gini): {metric:.6f}")

    # Produce a weighted MinHash signature
    sig = weighted_minhash_signature(probs, sheaf, seed=7, num_hashes=4, alpha=0.6)
    print(f"Weighted MinHash signature: {sig}")

    # Generate a short burst signal sequence
    print("Burst signal over 10 timesteps:")
    for t in range(10):
        val = burst_signal(metric, time_step=t, frequency=0.2, decay=0.05)
        print(f" t={t:2d} → {val:.6f}")

    sys.exit(0)