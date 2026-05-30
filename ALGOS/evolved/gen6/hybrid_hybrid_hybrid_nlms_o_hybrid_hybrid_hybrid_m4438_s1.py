# DARWIN HAMMER — match 4438, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_hybrid_m105_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2421_s0.py (gen5)
# born: 2026-05-29T23:55:50Z

"""Hybrid NLMS‑Sketch‑Sheaf algorithm.
Parents:
- hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_hybrid_m105_s3.py (adaptive NLMS + decision‑hygiene)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2421_s0.py (Count‑Min Sketch, Sheaf Laplacian, decision‑hygiene weights)

Mathematical bridge:
The NLMS weight vector `w` lives in a space that concatenates
    • the Count‑Min Sketch frequency vector `c ∈ ℝ^DIM`,
    • the digit‑count feature vector `f ∈ ℝ^9`.
Prediction is the linear form `y = wᵀ[x]` where `x = [c; f]`.
The learning‑rate `μ` is modulated by the hybrid‑hygiene score `H(f)`,
and a Sheaf‑Laplacian regularizer `L_s` (derived from node/edge dimensions)
adds a diffusion term `‑λ L_s w` to the NLMS update, enforcing smoothness across
the sketch’s graph structure.  This yields a single unified adaptive system."""

import math
import random
import sys
import re
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Constants (taken from parent B)
# ----------------------------------------------------------------------
DIM = 1000                     # reduced size for demo purposes
_NODE_DIMS = {i: 5 for i in range(DIM)}   # node space dimensions
_EDGE_DIM = 3                  # edge space dimension
_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.float64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.float64)

# ----------------------------------------------------------------------
# NLMS core (parent A)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """Standard NLMS weight update."""
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in (0, 2)")
    y = nlms_predict(weights, x)
    error = target - y
    power = float(x @ x) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error


# ----------------------------------------------------------------------
# Feature extraction & hygiene score (parent A)
# ----------------------------------------------------------------------
def extract_digit_features(text: str) -> np.ndarray:
    """9‑dimensional count of digits 0‑8 in the supplied text."""
    return np.array([text.count(str(i)) for i in range(9)], dtype=np.float64)


def hybrid_hygiene_score(features: np.ndarray) -> float:
    """
    Hygiene score = mean count * (1 + normalized Shannon entropy).
    """
    s = np.mean(features)
    # Shannon entropy with small offset to avoid log(0)
    H = -np.sum(features * np.log2(features + 1e-9))
    H_max = np.log2(features.size)
    return s * (1.0 + H / H_max)


# ----------------------------------------------------------------------
# Count‑Min Sketch (parent B)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Very small Count‑Min Sketch with pairwise‑independent hash functions."""
    def __init__(self, dim: int = DIM, depth: int = 5, seed: int = 0):
        self.dim = dim
        self.depth = depth
        self.tables = np.zeros((depth, dim), dtype=np.float64)
        rng = np.random.default_rng(seed)
        # hash parameters a, b for each row: (a * x + b) % prime % dim
        self.prime = 2 ** 31 - 1
        self.a = rng.integers(1, self.prime, size=depth, dtype=np.int64)
        self.b = rng.integers(0, self.prime, size=depth, dtype=np.int64)

    def _hash(self, item: int, row: int) -> int:
        return ((self.a[row] * item + self.b[row]) % self.prime) % self.dim

    def update(self, item: int, inc: float = 1.0) -> None:
        for r in range(self.depth):
            idx = self._hash(item, r)
            self.tables[r, idx] += inc

    def query(self, item: int) -> float:
        return min(self.tables[r, self._hash(item, r)] for r in range(self.depth))

    def vector(self) -> np.ndarray:
        """Return a flattened frequency estimate (mean across rows)."""
        return self.tables.mean(axis=0)


# ----------------------------------------------------------------------
# Sheaf Laplacian regularizer (parent B)
# ----------------------------------------------------------------------
def build_simple_graph(dim: int) -> Dict[int, List[int]]:
    """Create a ring graph on `dim` nodes for demonstration."""
    graph = defaultdict(list)
    for i in range(dim):
        graph[i].append((i + 1) % dim)
        graph[i].append((i - 1) % dim)
    return graph


def sheaf_laplacian_regularizer(
    weights: np.ndarray,
    node_dims: Dict[int, int],
    edge_dim: int,
    lam: float = 0.01,
) -> np.ndarray:
    """
    Compute a diffusion gradient -λ L_s w.
    For simplicity we treat each node as a scalar (the corresponding weight entry)
    and use the ring graph adjacency.
    """
    dim = len(weights)
    graph = build_simple_graph(dim)
    grad = np.zeros_like(weights)
    for i in range(dim):
        neighbor_sum = sum(weights[j] for j in graph[i])
        degree = len(graph[i])
        laplacian_i = degree * weights[i] - neighbor_sum
        grad[i] = -lam * laplacian_i
    return grad


# ----------------------------------------------------------------------
# Hybrid operations – three public functions
# ----------------------------------------------------------------------
def hybrid_predict(
    weights: np.ndarray,
    sketch: CountMinSketch,
    text: str,
) -> float:
    """
    Predict a scalar target from the concatenated sketch vector and digit features.
    """
    sketch_vec = sketch.vector()                     # shape (DIM,)
    digit_feat = extract_digit_features(text)       # shape (9,)
    x = np.concatenate([sketch_vec, digit_feat])    # total length DIM+9
    return nlms_predict(weights, x)


def hybrid_update(
    weights: np.ndarray,
    sketch: CountMinSketch,
    text: str,
    target: float,
    base_mu: float = 0.5,
    eps: float = 1e-9,
    reg_lambda: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """
    Perform a single hybrid NLMS update:
        1. Build input vector x = [sketch; digit_features].
        2. Compute hygiene‑driven learning rate μ = base_mu * (1 + H/scale).
        3. Apply NLMS step.
        4. Add Sheaf Laplacian regularisation gradient.
    Returns updated weights and prediction error.
    """
    # 1. Input vector
    sketch_vec = sketch.vector()
    digit_feat = extract_digit_features(text)
    x = np.concatenate([sketch_vec, digit_feat])

    # 2. Adaptive learning rate via hygiene score
    H = hybrid_hygiene_score(digit_feat)
    scale = max(1.0, np.mean(digit_feat) + 1e-6)   # avoid division by zero
    mu = base_mu * (1.0 + H / scale)

    # 3. NLMS core update
    new_w, error = nlms_update(weights, x, target, mu=mu, eps=eps)

    # 4. Sheaf regularisation (only on the sketch part)
    reg_grad = sheaf_laplacian_regularizer(
        new_w[:DIM], _NODE_DIMS, _EDGE_DIM, lam=reg_lambda
    )
    new_w[:DIM] += reg_grad

    return new_w, error


def initialize_hybrid_model(dim: int = DIM) -> Tuple[np.ndarray, CountMinSketch]:
    """
    Initialise the weight vector (zeros) and an empty Count‑Min Sketch.
    Weight length = dim (sketch) + 9 (digit features).
    """
    weights = np.zeros(dim + 9, dtype=np.float64)
    sketch = CountMinSketch(dim=dim, depth=5, seed=42)
    return weights, sketch


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # initialise model
    w, cms = initialize_hybrid_model()

    # synthetic stream of integer items and accompanying free‑text comments
    items = [random.randint(0, 5000) for _ in range(20)]
    texts = [
        "evidence 123 confirmed delay 4",
        "plan step 2 risk 7",
        "audit log 9",
        "support 5 scarcity 3",
        "impulsive 8",
    ]

    # run a few update cycles
    for i, item in enumerate(items):
        txt = random.choice(texts)
        cms.update(item, inc=1.0)

        # pretend target is the raw sketch count of the item (scaled)
        target = cms.query(item) * 0.1

        w, err = hybrid_update(w, cms, txt, target)
        pred = hybrid_predict(w, cms, txt)

        print(
            f"Iter {i:02d} | item={item:4d} | txt='{txt[:30]:30}' | target={target:6.3f} "
            f"| pred={pred:6.3f} | err={err:6.3f}"
        )
    print("\nFinal weight norm:", np.linalg.norm(w))