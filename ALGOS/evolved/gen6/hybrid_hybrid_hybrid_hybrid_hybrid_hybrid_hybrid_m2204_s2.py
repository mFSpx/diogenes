# DARWIN HAMMER — match 2204, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1136_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_liquid_m1166_s0.py (gen5)
# born: 2026-05-29T23:41:29Z

"""Hybrid algorithm combining MinHash similarity, Clifford geometric algebra, and regret-weighted strategy.

Parents:
- hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1136_s1 (MinHash, similarity, softmax, Fisher score)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_liquid_m1166_s0 (Clifford geometric product, weight matrix update)

Mathematical bridge:
The MinHash signature similarity provides a scalar quality q∈[0,1]. This scalar is used to
1) update a real‑valued weight matrix W via a simple Hebbian‑style rule:   W ← W + η·q·Δ,
   where Δ is an outer product of feature vectors.
2) construct a multivector M_W from the (flattened) matrix and transform a plan multivector P
   using the Clifford geometric product:   P' = P ⊗ M_W.
3) feed the resulting coefficients into a regret engine that uses the Fisher information
   (derived from a Gaussian beam) and a temperature‑scaled soft‑max to produce a
   probability distribution over actions.

The three core functions below demonstrate this unified pipeline.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import hashlib
from typing import Iterable, List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared by both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Parent A utilities: MinHash, similarity, softmax, Gaussian beam, Fisher score
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity between two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature‑scaled soft‑max."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = values / temperature
    max_val = np.max(scaled)
    exp_vals = np.exp(scaled - max_val)
    return exp_vals / exp_vals.sum()


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """Fisher information for a 1‑D Gaussian."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Parent B utilities: Clifford (geometric) algebra
# ----------------------------------------------------------------------
def _blade_sign(indices: Tuple[int, ...]) -> Tuple[List[int], int]:
    """Return sorted indices and sign after applying anti‑commutation rules."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate basis vector squares to scalar → remove both
                lst.pop(j)
                lst.pop(j)
                n -= 2
                i -= 1
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    combined = tuple(blade_a) + tuple(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def geometric_product(a: Dict[frozenset, float], b: Dict[frozenset, float]) -> Dict[frozenset, float]:
    """Geometric product of two multivectors represented as dicts."""
    result: Dict[frozenset, float] = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade, sign = _multiply_blades(blade_a, blade_b)
            result[blade] = result.get(blade, 0.0) + coef_a * coef_b * sign
    return result


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def update_weight_matrix(W: np.ndarray, q: float, eta: float = 0.01) -> np.ndarray:
    """
    Hebbian‑style update using scalar similarity q.
    Δ = q * (outer product of a random feature vector with itself).
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("similarity q must be in [0,1]")
    dim = W.shape[0]
    # Simple feature vector: normalized random Gaussian
    f = np.random.randn(dim)
    f /= np.linalg.norm(f) + 1e-12
    delta = q * np.outer(f, f)
    return W + eta * delta


def multivector_from_matrix(W: np.ndarray) -> Dict[frozenset, float]:
    """
    Encode a square matrix as a multivector.
    Diagonal elements become scalar (grade‑0) part,
    off‑diagonal (i,j) with i<j become bivector e_i∧e_j,
    and with i>j are handled by antisymmetry automatically.
    """
    n = W.shape[0]
    mv: Dict[frozenset, float] = {}
    # scalar (empty blade)
    scalar = np.trace(W) / n
    mv[frozenset()] = scalar
    for i in range(n):
        for j in range(i + 1, n):
            coeff = (W[i, j] - W[j, i]) / 2.0  # antisymmetric part
            if abs(coeff) > 1e-12:
                mv[frozenset({i, j})] = coeff
    return mv


def regret_weighted_strategy(
    actions: List[MathAction],
    similarity_q: float,
    temperature: float = 0.5,
) -> Tuple[List[str], np.ndarray]:
    """
    Compute a probability distribution over actions using:
    - Fisher information as a regret proxy,
    - similarity_q as a scaling factor,
    - temperature‑scaled softmax.
    Returns the ordered list of action IDs and the corresponding probabilities.
    """
    if not actions:
        raise ValueError("actions list cannot be empty")
    # Build a vector of raw scores: expected_value - cost - risk, scaled by similarity
    raw = np.array(
        [
            (a.expected_value - a.cost - a.risk) * similarity_q
            + fisher_score(a.expected_value) * (1.0 - similarity_q)
            for a in actions
        ],
        dtype=float,
    )
    probs = _softmax(raw, temperature)
    ids = [a.id for a in actions]
    return ids, probs


def hybrid_step(
    tokens_a: Iterable[str],
    tokens_b: Iterable[str],
    actions: List[MathAction],
    W: np.ndarray,
    plan_mv: Dict[frozenset, float],
) -> Tuple[Dict[frozenset, float], List[str], np.ndarray]:
    """
    Full hybrid pipeline:
    1. Compute MinHash signatures and similarity q.
    2. Update weight matrix W with q.
    3. Convert updated W to a multivector M_W.
    4. Transform the plan multivector via geometric product: P' = P ⊗ M_W.
    5. Derive a regret‑weighted action distribution from q and the transformed multivector
       (the latter influences the action scores through its scalar part).
    Returns (transformed_plan, action_ids, action_probabilities).
    """
    # 1. MinHash similarity
    sig_a = signature(tokens_a)
    sig_b = signature(tokens_b)
    q = similarity(sig_a, sig_b)

    # 2. Weight matrix update
    W_new = update_weight_matrix(W, q)

    # 3. Multivector from matrix
    M_W = multivector_from_matrix(W_new)

    # 4. Geometric product transformation
    transformed_plan = geometric_product(plan_mv, M_W)

    # Extract scalar part of transformed plan to bias action scores
    scalar_part = transformed_plan.get(frozenset(), 0.0)

    # 5. Regret‑weighted strategy (scalar part modulates similarity)
    mod_q = min(1.0, max(0.0, q + 0.1 * scalar_part))  # simple coupling
    ids, probs = regret_weighted_strategy(actions, mod_q)

    return transformed_plan, ids, probs


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Token sets for two hypothetical sequences
    tokens_x = ["alpha", "beta", "gamma", "delta"]
    tokens_y = ["beta", "epsilon", "gamma", "zeta"]

    # Simple action list
    action_list = [
        MathAction(id="A", expected_value=1.2, cost=0.2, risk=0.1),
        MathAction(id="B", expected_value=0.8, cost=0.1, risk=0.05),
        MathAction(id="C", expected_value=1.5, cost=0.5, risk=0.2),
    ]

    # Initialize a small weight matrix (3x3) and a plan multivector
    np.random.seed(42)
    W0 = np.random.randn(3, 3)
    # Ensure symmetry for a clean start
    W0 = (W0 + W0.T) / 2.0

    # Example plan multivector: scalar + one bivector e0∧e1
    plan = {
        frozenset(): 0.7,            # scalar
        frozenset({0, 1}): 0.3,      # bivector component
    }

    transformed, ids, probs = hybrid_step(tokens_x, tokens_y, action_list, W0, plan)

    print("Similarity q:", similarity(signature(tokens_x), signature(tokens_y)))
    print("Updated weight matrix (first row):", W0[0])
    print("Transformed plan multivector:", transformed)
    print("Action probabilities:")
    for aid, p in zip(ids, probs):
        print(f"  {aid}: {p:.4f}")