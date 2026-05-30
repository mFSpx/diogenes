# DARWIN HAMMER — match 2204, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1136_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_liquid_m1166_s0.py (gen5)
# born: 2026-05-29T23:41:29Z

import math
import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures
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
# Deterministic token → feature vector (replaces random Gaussian)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def token_feature_vector(tokens: Iterable[str], dim: int, seed: int = 0) -> np.ndarray:
    """
    Produce a reproducible, dense feature vector from a token set.
    Each token contributes a hashed one‑hot pattern that is accumulated.
    The result is L2‑normalised.
    """
    vec = np.zeros(dim, dtype=float)
    for t in tokens:
        if not t:
            continue
        h = _hash(seed, t)
        idx = h % dim
        sign = 1.0 if (h >> 32) & 1 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


# ----------------------------------------------------------------------
# MinHash utilities (unchanged)
# ----------------------------------------------------------------------
def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Soft‑max and Gaussian/Fisher utilities (unchanged)
# ----------------------------------------------------------------------
def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = values / temperature
    max_val = np.max(scaled)
    exp_vals = np.exp(scaled - max_val)
    return exp_vals / exp_vals.sum()


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Clifford (geometric) algebra utilities
# ----------------------------------------------------------------------
def _blade_sign(indices: Tuple[int, ...]) -> Tuple[List[int], int]:
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
    result: Dict[frozenset, float] = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade, sign = _multiply_blades(blade_a, blade_b)
            result[blade] = result.get(blade, 0.0) + coef_a * coef_b * sign
    return result


# ----------------------------------------------------------------------
# Multivector ↔ matrix conversion (full encoding, not just antisymmetric)
# ----------------------------------------------------------------------
def multivector_from_matrix(W: np.ndarray) -> Dict[frozenset, float]:
    """
    Encode a square matrix as a multivector:
    - scalar (grade‑0) = average of diagonal,
    - vector (grade‑1) = diagonal deviations,
    - bivector (grade‑2) = antisymmetric part,
    - higher grades are ignored (they would require a full exterior algebra).
    """
    n = W.shape[0]
    mv: Dict[frozenset, float] = {}

    # scalar part: mean of diagonal
    mv[frozenset()] = np.trace(W) / n

    # vector part: each diagonal element as e_i
    for i in range(n):
        dev = W[i, i] - mv[frozenset()]
        if abs(dev) > 1e-12:
            mv[frozenset({i})] = dev

    # bivector part: antisymmetric off‑diagonal
    for i in range(n):
        for j in range(i + 1, n):
            coeff = (W[i, j] - W[j, i]) / 2.0
            if abs(coeff) > 1e-12:
                mv[frozenset({i, j})] = coeff
    return mv


def matrix_from_multivector(mv: Dict[frozenset, float], dim: int) -> np.ndarray:
    """
    Approximate reconstruction of a square matrix from a multivector.
    Only the components that were injected by `multivector_from_matrix` are recovered.
    """
    W = np.zeros((dim, dim), dtype=float)

    # scalar contributes equally to diagonal
    scalar = mv.get(frozenset(), 0.0)
    for i in range(dim):
        W[i, i] = scalar

    # vector part adds deviations to diagonal
    for blade, coeff in mv.items():
        if len(blade) == 1:
            i = next(iter(blade))
            W[i, i] += coeff

    # bivector part fills antisymmetric entries
    for blade, coeff in mv.items():
        if len(blade) == 2:
            i, j = sorted(blade)
            W[i, j] += coeff
            W[j, i] -= coeff
    return W


# ----------------------------------------------------------------------
# Deeper hybrid core functions
# ----------------------------------------------------------------------
def update_weight_matrix_deep(
    W: np.ndarray,
    q: float,
    tokens: Iterable[str],
    eta: float = 0.01,
) -> np.ndarray:
    """
    Deterministic Hebbian‑style update that embeds similarity q
    directly into the geometric algebra layer.
    1. Build a deterministic feature vector from the token set.
    2. Form its outer product Δ = f ⊗ f.
    3. Encode Δ as a multivector, multiply with the scalar similarity multivector {∅: q},
       and add the result (scaled by η) back to W.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("similarity q must be in [0,1]")
    dim = W.shape[0]
    f = token_feature_vector(tokens, dim)
    delta = np.outer(f, f)                     # deterministic outer product
    mv_delta = multivector_from_matrix(delta)  # embed Δ in GA
    mv_q = {frozenset(): q}                    # pure scalar multivector
    mv_update = geometric_product(mv_q, mv_delta)
    W_update = matrix_from_multivector(mv_update, dim)
    return W + eta * W_update


def regret_weighted_strategy_deep(
    actions: List[MathAction],
    similarity_q: float,
    plan_mv: Dict[frozenset, float],
    weight_mv: Dict[frozenset, float],
    temperature: float = 0.5,
) -> Tuple[List[str], np.ndarray]:
    """
    A richer regret engine that:
    1. Couples the plan multivector with the weight multivector via geometric product,
       producing an action‑specific modulation multivector.
    2. Extracts a scalar score from each action's modulation (grade‑0 part).
    3. Blends this with a Fisher‑information based regret term.
    4. Applies temperature‑scaled soft‑max.
    """
    if not actions:
        raise ValueError("actions list cannot be empty")

    # Combine plan and learned weights once; the result modulates every action equally.
    combined_mv = geometric_product(plan_mv, weight_mv)

    # Extract a scalar proxy from the combined multivector (grade‑0)
    scalar_proxy = combined_mv.get(frozenset(), 0.0)

    raw_scores = np.empty(len(actions), dtype=float)
    for idx, a in enumerate(actions):
        base = (a.expected_value - a.cost - a.risk) * similarity_q
        fisher = fisher_score(a.expected_value) * (1.0 - similarity_q)
        # The scalar proxy acts as a learned bias term.
        raw_scores[idx] = base + fisher + scalar_proxy
    probs = _softmax(raw_scores, temperature)
    ids = [a.id for a in actions]
    return ids, probs


def hybrid_step_deep(
    tokens_a: Iterable[str],
    tokens_b: Iterable[str],
    actions: List[MathAction],
    W: np.ndarray,
    plan_mv: Dict[frozenset, float],
) -> Tuple[Dict[frozenset, float], List[str], np.ndarray]:
    """
    Full deep hybrid pipeline:
    1. Compute MinHash similarity q between token sets.
    2. Deterministically update weight matrix using q and token set A.
    3. Encode updated matrix as a multivector (weight_mv).
    4. Produce a regret‑aware action distribution using plan_mv, weight_mv and q.
    Returns the new weight multivector, ordered action IDs, and probabilities.
    """
    # 1. MinHash similarity
    sig_a = signature(tokens_a)
    sig_b = signature(tokens_b)
    q = similarity(sig_a, sig_b)

    # 2. Update weight matrix with deterministic token features from set A
    W_new = update_weight_matrix_deep(W, q, tokens_a)

    # 3. Encode updated matrix as multivector
    weight_mv = multivector_from_matrix(W_new)

    # 4. Regret‑weighted strategy using the deeper GA coupling
    ids, probs = regret_weighted_strategy_deep(
        actions, similarity_q=q, plan_mv=plan_mv, weight_mv=weight_mv
    )
    return weight_mv, ids, probs


# ----------------------------------------------------------------------
# Example initialisation (can be removed or replaced by caller)
# ----------------------------------------------------------------------
def _example_usage():
    # Dummy data
    tokens_a = ["alpha", "beta", "gamma"]
    tokens_b = ["beta", "delta", "epsilon"]
    actions = [
        MathAction(id="a1", expected_value=1.2, cost=0.2, risk=0.1),
        MathAction(id="a2", expected_value=0.8, cost=0.1, risk=0.05),
        MathAction(id="a3", expected_value=1.5, cost=0.3, risk=0.2),
    ]

    dim = 4
    W = np.eye(dim) * 0.5
    # Simple plan multivector: pure scalar 1.0
    plan_mv = {frozenset(): 1.0}

    weight_mv, ids, probs = hybrid_step_deep(tokens_a, tokens_b, actions, W, plan_mv)
    print("Updated weight multivector:", weight_mv)
    print("Action order:", ids)
    print("Probabilities:", probs)


if __name__ == "__main__":
    _example_usage()