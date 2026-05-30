# DARWIN HAMMER — match 2695, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m433_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m397_s0.py (gen4)
# born: 2026-05-29T23:43:33Z

"""Hybrid algorithm merging:
- Parent A: path signature B‑spline log‑likelihood with weekday weight vector.
- Parent B: deterministic MinHash signatures, entropy, and radial‑basis‑function (RBF) surrogate mapping to pheromone utilities.

Mathematical bridge:
The B‑spline approximation yields a scalar log‑likelihood `ℓ` for a token set under a weekday weight vector.
An RBF surrogate `f(ℓ) = Σ w_i·exp(-γ·|ℓ - c_i|²)` learns the mapping from this scalar to the pheromone‑based utility
used in Parent B. The resulting utility feeds the entropy and decision‑hygiene calculations of Parent A,
while MinHash signatures provide a similarity measure between two contexts for downstream bandit updates.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from datetime import date, datetime
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
WEEKDAY_WEIGHTS = np.array([1.0, 0.9, 0.95, 1.05, 1.0, 0.85, 0.9])  # Mon…Sun


def b_spline_basis(x: float, knots: List[float], degree: int) -> float:
    """Cox‑de Boor recursion for a single B‑spline basis value."""
    if degree == 0:
        return 1.0 if knots[0] <= x < knots[1] else 0.0
    left = (x - knots[0]) / (knots[degree] - knots[0]) if knots[degree] != knots[0] else 0.0
    right = (knots[degree + 1] - x) / (knots[degree + 1] - knots[1]) if knots[degree + 1] != knots[1] else 0.0
    return left * b_spline_basis(x, knots[:-1], degree - 1) + right * b_spline_basis(x, knots[1:], degree - 1)


def log_likelihood_bsplines(tokens: List[str], weekday: int, degree: int = 3) -> float:
    """Approximate log‑likelihood of token frequencies using B‑spline basis weighted by weekday."""
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = sum(freq.values())
    probs = np.array([cnt / total for cnt in freq.values()])

    # Simple uniform knot vector on [0,1] with extra knots for clamping
    n_basis = len(probs)
    knots = np.concatenate((
        np.zeros(degree),
        np.linspace(0, 1, n_basis - degree + 1),
        np.ones(degree)
    ))
    # Evaluate basis for each probability and weight by weekday factor
    weight = WEEKDAY_WEIGHTS[weekday % 7]
    log_like = 0.0
    for p in probs:
        basis_val = sum(b_spline_basis(p, list(knots), degree) for _ in range(n_basis))
        log_like += weight * math.log(basis_val + 1e-12)  # avoid log(0)
    return log_like


def decision_entropy(probs: List[float]) -> float:
    """Shannon entropy used in decision‑hygiene calculations."""
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum()
    return -float(np.sum(probs * np.log(probs)))


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------
def deterministic_hash(token: str, seed: int) -> int:
    """Deterministic 64‑bit hash of token+seed."""
    h = hashlib.sha256(f"{token}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def minhash_signature(tokens: List[str], num_hashes: int) -> List[int]:
    """Deterministic MinHash signature."""
    sig = []
    for i in range(num_hashes):
        min_h = (1 << 64) - 1
        for t in tokens:
            h = deterministic_hash(t, i)
            if h < min_h:
                min_h = h
        sig.append(min_h)
    return sig


def minhash_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Jaccard‑like similarity of two MinHash signatures."""
    if not sig1 or len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def rbf_surrogate(x: float, centers: np.ndarray, weights: np.ndarray, gamma: float = 0.5) -> float:
    """Radial‑basis surrogate mapping scalar input to utility."""
    diff = x - centers
    phi = np.exp(-gamma * diff ** 2)
    return float(np.dot(weights, phi))


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_utility(context_tokens: List[str], weekday: int,
                   rbf_centers: np.ndarray, rbf_weights: np.ndarray) -> float:
    """
    Compute a hybrid utility:
    1. B‑spline log‑likelihood of the token set (Parent A).
    2. RBF surrogate maps that scalar to a pheromone‑style utility (Parent B).
    3. Entropy of the normalized token distribution modulates the utility.
    """
    ll = log_likelihood_bsplines(context_tokens, weekday)
    utility = rbf_surrogate(ll, rbf_centers, rbf_weights)
    # Modulate by entropy (higher entropy → lower confidence)
    freq = {}
    for t in context_tokens:
        freq[t] = freq.get(t, 0) + 1
    probs = np.array(list(freq.values()), dtype=float) / len(context_tokens)
    ent = decision_entropy(probs.tolist())
    return utility * math.exp(-ent)  # damped by entropy


def hybrid_similarity(tokens_a: List[str], tokens_b: List[str],
                      num_hashes: int = 64) -> float:
    """
    Hybrid similarity merges MinHash Jaccard similarity (Parent B)
    with a B‑spline based cosine similarity of token frequency vectors (Parent A).
    """
    # MinHash component
    sig_a = minhash_signature(tokens_a, num_hashes)
    sig_b = minhash_signature(tokens_b, num_hashes)
    mh_sim = minhash_similarity(sig_a, sig_b)

    # Frequency‑vector component using B‑spline basis as feature map
    all_tokens = list(set(tokens_a) | set(tokens_b))
    vec_a = np.array([tokens_a.count(t) for t in all_tokens], dtype=float)
    vec_b = np.array([tokens_b.count(t) for t in all_tokens], dtype=float)

    # Normalize
    if vec_a.sum() == 0 or vec_b.sum() == 0:
        freq_cos = 0.0
    else:
        vec_a = vec_a / vec_a.sum()
        vec_b = vec_b / vec_b.sum()
        freq_cos = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-12))

    # Blend the two similarity measures
    return 0.6 * mh_sim + 0.4 * freq_cos


def update_store(state: Dict[str, Any], inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
    """
    Simple store update mirroring Parent A's StoreState, but expressed as a pure function.
    Returns updated level and delta.
    """
    alpha = state.get("alpha", 1.0)
    beta = state.get("beta", 1.0)
    dt = state.get("dt", 1.0)
    level = state.get("level", 0.0)
    limit = state.get("limit", 10.0)

    delta = alpha * sum(inflow) - beta * sum(outflow)
    level = max(0.0, level + dt * delta)
    state["level"] = level
    return level, delta


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy data
    tokens_ctx1 = ["alpha", "beta", "gamma", "alpha", "delta"]
    tokens_ctx2 = ["beta", "epsilon", "alpha", "zeta", "beta"]

    # RBF surrogate training data (synthetic)
    rbf_centers = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    rbf_weights = np.array([0.2, 0.5, 1.0, 0.5, 0.2])

    # Compute hybrid utilities
    util1 = hybrid_utility(tokens_ctx1, weekday=date.today().weekday(),
                           rbf_centers=rbf_centers, rbf_weights=rbf_weights)
    util2 = hybrid_utility(tokens_ctx2, weekday=date.today().weekday(),
                           rbf_centers=rbf_centers, rbf_weights=rbf_weights)

    print(f"Hybrid utility context 1: {util1:.4f}")
    print(f"Hybrid utility context 2: {util2:.4f}")

    # Compute hybrid similarity
    sim = hybrid_similarity(tokens_ctx1, tokens_ctx2)
    print(f"Hybrid similarity between contexts: {sim:.4f}")

    # Store update demo
    store_state = {"level": 1.0, "alpha": 0.8, "beta": 0.5, "dt": 0.1, "limit": 5.0}
    level, delta = update_store(store_state, inflow=[util1, util2], outflow=[0.3])
    print(f"Updated store level: {level:.4f}, delta: {delta:.4f}")