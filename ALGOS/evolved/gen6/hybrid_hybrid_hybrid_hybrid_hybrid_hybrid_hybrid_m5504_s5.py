# DARWIN HAMMER — match 5504, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_route_m2082_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_indy_l_m399_s0.py (gen5)
# born: 2026-05-30T00:02:30Z

"""Hybrid Algorithm combining Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_ternary_route_m2082_s2.py)
and Parent B (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_indy_l_m399_s0.py).

Mathematical Bridge
-------------------
Parent A provides a deterministic weekday‑dependent allocation vector
`w(d)` for the four model groups and a MinHash similarity based on 64‑bit
signatures.  
Parent B introduces a Fisher‑information weighting `F` derived from the
empirical token distribution and uses it to modulate allocation and
similarity calculations.

The hybrid builds the **allocation sheaf** `a = w(d) ⊙ F` (element‑wise
product) and maps it through the coboundary operator (pairwise differences)
to obtain edge‑wise residuals.  These residuals weight the MinHash similarity,
yielding a Fisher‑adjusted similarity `S_f`.  The same weighting is applied
inside the variational free‑energy functional, producing a Fisher‑regularized
energy `F_FE`.

The three core functions below demonstrate this integration:
1. `hybrid_allocation` – constructs `a = w(d) ⊙ F`.
2. `fisher_weighted_minhash` – computes `S_f` using edge residuals.
3. `fisher_variational_free_energy` – evaluates the regularized free energy.

All operations rely only on NumPy and the standard library, satisfying the
execution constraints.
"""

import math
import random
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared constants and helpers (from both parents)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Tuple[str, ...], dow: int) -> np.ndarray:
    """
    Produce a normalized weight vector for *groups* based on the weekday ``dow``.
    The construction follows Parent A: a sinusoidal modulation around a
    uniform base.
    """
    n = len(groups)
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def _hash(seed: int, token: str) -> int:
    """Hash a token with a 4‑byte seed using a simple mixing function."""
    h = np.uint64(seed)
    for c in token:
        h = np.uint64(h ^ ord(c))
        h = np.uint64(h * 0x100000001B3)
        h &= MAX64
    return int(h)


def minhash_similarity(token_a: str, token_b: str, seed: int = 0xDEADBEEF) -> float:
    """
    Compute a Jaccard‑like similarity from two MinHash signatures.
    The result is in [0, 1].
    """
    ha = _hash(seed, token_a)
    hb = _hash(seed, token_b)
    diff = bin(ha ^ hb).count("1")
    return 1.0 - diff / 64.0


def variational_free_energy(q: np.ndarray, p: np.ndarray) -> float:
    """
    Classical variational free energy:  F = KL(q‖p) - H(q)
    where KL is the Kullback‑Leibler divergence and H is the entropy of q.
    Both q and p must be valid probability vectors.
    """
    eps = np.finfo(float).eps
    q = np.clip(q, eps, 1.0)
    p = np.clip(p, eps, 1.0)
    kl = np.sum(q * np.log(q / p))
    entropy = -np.sum(q * np.log(q))
    return kl - entropy


# ----------------------------------------------------------------------
# Fisher‑information utilities (originating from Parent B)
# ----------------------------------------------------------------------
def token_frequencies(tokens: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute empirical frequencies of each distinct token.
    Returns (probabilities, ordered token list).
    """
    if not tokens:
        return np.array([], dtype=np.float64), []
    uniq, counts = np.unique(tokens, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    return probs, uniq.tolist()


def fisher_information(probs: np.ndarray) -> np.ndarray:
    """
    Simple diagonal Fisher information for a categorical distribution:
    I_i = 1 / p_i   (ignoring the constant factor).
    """
    eps = np.finfo(float).eps
    probs = np.clip(probs, eps, 1.0)
    return 1.0 / probs


def fisher_weights_from_tokens(tokens: List[str]) -> np.ndarray:
    """
    Build a Fisher weighting vector aligned with GROUPS.
    The function maps token‑level Fisher information onto the group level
    by averaging over tokens that contain the group name as a substring.
    If a group receives no tokens, a neutral weight of 1.0 is used.
    """
    probs, uniq = token_frequencies(tokens)
    if probs.size == 0:
        return np.ones(len(GROUPS), dtype=np.float64)

    fisher = fisher_information(probs)

    # Map each group to the mean Fisher weight of tokens that mention it.
    group_weights = np.empty(len(GROUPS), dtype=np.float64)
    for i, grp in enumerate(GROUPS):
        mask = np.array([grp in t for t in uniq])
        if mask.any():
            group_weights[i] = fisher[mask].mean()
        else:
            group_weights[i] = 1.0
    # Normalise to keep scale comparable with weekday weights
    group_weights /= group_weights.sum()
    return group_weights


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_allocation(
    groups: Tuple[str, ...],
    dow: int,
    token_corpus: List[str],
) -> np.ndarray:
    """
    Combine the weekday‑dependent allocation vector with Fisher‑information
    weighting derived from a token corpus.

    a_i = w_i(dow) * F_i   (element‑wise product)
    The resulting vector is re‑normalised to sum to 1.
    """
    w = weekday_weight_vector(groups, dow)          # Parent A component
    F = fisher_weights_from_tokens(token_corpus)    # Parent B component
    a = w * F
    a /= a.sum()
    return a.astype(np.float64)


def _coboundary_operator(vec: np.ndarray) -> np.ndarray:
    """
    Simple coboundary for a complete graph on len(vec) nodes.
    Returns the vector of pairwise differences (i<j) ordered lexicographically.
    """
    n = vec.shape[0]
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(vec[j] - vec[i])
    return np.array(diffs, dtype=np.float64)


def fisher_weighted_minhash(
    token_a: str,
    token_b: str,
    groups: Tuple[str, ...],
    dow: int,
    token_corpus: List[str],
    seed: int = 0xDEADBEEF,
) -> float:
    """
    Compute a MinHash similarity that is modulated by the Fisher‑adjusted
    allocation sheaf.

    Steps:
    1. Build allocation vector a = hybrid_allocation(...).
    2. Apply the coboundary operator to obtain edge residuals r.
    3. Derive edge weights w_edge = 1 + |r| (ensuring positivity).
    4. Compute raw MinHash similarity s0 = minhash_similarity(...).
    5. Return weighted similarity s = s0 * mean(w_edge).
    """
    a = hybrid_allocation(groups, dow, token_corpus)
    r = _coboundary_operator(a)
    w_edge = 1.0 + np.abs(r)          # positive edge weights
    s0 = minhash_similarity(token_a, token_b, seed)
    s = s0 * w_edge.mean()
    return _pct(s)


def fisher_variational_free_energy(
    q: np.ndarray,
    p: np.ndarray,
    groups: Tuple[str, ...],
    dow: int,
    token_corpus: List[str],
) -> float:
    """
    Regularised free energy where the prior distribution p is first
    transformed by the Fisher‑adjusted allocation sheaf.

    p' = normalize( p * hybrid_allocation(...) )
    Returns the classic variational free energy between q and p'.
    """
    a = hybrid_allocation(groups, dow, token_corpus)
    p_prime_unnorm = p * a
    if p_prime_unnorm.sum() == 0:
        p_prime = np.full_like(p, 1.0 / len(p))
    else:
        p_prime = p_prime_unnorm / p_prime_unnorm.sum()
    fe = variational_free_energy(q, p_prime)
    return _pct(fe)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Deterministic weekday for today
    today = datetime.utcnow()
    dow_today = doomsday(today.year, today.month, today.day)

    # Example token corpus (could be any text split into tokens)
    corpus = [
        "codex_alpha", "groq_beta", "cohere_gamma",
        "local_models_delta", "codex_epsilon", "groq_zeta"
    ]

    # Hybrid allocation demo
    alloc = hybrid_allocation(GROUPS, dow_today, corpus)
    print("Hybrid allocation vector:", alloc)

    # Fisher‑weighted MinHash similarity demo
    s = fisher_weighted_minhash(
        "codex_alpha", "groq_beta", GROUPS, dow_today, corpus
    )
    print("Fisher‑weighted MinHash similarity:", s)

    # Variational free energy demo
    q = np.array([0.25, 0.25, 0.25, 0.25])
    p = np.array([0.4, 0.3, 0.2, 0.1])
    fe = fisher_variational_free_energy(q, p, GROUPS, dow_today, corpus)
    print("Fisher‑regularized variational free energy:", fe)