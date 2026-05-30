# DARWIN HAMMER — match 3788, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hdc_se_m1184_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_decision_hygi_m338_s0.py (gen4)
# born: 2026-05-29T23:51:39Z

"""Hybrid Bandit‑RBF‑HDC & Decision‑Sheaf Entropy Fusion

This module unifies two Darwin‑Hammer parents:

* **Parent A** – a contextual multi‑armed bandit enriched with an RBF surrogate
  and hyperdimensional computing (HDC) utilities (`morphology_hv`,
  `sparse_wta_hv`, `hybrid_priority`).

* **Parent B** – a weekday‑dependent weight vector that allocates textual
  regex‑derived features across semantic groups and evaluates their Shannon
  entropy.

**Mathematical bridge**

1. The bandit supplies an *expected reward* `r̂` for a given
   `(context, action)` pair via an RBF surrogate.
2. The HDC side encodes the same context into a bipolar hypervector `h`.
   A sparse winner‑take‑all hypervector `w` is built from the scalar
   reward `r̂`; their dot‑product `s = h·w` measures hyperdimensional
   similarity.
3. The decision‑sheaf side produces a weekday weight vector `α ∈ ℝ^G`
   (G = number of groups) and allocates regex‑extracted feature counts
   `c ∈ ℝ^G` into a normalized distribution `p = α ⊙ c / Σ(α ⊙ c)`.
   Its Shannon entropy `H(p)` quantifies the spread of information across
   groups.
4. The final hybrid priority fuses the two domains:


priority = s * (1 + H(p))


Thus the bandit‑driven similarity is amplified (or attenuated) by the
information‑theoretic richness of the textual features.

The module provides three public functions that showcase this fusion:
`morphology_hv`, `sparse_wta_hv`, and `hybrid_decision_priority`.  A small
smoke‑test runs when the file is executed as a script."""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Sequence, Dict, Callable

import datetime as dt
import re

import numpy as np

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = Sequence[float]
BipolarVector = List[int]

# ----------------------------------------------------------------------
# Bandit / RBF core (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


class RBFSurrogate:
    """
    Simple radial‑basis‑function surrogate.
    Stores a list of centres C_i ∈ ℝ^d, associated weights w_i and a shared
    bandwidth σ.
    Prediction for x is Σ_i w_i * exp(-||x-C_i||² / (2σ²)).
    """

    def __init__(self, centres: np.ndarray, weights: np.ndarray, sigma: float = 1.0):
        assert centres.shape[0] == weights.shape[0], "centres and weights must align"
        self.centres = centres  # shape (N, d)
        self.weights = weights  # shape (N,)
        self.sigma = sigma

    def predict(self, x: np.ndarray) -> float:
        diff = self.centres - x  # (N, d)
        d2 = np.einsum("ij,ij->i", diff, diff)  # squared Euclidean distances, shape (N,)
        kernels = np.exp(-d2 / (2 * self.sigma ** 2))
        return float(np.dot(self.weights, kernels))


# ----------------------------------------------------------------------
# Hyperdimensional utilities (Parent A)
# ----------------------------------------------------------------------
def morphology_hv(scalars: Vector, dim: int = 10_000) -> BipolarVector:
    """
    Encode a list of real scalars into a bipolar hypervector of length `dim`.
    Each scalar is projected onto a random Gaussian vector; the sign of the
    sum determines the bipolar (+1 / -1) entry.
    """
    rng = np.random.default_rng(seed=42)
    proj = rng.normal(size=(dim, len(scalars)))  # (dim, d)
    weighted = np.dot(proj, np.array(scalars))  # (dim,)
    hv = np.where(weighted >= 0, 1, -1).tolist()
    return hv


def sparse_wta_hv(scores: Vector, k: int, dim: int = 10_000) -> BipolarVector:
    """
    Create a sparse winner‑take‑all hypervector.
    The `k` largest scores obtain a +1, all others -1.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(scores):
        raise ValueError("k cannot exceed number of scores")
    # initialise all -1
    hv = [-1] * dim
    # map each score to a random position (deterministic via seed)
    rng = np.random.default_rng(seed=123)
    positions = rng.choice(dim, size=len(scores), replace=False)
    # select top‑k positions
    top_idx = np.argpartition(scores, -k)[-k:]
    for idx in top_idx:
        hv[positions[idx]] = 1
    return hv


def hyperdot(hv1: BipolarVector, hv2: BipolarVector) -> int:
    """Signed dot product between two bipolar hypervectors."""
    if len(hv1) != len(hv2):
        raise ValueError("Hypervectors must have equal length")
    return sum(a * b for a, b in zip(hv1, hv2))


# ----------------------------------------------------------------------
# Decision‑Sheaf utilities (Parent B)
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
    return (dt.date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector that varies with the day‑of‑week.
    For reproducibility a deterministic pseudo‑random generator seeded by
    the weekday index is used.
    """
    rng = np.random.default_rng(seed=dow)
    weights = rng.random(len(groups))
    return weights / weights.sum()


# Simple regex‑to‑group mapping for demonstration
_REGEX_GROUP_MAP: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"\bAI\b", re.IGNORECASE), 0),   # codex
    (re.compile(r"\bLLM\b", re.IGNORECASE), 1),  # groq
    (re.compile(r"\bmodel\b", re.IGNORECASE), 2),# cohere
    (re.compile(r"\bopen[- ]source\b", re.IGNORECASE), 3), # local_models
]


def extract_group_counts(text: str) -> np.ndarray:
    """
    Count occurrences of each regex pattern and return a vector of length
    `len(GROUPS)`.  Missing groups get a count of zero.
    """
    counts = np.zeros(len(GROUPS), dtype=float)
    for pat, grp_idx in _REGEX_GROUP_MAP:
        matches = pat.findall(text)
        counts[grp_idx] += len(matches)
    return counts


def shannon_entropy(probs: np.ndarray) -> float:
    """
    Compute Shannon entropy (base e) of a probability distribution.
    Zero probabilities are ignored.
    """
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return -float(np.sum(probs * np.log(probs)))


# ----------------------------------------------------------------------
# Hybrid Fusion
# ----------------------------------------------------------------------
def hybrid_decision_priority(
    context: Vector,
    action_one_hot: Vector,
    text: str,
    rbf_model: RBFSurrogate,
    dim_hv: int = 10_000,
    wta_k: int = 3,
) -> float:
    """
    Compute a unified priority value that blends:
    * Bandit‑RBF expected reward (via `rbf_model`),
    * Hyperdimensional similarity (`hyperdot` of morphology HV and sparse WTA HV),
    * Weekday‑weighted feature entropy derived from `text`.

    Steps:
    1. Concatenate context and action_one_hot, predict reward `r̂`.
    2. Encode `context` into a bipolar hypervector `h`.
    3. Build a sparse WTA hypervector `w` from the scalar `[r̂]`.
    4. Similarity `s = hyperdot(h, w)`.
    5. Determine today's weekday weight vector `α`.
    6. Extract regex‑based group counts `c` from `text`.
    7. Allocate counts: `a = α ⊙ c`; normalise to a probability vector `p`.
    8. Entropy `H = shannon_entropy(p)`.
    9. Return `priority = s * (1 + H)`.
    """
    # 1. RBF prediction
    x = np.concatenate([np.asarray(context, dtype=float), np.asarray(action_one_hot, dtype=float)])
    r_hat = rbf_model.predict(x)

    # 2. Morphology hypervector
    h = morphology_hv(context, dim=dim_hv)

    # 3. Sparse WTA hypervector from reward scalar
    w = sparse_wta_hv([r_hat], k=wta_k, dim=dim_hv)

    # 4. Hyperdimensional similarity
    s = hyperdot(h, w)

    # 5. Weekday weight vector
    today = dt.date.today()
    dow = doomsday(today.year, today.month, today.day)
    alpha = weekday_weight_vector(GROUPS, dow)  # shape (G,)

    # 6. Feature extraction
    c = extract_group_counts(text)  # shape (G,)

    # 7. Allocation and normalisation
    a = alpha * c
    total = a.sum()
    if total > 0:
        p = a / total
    else:
        p = np.zeros_like(a)

    # 8. Entropy
    H = shannon_entropy(p)

    # 9. Final priority
    priority = float(s) * (1.0 + H)
    return priority


# ----------------------------------------------------------------------
# Simple demonstration / smoke test
# ----------------------------------------------------------------------
def _demo():
    # Dummy context (3‑dimensional) and one‑hot action (2 actions)
    context = [0.2, -0.5, 1.1]
    action_one_hot = [1, 0]  # action 0 selected

    # Build a tiny RBF surrogate with random centres/weights
    rng = np.random.default_rng(seed=7)
    centres = rng.normal(size=(5, len(context) + len(action_one_hot)))  # 5 centres
    weights = rng.uniform(-1, 1, size=5)
    rbf = RBFSurrogate(centres, weights, sigma=0.8)

    # Sample text containing a few of the regex keywords
    sample_text = (
        "The AI community is exploring large LLM models. "
        "Open-source initiatives accelerate model development."
    )

    priority = hybrid_decision_priority(
        context=context,
        action_one_hot=action_one_hot,
        text=sample_text,
        rbf_model=rbf,
        dim_hv=4096,
        wta_k=1,
    )
    print(f"Hybrid priority: {priority:.4f}")


if __name__ == "__main__":
    _demo()