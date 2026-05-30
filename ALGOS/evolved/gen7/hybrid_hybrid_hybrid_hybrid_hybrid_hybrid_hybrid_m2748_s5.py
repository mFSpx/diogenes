# DARWIN HAMMER — match 2748, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1213_s8.py (gen6)
# born: 2026-05-29T23:45:36Z

"""Hybrid Regret‑Weighted Similarity‑Entropy Pruning (RW‑SE‑P).

This module fuses the core mathematics of the two parent algorithms:

* **Parent A** – *Hybrid Regret‑Weighted Ternary Lens*:
  - Computes a regret‑weighted probability vector `p` over a set of actions.
  - Quantises a real‑valued vector onto the ternary alphabet {‑1,0,+1}.

* **Parent B** – *Hybrid Morphology Similarity & Entropy*:
  - Provides a cosine‑based similarity `S(i,j)` between morphology descriptors.
  - Supplies a normalized Shannon entropy `H(tokens)` and a BIC‑derived weight.

**Mathematical bridge** – Both parents operate on *discrete probability‑mass samples*.
`p` (Parent A) and the similarity scores `s_i = S(morph_i, μ)` (Parent B) live in the same
probability simplex after normalisation.  By element‑wise multiplication
`h_i = p_i · s_i` we obtain a *joint* distribution that respects regret weighting
and morphological similarity.  This joint vector is then:

1. **Quantised** onto the ternary alphabet (sign‑quantisation).
2. **Encoded** as a simple path signature (cumulative sum) to capture ordering.
3. **Pruned** using a BIC‑derived weight that is modulated by the normalized
   Shannon entropy of an auxiliary token set.

The resulting pipeline yields a single scalar “pruned score’’ for each action
that blends regret, shape similarity, information‑theoretic uncertainty and
model‑complexity penalties.

"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared between parents)
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


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
def regret_weighted_probabilities(actions: List[MathAction]) -> np.ndarray:
    """Regret‑weighted softmax over (expected_value – cost)."""
    utilities = np.array([a.expected_value - a.cost for a in actions])
    regret = utilities - np.max(utilities)          # shift for numerical stability
    exp_regret = np.exp(regret)
    probabilities = exp_regret / np.sum(exp_regret)
    return probabilities


def ternary_quantisation(values: np.ndarray,
                         low: float = 1.0 / 3.0,
                         high: float = 2.0 / 3.0) -> np.ndarray:
    """
    Quantise a real‑valued vector onto the ternary alphabet {‑1, 0, +1}.

    * `value <= low`   → -1
    * `low < value < high` → 0
    * `value >= high` → +1
    """
    q = np.zeros_like(values, dtype=int)
    q[values <= low] = -1
    q[values >= high] = +1
    # values in (low, high) remain 0
    return q


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Compactness measure used in the similarity vector."""
    vol = length * width * height
    surface = math.sqrt(length ** 2 + width ** 2 + height ** 2)
    if surface == 0:
        return 0.0
    return (math.pi ** (1 / 3)) * (vol ** (1 / 3)) / surface


def similarity_score(morph_a: Morphology, morph_b: Morphology) -> float:
    """Cosine‑based similarity in a 5‑dimensional feature space."""
    vec_a = np.array([
        morph_a.length,
        morph_a.width,
        morph_a.height,
        morph_a.mass,
        sphericity_index(morph_a.length, morph_a.width, morph_a.height),
    ])
    vec_b = np.array([
        morph_b.length,
        morph_b.width,
        morph_b.height,
        morph_b.mass,
        sphericity_index(morph_b.length, morph_b.width, morph_b.height),
    ])
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return (cosine + 1.0) / 2.0          # map from [-1,1] to [0,1]


def normalized_shannon_entropy(tokens: List[str]) -> float:
    """Entropy of a token multiset, normalised to [0,1]."""
    if not tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    probs = np.array([c / total for c in counts.values()], dtype=float)
    H_raw = -np.sum(probs * np.log2(probs + 1e-12))
    H_max = math.log2(len(counts)) if len(counts) > 1 else 0.0
    return H_raw / H_max if H_max > 0 else 0.0


def bic(log_likelihood: float, num_params: int, num_samples: int) -> float:
    """Bayesian Information Criterion."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    return -2.0 * log_likelihood + num_params * math.log(num_samples)


def normalized_bic_weight(bic_value: float, scale: float = 10.0) -> float:
    """Sigmoidal mapping of BIC to a weight in (0,1]."""
    return 1.0 / (1.0 + math.exp(bic_value / scale))


# ----------------------------------------------------------------------
# Hybrid operations (the fused core)
# ----------------------------------------------------------------------
def hybrid_regret_similarity(actions: List[MathAction],
                             morphologies: List[Morphology]) -> np.ndarray:
    """
    1. Compute regret‑weighted probabilities `p`.
    2. Compute similarity of each morphology to the centroid morphology `μ`.
    3. Combine element‑wise: `h_i = p_i * s_i`.
    Returns the joint vector `h`.
    """
    if len(actions) != len(morphologies):
        raise ValueError("actions and morphologies must have the same length")

    p = regret_weighted_probabilities(actions)                     # (n,)
    # centroid morphology (mean of each attribute)
    lengths = np.mean([m.length for m in morphologies])
    widths  = np.mean([m.width  for m in morphologies])
    heights = np.mean([m.height for m in morphologies])
    masses  = np.mean([m.mass   for m in morphologies])
    centroid = Morphology(lengths, widths, heights, masses)

    s = np.array([similarity_score(m, centroid) for m in morphologies])  # (n,)
    h = p * s
    # re‑normalise to keep a proper probability simplex (optional but convenient)
    if np.sum(h) > 0:
        h = h / np.sum(h)
    return h


def hybrid_ternary_signature(joint_vector: np.ndarray) -> np.ndarray:
    """
    Quantise the joint vector onto the ternary alphabet and compute a simple
    path signature (cumulative sum) that preserves ordering information.
    Returns the path signature as a 1‑D array of integers.
    """
    ternary = ternary_quantisation(joint_vector)          # values in {‑1,0,+1}
    signature = np.cumsum(ternary)                        # path signature
    return signature


def hybrid_pruned_score(actions: List[MathAction],
                        morphologies: List[Morphology],
                        tokens: List[str],
                        num_params: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full hybrid pipeline returning:
    * `pruned_scores` – a weighted scalar for each action.
    * `signature`      – the ternary path signature used for downstream tasks.

    Steps:
    1. Joint vector `h` via regret‑similarity.
    2. Path signature from ternary quantisation of `h`.
    3. Approximate log‑likelihood with `log(h_i + ε)`.
    4. BIC weight modulated by normalized token entropy.
    5. Final pruned score = h_i * weight.
    """
    h = hybrid_regret_similarity(actions, morphologies)          # (n,)
    signature = hybrid_ternary_signature(h)                     # (n,)

    # Avoid log(0) – add a tiny epsilon
    eps = 1e-12
    log_likelihoods = np.log(h + eps)                           # (n,)

    # BIC per action (treat each action as an independent observation)
    num_samples = len(actions)
    bic_vals = np.array([bic(ll, num_params, num_samples) for ll in log_likelihoods])

    # Entropy‑scaled BIC weight (same weight for all actions)
    entropy = normalized_shannon_entropy(tokens)                # scalar in [0,1]
    bic_weight = normalized_bic_weight(np.mean(bic_vals)) * (1.0 - entropy)

    pruned_scores = h * bic_weight                               # (n,)
    return pruned_scores, signature


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic scenario
    actions = [
        MathAction(id="A1", expected_value=10.0, cost=2.0),
        MathAction(id="A2", expected_value=8.0,  cost=1.0),
        MathAction(id="A3", expected_value=6.0,  cost=0.5),
    ]

    morphologies = [
        Morphology(length=1.0, width=2.0, height=3.0, mass=4.0),
        Morphology(length=1.5, width=2.5, height=3.5, mass=4.5),
        Morphology(length=0.9, width=1.8, height=2.7, mass=3.6),
    ]

    tokens = ["alpha", "beta", "alpha", "gamma", "beta", "beta"]

    scores, sig = hybrid_pruned_score(actions, morphologies, tokens)

    print("Regret‑Similarity joint vector (normalized):")
    print(hybrid_regret_similarity(actions, morphologies))
    print("\nTernary path signature:")
    print(sig)
    print("\nPruned scores:")
    for act, sc in zip(actions, scores):
        print(f"  Action {act.id}: {sc:.6f}")

    # Simple sanity checks – ensure no NaNs and values lie in [0,1]
    assert not np.isnan(scores).any()
    assert (scores >= 0).all() and (scores <= 1).all()
    print("\nSmoke test passed.")