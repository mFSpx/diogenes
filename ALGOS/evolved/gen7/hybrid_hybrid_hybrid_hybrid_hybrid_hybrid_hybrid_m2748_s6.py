# DARWIN HAMMER — match 2748, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1213_s8.py (gen6)
# born: 2026-05-29T23:45:36Z

"""Hybrid Regret-Weighted Ternary‑Morphology Fusion (RW‑TD‑M‑Hybrid)

This module merges the core of the two parent algorithms:

* **Parent A** – *Hybrid Regret‑Weighted Ternary Lens*  
  Provides a regret‑weighted probability vector `p` over a set of
  `MathAction`s and a ternary quantisation of that vector.

* **Parent B** – *Hybrid Morphology Similarity & BIC weighting*  
  Supplies a continuous similarity score between `Morphology` objects,
  a normalized Shannon entropy over token streams and a BIC‑derived
  weight.

The **mathematical bridge** is the discrete probability mass
distribution `p`.  We first map `p` to a ternary symbol sequence,
then re‑weight each component of `p` with (i) a morphology‑based
similarity factor, (ii) an entropy‑derived confidence factor and (iii)
a normalized BIC weight derived from the similarity‑induced log‑likelihood.
The resulting hybrid score respects both the regret‑driven decision
policy and the smooth, information‑theoretic pruning schedule of the
second parent.

The public API consists of three representative functions that showcase
the fused workflow.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (identical in both parents)
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
    """Regret‑weighted probability distribution over `actions`."""
    utilities = np.array([a.expected_value - a.cost for a in actions])
    regret = utilities - np.max(utilities)          # negative regrets
    exp_regret = np.exp(regret)
    probabilities = exp_regret / np.sum(exp_regret)
    return probabilities


def ternary_quantisation(probabilities: np.ndarray) -> np.ndarray:
    """
    Quantise a probability vector to the ternary alphabet {‑1, 0, +1}.

    - values > 2/3 → +1
    - values < 1/3 → ‑1
    - otherwise   →  0
    """
    ternary = np.empty_like(probabilities, dtype=int)
    ternary[probabilities > 2.0 / 3.0] = 1
    ternary[probabilities < 1.0 / 3.0] = -1
    mask = (probabilities >= 1.0 / 3.0) & (probabilities <= 2.0 / 3.0)
    ternary[mask] = 0
    return ternary


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Compactness measure used in the similarity score."""
    vol = length * width * height
    surface = math.sqrt(length ** 2 + width ** 2 + height ** 2)
    return (math.pi ** (1 / 3)) * (vol ** (1 / 3)) / surface


def similarity_score(morph_a: Morphology, morph_b: Morphology) -> float:
    """Cosine‑based similarity of two morphology vectors (0 … 1)."""
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
    return (cosine + 1.0) / 2.0


def normalized_shannon_entropy(tokens: List[str]) -> float:
    """Entropy of a token stream normalised to [0, 1]."""
    if not tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    probs = np.array([c / total for c in counts.values()], dtype=float)
    H_raw = -np.sum(probs * np.log2(probs + 1e-12))
    H_max = math.log2(len(counts))
    return H_raw / H_max if H_max > 0 else 0.0


def bic(log_likelihood: float, num_params: int, num_samples: int) -> float:
    """Bayesian Information Criterion."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    return -2.0 * log_likelihood + num_params * math.log(num_samples)


def normalized_bic_weight(bic_value: float, scale: float = 10.0) -> float:
    """Sigmoidal mapping of BIC to a weight in (0, 1]."""
    return 1.0 / (1.0 + math.exp(bic_value / scale))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_action_scores(
    actions: List[MathAction],
    morph_map: Dict[str, Morphology],
    reference_morph: Morphology,
    tokens: List[str],
    num_params: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a hybrid score vector for `actions`.

    Returns
    -------
    scores : np.ndarray
        The fused score for each action.
    ternary : np.ndarray
        The ternary quantisation of the regret‑weighted probabilities.
    """
    # 1️⃣ Regret‑weighted probabilities (Parent A)
    p = regret_weighted_probabilities(actions)

    # 2️⃣ Ternary quantisation (Parent A)
    ternary = ternary_quantisation(p)

    # 3️⃣ Morphology similarity factors (Parent B)
    sim_factors = np.array([
        similarity_score(morph_map.get(a.id, reference_morph), reference_morph)
        for a in actions
    ])

    # 4️⃣ Entropy‑derived confidence (Parent B)
    entropy = normalized_shannon_entropy(tokens)          # scalar ∈[0,1]
    entropy_factor = 1.0 + entropy                        # boost for high entropy

    # 5️⃣ BIC‑based weight (Parent B)
    # Use similarity as a proxy for log‑likelihood.
    log_like = np.log(sim_factors + 1e-12).sum()
    bic_val = bic(log_like, num_params, len(actions))
    bic_weight = normalized_bic_weight(bic_val)

    # 6️⃣ Fuse everything.
    #   - p provides the base decision mass.
    #   - ternary contributes a sign bias (+1,0,‑1) → shift to [0,2] then normalise.
    #   - similarity, entropy and BIC are multiplicative confidence scalars.
    ternary_bias = (ternary + 1) / 2.0          # map {‑1,0,1} → {0,0.5,1}
    scores = p * ternary_bias * sim_factors * entropy_factor * bic_weight
    # Normalise to a proper distribution for downstream use.
    if scores.sum() > 0:
        scores = scores / scores.sum()
    return scores, ternary


def path_signature_from_ternary(ternary: np.ndarray) -> np.ndarray:
    """
    Very lightweight “signature” derived from the cumulative sum of the
    ternary sequence.  In the original parent this would be a full
    iterated‑integral signature; here we keep the spirit while staying
    within the standard‑library constraint.
    """
    return np.cumsum(ternary).astype(float)


def prune_actions(
    actions: List[MathAction],
    scores: np.ndarray,
    schedule_factor: float = 0.5,
) -> List[MathAction]:
    """
    Prune the action list according to a decreasing schedule.

    Parameters
    ----------
    actions : List[MathAction]
        Original candidate actions.
    scores : np.ndarray
        Hybrid scores aligned with `actions`.
    schedule_factor : float ∈ (0, 1]
        Fraction of actions to keep; can be decayed over iterations.

    Returns
    -------
    List[MathAction]
        The retained subset, ordered by descending hybrid score.
    """
    if not (0 < schedule_factor <= 1):
        raise ValueError("schedule_factor must be in (0, 1]")
    keep_n = max(1, int(len(actions) * schedule_factor))
    # sort indices by descending score
    sorted_idx = np.argsort(-scores)[:keep_n]
    return [actions[i] for i in sorted_idx]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny catalogue of actions.
    actions = [
        MathAction(id="A", expected_value=10.0, cost=2.0),
        MathAction(id="B", expected_value=8.0, cost=1.0),
        MathAction(id="C", expected_value=5.0, cost=0.5),
        MathAction(id="D", expected_value=7.0, cost=3.0),
    ]

    # Dummy morphology mapping (one per action).
    morph_map = {
        "A": Morphology(1.0, 0.5, 0.3, 2.0),
        "B": Morphology(0.9, 0.6, 0.4, 1.8),
        "C": Morphology(1.2, 0.4, 0.5, 2.2),
        "D": Morphology(1.1, 0.55, 0.35, 2.0),
    }
    # Reference morphology (e.g., ideal candidate).
    reference = Morphology(1.0, 0.5, 0.4, 2.0)

    # Example token stream for entropy calculation.
    tokens = ["alpha", "beta", "alpha", "gamma", "beta", "delta"]

    # Hybrid scoring.
    scores, tern = hybrid_action_scores(actions, morph_map, reference, tokens)

    # Path signature (purely illustrative).
    signature = path_signature_from_ternary(tern)

    # Prune to the top 50 % of actions.
    retained = prune_actions(actions, scores, schedule_factor=0.5)

    # Output for verification – not required but helpful.
    print("Regret‑weighted probabilities:", regret_weighted_probabilities(actions))
    print("Ternary quantisation:", tern)
    print("Hybrid scores:", scores)
    print("Signature:", signature)
    print("Retained actions:", [a.id for a in retained])