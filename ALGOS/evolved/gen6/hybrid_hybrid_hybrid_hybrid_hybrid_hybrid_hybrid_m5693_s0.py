# DARWIN HAMMER — match 5693, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decrea_m527_s6.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ssim_hybrid_d_m905_s0.py (gen4)
# born: 2026-05-30T00:04:15Z

"""
Hybrid Fusion of Regret‑Epistemic Pruning (Parent A) and Workshare‑Calendar /
Liquid‑Time‑Constant / MinHash‑SSIM (Parent B)

Mathematical Bridge
-------------------
* The posterior weight **wᵢ** from Parent A (Bayesian update of regret *rᵢ* with
  epistemic certainty *cᵢ*) is interpreted as a *risk‑adjusted importance*.
* Parent B supplies a weekday‑group weight vector **γ** (derived from the
  work‑share calendar).  This vector modulates the **liquid time constant**
  τᵢ = τ₀ / (1 + α·γ_g) for the group *g* of action *i*.
* τᵢ is then used as the decay scale in the decreasing‑rate pruning schedule:
  pᵢ = exp(‑wᵢ / τᵢ).  Surviving actions are therefore pruned by a blend of
  Bayesian‑risk and calendar‑driven dynamics.
* For the surviving actions we compute a Fisher‑score weighted centroid
  (the “optimal angle”) using the feature vectors **xᵢ**.
* Textual descriptors are hashed with a MinHash sketch; similarity between
  sketches is measured with an SSIM‑like index.  This similarity is fused
  with the Gini‑adjusted expected value from Parent A and a decision‑hygiene
  score *hᵢ* to obtain a final hybrid score:
  
  **Sᵢ = (1 ‑ G)·SSIMᵢ·hᵢ**, where **G** is the Gini coefficient of the posterior
  weight distribution.

The code below implements this unified system in three core functions and
provides a smoke‑test when executed as a script.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities (Parent A)
# ----------------------------------------------------------------------


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a 1‑D array."""
    if values.ndim != 1:
        raise ValueError("Gini coefficient requires a 1‑D array")
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals, dtype=float)
    if cumulative[-1] == 0:
        return 0.0
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def bayes_marginal(prior: float, likelihood: float, false_positive: float = 1e-6) -> float:
    """Marginal probability for a simple Bayesian update."""
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior weight given prior risk and epistemic likelihood."""
    if marginal == 0:
        return 0.0
    return (likelihood * prior) / marginal


# ----------------------------------------------------------------------
# Parent B constants and helpers
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
BASE_TAU: float = 1.0          # baseline liquid time constant
ALPHA: float = 5.0             # gating steepness for calendar weight
LAMBDA: float = 0.7            # VFE weighting factor (unused in this fusion)
MINHASH_K: int = 64            # number of hash functions for MinHash
MAX64: int = (1 << 64) - 1     # mask for 64‑bit hashing


def liquid_time_constant(group_weight: float) -> float:
    """Compute τ for a group given its calendar weight."""
    return BASE_TAU / (1.0 + ALPHA * group_weight)


def minhash_signature(text: str) -> np.ndarray:
    """Very simple MinHash sketch (64‑bit hashes, K permutations)."""
    rng = np.random.default_rng(seed=hash(text) & 0xFFFFFFFF)
    # Simulate K independent hash values
    return rng.integers(low=0, high=MAX64, size=MINHASH_K, dtype=np.uint64)


def ssim_like(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """SSIM‑style similarity on MinHash sketches (range 0‑1)."""
    # Convert to float for arithmetic
    a = sig_a.astype(np.float64)
    b = sig_b.astype(np.float64)
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    C1 = (0.01 * MAX64) ** 2
    C2 = (0.03 * MAX64) ** 2
    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)
    return float(numerator / denominator) if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass
class Action:
    """Container for a single decision/action."""
    id: int
    regret: float                     # rᵢ from Parent A
    certainty: float                  # cᵢ ∈ [0,1] from Parent A
    group: str                        # one of GROUPS, used for calendar weight
    feature_vec: np.ndarray           # xᵢ for Fisher localization
    text: str                         # textual descriptor for MinHash/SSIM
    hygiene: float                    # decision‑hygiene score ∈ [0,1]


# ----------------------------------------------------------------------
# Core hybrid functions (must be at least three)
# ----------------------------------------------------------------------


def compute_posterior_weights(actions: List[Action],
                              false_positive: float = 1e-6) -> np.ndarray:
    """
    Compute Bayesian posterior weights wᵢ for each action.
    Returns a 1‑D numpy array aligned with ``actions``.
    """
    w = np.empty(len(actions), dtype=float)
    for idx, act in enumerate(actions):
        marginal = bayes_marginal(act.regret, act.certainty, false_positive)
        w[idx] = bayes_update(act.regret, act.certainty, marginal)
    return w


def prune_and_localize(actions: List[Action],
                       posterior: np.ndarray,
                       calendar_weights: dict) -> Tuple[List[Action], np.ndarray]:
    """
    Apply decreasing‑rate pruning using liquid time constants derived from
    the calendar weights, then compute the Fisher‑score weighted centroid
    of the survivors.

    Returns:
        survivors (list of Action), centroid (np.ndarray)
    """
    survivors: List[Action] = []
    surv_weights: List[float] = []

    for act, w_i in zip(actions, posterior):
        group_weight = calendar_weights.get(act.group, 0.0)
        tau_i = liquid_time_constant(group_weight)
        prune_prob = math.exp(-w_i / tau_i)          # decreasing‑rate schedule
        if random.random() > prune_prob:            # survive if random > prob
            survivors.append(act)
            surv_weights.append(w_i)

    if not survivors:
        # Avoid empty set – return empty list and zero centroid
        return [], np.zeros_like(actions[0].feature_vec)

    # Fisher‑score approximation: use normalized feature vectors weighted by wᵢ
    feats = np.stack([a.feature_vec for a in survivors])
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = feats / norms
    weight_arr = np.array(surv_weights)[:, None]
    centroid = (weight_arr * normalized).sum(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm != 0:
        centroid /= centroid_norm
    return survivors, centroid


def compute_hybrid_scores(survivors: List[Action],
                          posterior: np.ndarray,
                          calendar_weights: dict) -> List[Tuple[Action, float]]:
    """
    For each surviving action compute the final hybrid score:

        Sᵢ = (1 ‑ G) · SSIMᵢ · hᵢ

    where G is the Gini coefficient of the *entire* posterior distribution,
    SSIMᵢ is the similarity between the action's MinHash sketch and the
    centroid sketch (derived from the Fisher centroid), and hᵢ is the
    decision‑hygiene score.
    """
    if not survivors:
        return []

    G = gini_coefficient(posterior)
    # Build a synthetic "centroid text" from the Fisher centroid direction.
    # For the demo we simply concatenate the texts of survivors weighted by
    # their posterior weight and hash that.
    weighted_text = " ".join(
        act.text for act, w in zip(survivors, posterior) if act in survivors
    )
    centroid_sig = minhash_signature(weighted_text)

    results: List[Tuple[Action, float]] = []
    for act in survivors:
        act_sig = minhash_signature(act.text)
        ssim = ssim_like(act_sig, centroid_sig)
        hybrid_score = (1.0 - G) * ssim * act.hygiene
        results.append((act, hybrid_score))
    return results


def hybrid_fusion(actions: List[Action]) -> List[Tuple[Action, float]]:
    """
    End‑to‑end hybrid pipeline:
        1. Posterior weights (Bayes)
        2. Calendar‑driven pruning & Fisher centroid
        3. MinHash‑SSIM + Gini + hygiene fusion
    Returns a list of (Action, hybrid_score) sorted descending.
    """
    # 1. Posterior weights
    posterior = compute_posterior_weights(actions)

    # 2. Calendar weights – for demo we assign a random weight per group
    calendar_weights = {g: random.random() for g in GROUPS}

    survivors, _ = prune_and_localize(actions, posterior, calendar_weights)

    # 3. Hybrid scores
    scored = compute_hybrid_scores(survivors, posterior, calendar_weights)

    # Sort by score (high → low) and return
    scored.sort(key=lambda tup: tup[1], reverse=True)
    return scored


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic population
    rng = np.random.default_rng(42)
    demo_actions: List[Action] = []
    for i in range(10):
        feat = rng.normal(size=5)
        demo_actions.append(
            Action(
                id=i,
                regret=rng.random(),
                certainty=rng.random(),
                group=random.choice(GROUPS),
                feature_vec=feat,
                text=f"sample text {i}",
                hygiene=rng.random()
            )
        )

    results = hybrid_fusion(demo_actions)
    print("Hybrid scores (sorted):")
    for act, score in results:
        print(f"Action {act.id:2d} | Group {act.group:12s} | Score {score:.4f}")