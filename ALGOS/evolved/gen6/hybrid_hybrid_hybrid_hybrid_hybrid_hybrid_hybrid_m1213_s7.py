# DARWIN HAMMER — match 1213, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m959_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rlct_g_m436_s1.py (gen5)
# born: 2026-05-29T23:34:36Z

"""Hybrid Algorithm: hybrid_morphology_regret_rlct
Integrates:
- Parent A: `hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s6.py` (morphology vectors, SSIM‑like similarity, token entropy)
- Parent B: `hybrid_hybrid_hybrid_rlct_grokking_hybrid_hybrid_distri_m40_s1.py` (Math Action regret minimization, BIC weighting, Real Log Canonical Threshold (RLCT) in NLMS adaptation)

Mathematical Bridge
------------------
Both parents manipulate probability‑like quantities that can be expressed in the log‑domain.
Parent A produces a **Hybrid Recovery Score** Ψ = (α·S + (1‑α)·R̄)·(1‑β·H) where
* S – similarity between two morphology vectors,
* R̄ – mean recovery priority,
* H – normalized Shannon entropy of token frequencies.

Parent B needs a log‑likelihood to compute the Bayesian Information Criterion (BIC) and the
Real Log Canonical Threshold (RLCT). We treat ‑log Ψ as a surrogate negative log‑likelihood.
Thus:

* BIC = n·log(L) + k·log(n)  →  BIC(Ψ) = n·log(Ψ) + k·log(n)
* RLCT = (d/2)·log(n) – log(L) → RLCT(Ψ) = (d/2)·log(n) – log(Ψ)

where *n* is the number of observed actions, *k* the number of model parameters
(and *d* the dimensionality of the NLMS filter). This creates a single scalar
that can weight regret updates and adapt the NLMS step size, fully fusing the two
topologies into one unified decision system.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Data structures (merged from both parents)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


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
# Core functions from Parent A
# ----------------------------------------------------------------------


def sphericity_index(length: float, width: float, height: float) -> float:
    """Return the sphericity index of a rectangular prism."""
    vol = length * width * height
    surface = 2 * (length * width + width * height + height * length)
    return (math.pi ** (1 / 3)) * (vol ** (1 / 3)) / ((surface / 6) ** (1 / 2))


def similarity_score(morph_a: Morphology, morph_b: Morphology) -> float:
    """
    Compute an SSIM‑like similarity between two morphology vectors.
    The score is the cosine similarity of the raw feature vectors
    multiplied by the ratio of their sphericity indices.
    """
    vec_a = np.array([morph_a.length, morph_a.width, morph_a.height, morph_a.mass])
    vec_b = np.array([morph_b.length, morph_b.width, morph_b.height, morph_b.mass])
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        cos_sim = 0.0
    else:
        cos_sim = dot / (norm_a * norm_b)

    sph_a = sphericity_index(morph_a.length, morph_a.width, morph_a.height)
    sph_b = sphericity_index(morph_b.length, morph_b.width, morph_b.height)
    sph_ratio = min(sph_a, sph_b) / max(sph_a, sph_b) if max(sph_a, sph_b) > 0 else 0.0

    return cos_sim * sph_ratio


def token_entropy(tokens: List[str]) -> float:
    """
    Compute normalized Shannon entropy of a token list.
    Tokens are expected to be pre‑tokenized words.
    """
    if not tokens:
        return 0.0
    total = len(tokens)
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    probs = np.array(list(freq.values())) / total
    ent = -np.sum(probs * np.log2(probs))
    max_ent = math.log2(len(freq)) if len(freq) > 1 else 0.0
    return ent / max_ent if max_ent > 0 else 0.0


def hybrid_recovery_score(
    morph_a: Morphology,
    morph_b: Morphology,
    tokens: List[str],
    R1: float,
    R2: float,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> float:
    """
    Parent‑A style hybrid score Ψ.
    """
    S = similarity_score(morph_a, morph_b)
    H = token_entropy(tokens)
    R_bar = (R1 + R2) / 2.0
    psi = (alpha * S + (1 - alpha) * R_bar) * (1 - beta * H)
    # Clamp to [0,1] for stability
    return max(0.0, min(1.0, psi))


# ----------------------------------------------------------------------
# Core functions from Parent B (Regret + RLCT)
# ----------------------------------------------------------------------


def bic_weight(psi: float, n: int, k: int) -> float:
    """
    Compute a BIC‑derived weight from the hybrid recovery score Ψ.
    We treat -log(Ψ) as a surrogate negative log‑likelihood.
    """
    if psi <= 0.0:
        psi = 1e-12
    log_likelihood = -math.log(psi)  # negative log‑likelihood surrogate
    bic = n * log_likelihood + k * math.log(max(n, 1))
    # Convert to a positive weight (smaller BIC → larger weight)
    return math.exp(-bic / n)


def regret_vector(actions: List[MathAction], counterfactuals: List[MathCounterfactual]) -> np.ndarray:
    """
    Compute the regret for each action based on counterfactual outcomes.
    Regret_i = max_j(EV_j) - EV_i
    """
    if not actions:
        return np.array([])
    # Map counterfactuals to actions
    cf_map: Dict[str, List[MathCounterfactual]] = {}
    for cf in counterfactuals:
        cf_map.setdefault(cf.action_id, []).append(cf)

    expected_vals = []
    for act in actions:
        cfs = cf_map.get(act.id, [])
        if cfs:
            # Weighted average of counterfactual outcomes
            weighted = sum(cf.outcome_value * cf.probability for cf in cfs)
            norm = sum(cf.probability for cf in cfs)
            exp_val = weighted / norm if norm > 0 else act.expected_value
        else:
            exp_val = act.expected_value
        expected_vals.append(exp_val)

    ev_arr = np.array(expected_vals)
    best = np.max(ev_arr)
    regrets = best - ev_arr
    return regrets


def weighted_regret(actions: List[MathAction], counterfactuals: List[MathCounterfactual], psi: float) -> np.ndarray:
    """
    Apply BIC‑derived weight to the raw regret vector.
    """
    n = len(actions)
    k = 4  # number of parameters we assume for the simple model
    weight = bic_weight(psi, n, k)
    raw_regret = regret_vector(actions, counterfactuals)
    return weight * raw_regret


def rlct_factor(psi: float, n: int, d: int) -> float:
    """
    Compute a scalar factor derived from the Real Log Canonical Threshold (RLCT).
    RLCT = (d/2)·log(n) – log(L)   with L ≈ Ψ
    The factor is used to scale the NLMS learning rate.
    """
    if psi <= 0.0:
        psi = 1e-12
    rlct = (d / 2.0) * math.log(max(n, 1)) - math.log(psi)
    # Map RLCT to (0, 1] by a simple exponential decay; larger RLCT → smaller step
    return math.exp(-rlct)


def nlms_update(
    coeffs: np.ndarray,
    x: np.ndarray,
    d: float,
    mu: float,
    psi: float,
    n: int,
) -> np.ndarray:
    """
    Normalized Least‑Mean‑Squares (NLMS) coefficient update.
    The adaptation step size μ is modulated by the RLCT factor derived from Ψ.
    """
    eps = 1e-8
    x_norm_sq = np.dot(x, x) + eps
    y = np.dot(coeffs, x)
    e = d - y
    rlct = rlct_factor(psi, n, d=coeffs.size)
    step = (mu * rlct) / x_norm_sq
    return coeffs + step * e * x


# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------


def hybrid_decision_pipeline(
    morph_a: Morphology,
    morph_b: Morphology,
    tokens: List[str],
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    R1: float,
    R2: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    End‑to‑end hybrid pipeline:
    1. Compute Ψ from morphology, tokens and recovery priorities.
    2. Compute BIC‑weighted regret for the supplied actions.
    3. Perform one NLMS adaptation step using the RLCT‑scaled learning rate.
    Returns (Ψ, weighted_regret_vector, updated_coeffs).
    """
    psi = hybrid_recovery_score(morph_a, morph_b, tokens, R1, R2)

    # Regret
    w_regret = weighted_regret(actions, counterfactuals, psi)

    # NLMS (simple synthetic example)
    dim = 4
    coeffs = np.random.randn(dim)
    x = np.random.randn(dim)
    desired = np.dot(coeffs, x) + random.gauss(0, 0.1)  # noisy observation
    updated = nlms_update(coeffs, x, desired, mu=0.5, psi=psi, n=len(actions))

    return psi, w_regret, updated


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Morphology instances
    m1 = Morphology(length=2.0, width=1.0, height=1.5, mass=3.0)
    m2 = Morphology(length=2.1, width=0.9, height=1.6, mass=2.9)

    # Token list (simulated log message)
    token_list = "error connecting to database retrying after timeout".split()

    # Math actions and counterfactuals
    acts = [
        MathAction(id="A", expected_value=0.8, cost=0.1, risk=0.05),
        MathAction(id="B", expected_value=0.6, cost=0.2, risk=0.1),
        MathAction(id="C", expected_value=0.9, cost=0.05, risk=0.02),
    ]

    cfs = [
        MathCounterfactual(action_id="A", outcome_value=0.85, probability=0.7),
        MathCounterfactual(action_id="B", outcome_value=0.55, probability=0.6),
        MathCounterfactual(action_id="C", outcome_value=0.95, probability=0.9),
    ]

    # Recovery priorities
    R1, R2 = 0.75, 0.65

    psi_val, regret_vec, coeffs_updated = hybrid_decision_pipeline(
        m1, m2, token_list, acts, cfs, R1, R2
    )

    print(f"Hybrid Recovery Score Ψ = {psi_val:.4f}")
    print(f"Weighted Regret Vector = {regret_vec}")
    print(f"Updated NLMS Coefficients = {coeffs_updated}")