# DARWIN HAMMER — match 1213, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m959_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rlct_g_m436_s1.py (gen5)
# born: 2026-05-29T23:34:36Z

"""Hybrid Algorithm: hybrid_morphology_regret_rlct
Combines the core topologies of:

* Parent A – `hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s6.py`
  Provides morphology vectors, an SSIM‑like similarity `S` and an entropy
  measure `H` derived from token frequencies.

* Parent B – `hybrid_hybrid_hybrid_regret_hybrid_hard_truth_ma_m257_s1.py`
  Supplies a Counterfactual Regret Minimization (CFR) framework weighted by
  the Bayesian Information Criterion (BIC) and an adaptation step guided by
  the Real Log‑Canonical Threshold (RLCT) used in NLMS‑style updates.

**Mathematical Bridge**

The bridge is built on information‑theoretic quantities that appear in both
parents:

* Entropy `H` (Parent A) and BIC `B` (Parent B) are both logarithmic penalties
  on model complexity. We map `B` to a normalized weight `w_B ∈ [0,1]` and
  combine it with `H` to obtain a joint complexity factor `C = (1‑β·H)·w_B`.

* RLCT `λ` (Parent B) is a scalar that measures the singularity of the loss
  landscape. We use `λ` to scale the learning rate of the regret update,
  yielding an NLMS‑style step size `η = 1/(1+λ)`.

The unified **Hybrid Recovery Score** `Ψ` therefore becomes


Ψ = (α·S + (1‑α)·R̄) · C
where  R̄ = (R₁+R₂)/2  (average recovery priority)


The algorithm proceeds by:
1. Computing `S` from morphology vectors.
2. Computing `H` from token frequencies.
3. Computing `B` from a log‑likelihood, parameter count and sample size.
4. Forming `Ψ`.
5. Updating regret values with NLMS‑scaled CFR weighted by `w_B`.
6. Selecting the action with maximal expected utility adjusted by `Ψ`.

The implementation below respects the required import set and provides three
core functions plus a smoke‑test block.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology & Entropy utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Return a dimensionless sphericity index (0‑1)."""
    vol = length * width * height
    surface = math.sqrt(length ** 2 + width ** 2 + height ** 2)
    return (math.pi ** (1 / 3)) * (vol ** (1 / 3)) / surface


def similarity_score(morph_a: Morphology, morph_b: Morphology) -> float:
    """
    Compute an SSIM‑like similarity between two morphology vectors.
    The formula is a normalized dot product after scaling by sphericity.
    """
    vec_a = np.array(
        [
            morph_a.length,
            morph_a.width,
            morph_a.height,
            morph_a.mass,
            sphericity_index(morph_a.length, morph_a.width, morph_a.height),
        ]
    )
    vec_b = np.array(
        [
            morph_b.length,
            morph_b.width,
            morph_b.height,
            morph_b.mass,
            sphericity_index(morph_b.length, morph_b.width, morph_b.height),
        ]
    )
    # Normalize vectors
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    # Map cosine (-1,1) to (0,1)
    return (cosine + 1.0) / 2.0


def normalized_shannon_entropy(tokens: List[str]) -> float:
    """Return H ∈ [0,1] for a list of categorical tokens."""
    if not tokens:
        return 0.0
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    probs = np.array([c / total for c in counts.values()])
    # Shannon entropy
    H_raw = -np.sum(probs * np.log2(probs + 1e-12))
    # Normalise by log2(|alphabet|)
    H_max = math.log2(len(counts))
    return H_raw / H_max if H_max > 0 else 0.0


# ----------------------------------------------------------------------
# Parent B – Regret, BIC and RLCT utilities
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


def bic(log_likelihood: float, num_params: int, num_samples: int) -> float:
    """Bayesian Information Criterion."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    return -2.0 * log_likelihood + num_params * math.log(num_samples)


def normalized_bic_weight(bic_value: float, scale: float = 10.0) -> float:
    """
    Convert BIC to a weight in [0,1] using a sigmoid‑like mapping.
    Larger BIC (worse model) → smaller weight.
    """
    return 1.0 / (1.0 + math.exp(bic_value / scale))


def rlct_from_covariance(cov: np.ndarray) -> float:
    """
    Approximate the Real Log‑Canonical Threshold (RLCT) from a covariance matrix.
    For a positive‑definite matrix, λ = sum_i 1/(σ_i + ε) where σ_i are eigenvalues.
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square matrix")
    eigvals = np.linalg.eigvalsh(cov)
    eps = 1e-12
    return float(np.sum(1.0 / (eigvals + eps)))


def nlms_regret_update(
    regrets: Dict[str, float],
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    eta: float,
    bic_weight: float,
) -> Dict[str, float]:
    """
    Perform a single NLMS‑style CFR update.
    regrets are scaled by η and weighted by the BIC‑derived factor.
    """
    # Compute instantaneous regret for each action
    cf_dict = {cf.action_id: cf for cf in counterfactuals}
    new_regrets = {}
    for act in actions:
        cf = cf_dict.get(act.id, MathCounterfactual(act.id, act.expected_value))
        instant_regret = cf.outcome_value - act.expected_value
        # NLMS update: r_{new} = r_{old} + η * (instant_regret - r_{old})
        r_old = regrets.get(act.id, 0.0)
        r_new = r_old + eta * (instant_regret - r_old) * bic_weight
        new_regrets[act.id] = r_new
    return new_regrets


# ----------------------------------------------------------------------
# Hybrid core functions (bridge of both parents)
# ----------------------------------------------------------------------


def hybrid_recovery_score(
    morph_a: Morphology,
    morph_b: Morphology,
    tokens: List[str],
    r1: float,
    r2: float,
    alpha: float = 0.6,
    beta: float = 0.4,
    log_likelihood: float = -100.0,
    num_params: int = 10,
    num_samples: int = 500,
) -> float:
    """
    Compute Ψ = (α·S + (1‑α)·R̄) · (1‑β·H) · w_B
    where:
        S – similarity_score (Parent A)
        H – normalized_shannon_entropy (Parent A)
        R̄ – average recovery priority
        w_B – normalized BIC weight (Parent B)
    """
    S = similarity_score(morph_a, morph_b)
    H = normalized_shannon_entropy(tokens)
    R_bar = (r1 + r2) / 2.0
    base = alpha * S + (1.0 - alpha) * R_bar
    complexity = (1.0 - beta * H)
    B = bic(log_likelihood, num_params, num_samples)
    w_B = normalized_bic_weight(B)
    return base * complexity * w_B


def hybrid_regret_step(
    regrets: Dict[str, float],
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    cov_matrix: np.ndarray,
    log_likelihood: float,
    num_params: int,
    num_samples: int,
) -> Dict[str, float]:
    """
    Perform a regret update that uses:
        * η = 1/(1+λ) where λ = RLCT(cov_matrix) (Parent B)
        * bic_weight derived from the current model (Parent B)
    The resulting regrets are ready for action selection.
    """
    lam = rlct_from_covariance(cov_matrix)
    eta = 1.0 / (1.0 + lam)
    B = bic(log_likelihood, num_params, num_samples)
    w_B = normalized_bic_weight(B)
    return nlms_regret_update(regrets, actions, counterfactuals, eta, w_B)


def select_action(
    actions: List[MathAction],
    regrets: Dict[str, float],
    recovery_score: float,
) -> Tuple[MathAction, float]:
    """
    Choose the action maximizing a utility that blends expected value,
    regret magnitude and the hybrid recovery score Ψ.
    Utility = expected_value - cost + risk + Ψ·|regret|
    """
    best_util = -math.inf
    best_act = None
    for act in actions:
        regret = regrets.get(act.id, 0.0)
        util = (
            act.expected_value
            - act.cost
            + act.risk
            + recovery_score * abs(regret)
        )
        if util > best_util:
            best_util = util
            best_act = act
    if best_act is None:
        raise RuntimeError("No actions provided")
    return best_act, best_util


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Morphologies (Parent A)
    m1 = Morphology(2.0, 1.5, 1.0, 0.8)
    m2 = Morphology(2.1, 1.4, 1.05, 0.85)

    # Token list (simulated log tokens)
    token_list = ["error", "warning", "info", "error", "debug", "info", "warning"]

    # Recovery priorities
    R1, R2 = 0.7, 0.4

    # Compute hybrid recovery score Ψ
    psi = hybrid_recovery_score(
        m1,
        m2,
        token_list,
        R1,
        R2,
        alpha=0.5,
        beta=0.3,
        log_likelihood=-120.0,
        num_params=15,
        num_samples=800,
    )
    print(f"Hybrid Recovery Score Ψ = {psi:.4f}")

    # Define actions (Parent B)
    actions = [
        MathAction(id="A", expected_value=1.2, cost=0.1, risk=0.05),
        MathAction(id="B", expected_value=0.9, cost=0.05, risk=0.02),
        MathAction(id="C", expected_value=1.5, cost=0.2, risk=0.1),
    ]

    # Counterfactual outcomes (simulated)
    counterfactuals = [
        MathCounterfactual(action_id="A", outcome_value=1.3, probability=0.6),
        MathCounterfactual(action_id="B", outcome_value=0.8, probability=0.3),
        MathCounterfactual(action_id="C", outcome_value=1.6, probability=0.1),
    ]

    # Initial regrets (zero)
    regrets = {a.id: 0.0 for a in actions}

    # Covariance matrix for RLCT (simple positive‑definite)
    cov = np.array([[0.5, 0.1, 0.0], [0.1, 0.4, 0.05], [0.0, 0.05, 0.3]])

    # Update regrets using hybrid step
    regrets = hybrid_regret_step(
        regrets,
        actions,
        counterfactuals,
        cov,
        log_likelihood=-115.0,
        num_params=12,
        num_samples=750,
    )
    print("Updated regrets:", {k: round(v, 4) for k, v in regrets.items()})

    # Select best action
    best_action, utility = select_action(actions, regrets, psi)
    print(f"Selected action: {best_action.id} with utility {utility:.4f}")

    sys.exit(0)