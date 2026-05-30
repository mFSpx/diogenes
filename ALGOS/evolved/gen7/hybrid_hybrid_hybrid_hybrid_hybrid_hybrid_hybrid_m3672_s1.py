# DARWIN HAMMER — match 3672, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1573_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m2551_s1.py (gen5)
# born: 2026-05-29T23:51:15Z

"""Hybrid Bandit‑RBF‑Entropy‑Schoolfield Module
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1573_s0.py (entropy‑weighted bandit with pheromone tracking)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m2551_s1.py (RBF similarity matrix modulating bandit propensities and a temperature‑dependent Schoolfield rate)

Mathematical bridge:
Both parents expose a *propensity* vector **p** over actions.  
Parent B multiplies **p** by an RBF similarity matrix **S** (ϕ(i,j)=exp(−ε²‖v_i−v_j‖²)) and scales the result with the temperature‑dependent developmental rate ρ(T) from the Schoolfield equation.  
Parent A supplies a Shannon entropy **H** computed over a decision‑hygiene score distribution; this entropy quantifies uncertainty and is used as a multiplicative weight on the bandit’s confidence bound, i.e.  

    CB_i ← CB_i · (1 + H)

The fused system therefore computes

    adjusted_propensity = ρ(T) · S · p
    adjusted_confidence = CB · (1 + H)

which are then fed into the usual Upper‑Confidence‑Bound (UCB) selection rule.  The following implementation realises this coupling while keeping the original data‑structures of both parents."""


import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Shared data structures (identical in both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0          # cal·mol⁻¹
    t_low: float = 283.15                         # K
    t_high: float = 307.15                        # K
    delta_h_low: float = -45_000.0                # cal·mol⁻¹
    delta_h_high: float = 65_000.0                # cal·mol⁻¹
    r_cal: float = 1.987                          # cal·mol⁻¹·K⁻¹


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------
def schoolfield_rate(T: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Temperature‑dependent developmental rate ρ(T) using the full Schoolfield model.

    ρ(T) = ρ25 * exp( -ΔH_a / R * (1/T - 1/Tref) ) /
           ( 1 + exp( ΔH_low / R * (1/T_low - 1/T) ) +
                 exp( ΔH_high / R * (1/T - 1/T_high) ) )
    """
    T_ref = 298.15  # reference temperature (25 °C) in Kelvin
    exp_activation = math.exp(
        -params.delta_h_activation / params.r_cal * (1.0 / T - 1.0 / T_ref)
    )
    exp_low = math.exp(
        params.delta_h_low / params.r_cal * (1.0 / params.t_low - 1.0 / T)
    )
    exp_high = math.exp(
        params.delta_h_high / params.r_cal * (1.0 / T - 1.0 / params.t_high)
    )
    denominator = 1.0 + exp_low + exp_high
    return params.rho_25 * exp_activation / denominator


def rbf_similarity_matrix(features: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """Return the symmetric RBF similarity matrix ϕ(i,j)=exp(−ε²‖v_i−v_j‖²).

    `features` is an (n_actions, n_features) array.
    """
    if features.ndim != 2:
        raise ValueError("features must be a 2‑D array")
    # Squared Euclidean distances via broadcasting
    diff = features[:, None, :] - features[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    return np.exp(-epsilon ** 2 * d2)


def shannon_entropy(prob_dist: List[float]) -> float:
    """Compute Shannon entropy H = - Σ p_i log₂ p_i for a probability distribution."""
    eps = np.finfo(float).eps
    probs = np.array(prob_dist, dtype=float)
    probs = probs / (probs.sum() + eps)  # normalise safely
    nonzero = probs > 0
    return -np.sum(probs[nonzero] * np.log2(probs[nonzero] + eps))


# ----------------------------------------------------------------------
# Hybrid bandit operations
# ----------------------------------------------------------------------
def hybrid_propensity_vector(actions: List[BanditAction],
                             features: np.ndarray,
                             temperature: float,
                             epsilon: float = 1.0) -> np.ndarray:
    """Compute the temperature‑scaled, RBF‑smoothed propensity vector.

    Steps:
    1. Build similarity matrix S from `features`.
    2. Extract raw propensity vector p.
    3. Multiply: ρ(T) * S * p.
    """
    S = rbf_similarity_matrix(features, epsilon)          # (n, n)
    p = np.array([a.propensity for a in actions])        # (n,)
    rho_T = schoolfield_rate(temperature)                # scalar
    return rho_T * S.dot(p)


def hybrid_confidence_bounds(actions: List[BanditAction],
                             hygiene_scores: List[float]) -> np.ndarray:
    """Weight each action's confidence bound by (1 + H), where H is the entropy
    of the hygiene score distribution."""
    H = shannon_entropy(hygiene_scores)
    cb = np.array([a.confidence_bound for a in actions])
    return cb * (1.0 + H)


def select_action_ucb(actions: List[BanditAction],
                      features: np.ndarray,
                      temperature: float,
                      hygiene_scores: List[float],
                      epsilon: float = 1.0) -> BanditAction:
    """Upper‑Confidence‑Bound (UCB) selection using hybrid propensities and
    entropy‑adjusted confidence bounds.

    UCB_i = adjusted_propensity_i + adjusted_confidence_i
    """
    adj_prop = hybrid_propensity_vector(actions, features, temperature, epsilon)
    adj_cb = hybrid_confidence_bounds(actions, hygiene_scores)

    ucb_values = adj_prop + adj_cb
    best_idx = int(np.argmax(ucb_values))
    # Return a *new* BanditAction reflecting the hybrid values
    best = actions[best_idx]
    return BanditAction(
        action_id=best.action_id,
        propensity=adj_prop[best_idx],
        expected_reward=best.expected_reward,  # unchanged
        confidence_bound=adj_cb[best_idx],
        algorithm="hybrid_ucb"
    )


def apply_bandit_update(actions: List[BanditAction],
                        update: BanditUpdate,
                        learning_rate: float = 0.1) -> List[BanditAction]:
    """Simple stochastic gradient‑like update of propensity and confidence bound."""
    new_actions = []
    for a in actions:
        if a.action_id == update.action_id:
            new_prop = max(0.0, a.propensity + learning_rate * (update.reward - a.propensity))
            new_cb = max(0.0, a.confidence_bound * (1.0 - learning_rate) + learning_rate * abs(update.reward - a.propensity))
            new_actions.append(
                BanditAction(
                    action_id=a.action_id,
                    propensity=new_prop,
                    expected_reward=a.expected_reward,
                    confidence_bound=new_cb,
                    algorithm=a.algorithm,
                )
            )
        else:
            new_actions.append(a)
    return new_actions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic scenario with 3 actions
    actions = [
        BanditAction("a1", propensity=0.2, expected_reward=1.0, confidence_bound=0.5),
        BanditAction("a2", propensity=0.5, expected_reward=0.8, confidence_bound=0.3),
        BanditAction("a3", propensity=0.3, expected_reward=1.2, confidence_bound=0.4),
    ]

    # Random feature vectors (e.g., stylometric embeddings)
    rng = np.random.default_rng(42)
    features = rng.normal(size=(3, 5))

    # Simulated hygiene scores (e.g., decision‑hygiene metric)
    hygiene_scores = [0.7, 0.2, 0.1]  # already a distribution-like list

    temperature = 295.0  # Kelvin (~22 °C)

    # Select an action using the hybrid UCB rule
    chosen = select_action_ucb(
        actions,
        features,
        temperature,
        hygiene_scores,
        epsilon=0.8,
    )
    print(f"Chosen action: {chosen.action_id}")
    print(f"Hybrid propensity: {chosen.propensity:.4f}")
    print(f"Hybrid confidence bound: {chosen.confidence_bound:.4f}")

    # Simulate receiving a reward and update the bandit
    simulated_reward = rng.uniform(0, 2)  # arbitrary reward
    update = BanditUpdate(context_id="ctx1", action_id=chosen.action_id,
                         reward=simulated_reward, propensity=chosen.propensity)

    actions = apply_bandit_update(actions, update)
    print("\nPost‑update propensities:")
    for a in actions:
        print(f"  {a.action_id}: {a.propensity:.4f}, CB={a.confidence_bound:.4f}")

    sys.exit(0)