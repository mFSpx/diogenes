# DARWIN HAMMER — match 4312, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_hybrid_hybrid_m1453_s2.py (gen6)
# parent_b: hybrid_bayes_update_hybrid_hybrid_hybrid_m1354_s2.py (gen6)
# born: 2026-05-29T23:54:48Z

"""Hybrid Fusion of VRAM‑Bandit Scheduler, Pheromone Decay, SSIM Decision & Bayesian Update.

Parents:
- hybrid_hybrid_model_vram_sc_hybrid_hybrid_m1453_s2.py (VRAM‑Bandit Scheduler + Pheromone + SSIM)
- hybrid_bayes_update_hybrid_hybrid_hybrid_m1354_s2.py (Bandit ↔ Bayesian update)

Mathematical bridge:
1. The BanditAction.propensity is interpreted as a likelihood L_i for action i.
2. A Bayesian update yields a posterior belief B_i' = prior_i * L_i / Z, where Z is the marginal.
3. The posterior B_i' is fed back as the new confidence_bound of the BanditAction,
   closing the loop between exploration‑exploitation and probabilistic learning.
4. Pheromone levels Φ_i decay exponentially (Φ_i←Φ_i·(1‑δ)) and are reinforced by the
   posterior B_i' (Φ_i←Φ_i + η·B_i'). This fuses the pheromone dynamics with the Bayesian
   learning.
5. An SSIM‑like structural similarity S_i between the current resource‑state vector
   v and a target prototype t is combined with the Bayesian‑pheromone term into a
   unified hybrid score:
        H_i = w₁·B_i' + w₂·Φ_i + w₃·S_i
   The action with maximal H_i is selected.

The module implements this unified pipeline through three core functions.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

import numpy as np

# ----------------------------------------------------------------------
# Data structures (merged from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # interpreted as inflow rate / likelihood
    expected_reward: float
    confidence_bound: float    # will be overwritten by Bayesian posterior
    algorithm: str


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["morphology"] = asdict(self.morphology)
        return d


# ----------------------------------------------------------------------
# Helper functions for the hybrid algorithm
# ----------------------------------------------------------------------
def decay_pheromones(pheromones: Dict[str, float], decay_rate: float) -> Dict[str, float]:
    """Exponential decay of pheromone levels.

    Φ_i ← Φ_i * (1 - decay_rate)
    """
    if not (0.0 <= decay_rate <= 1.0):
        raise ValueError("decay_rate must be in [0,1]")
    return {k: v * (1.0 - decay_rate) for k, v in pheromones.items()}


def bayesian_update_with_propensity(
    priors: Dict[str, float],
    actions: List[BanditAction],
    false_positive: float = 1e-6,
) -> Dict[str, float]:
    """Perform a Bayesian update where each action.propensity is the likelihood.

    Returns a dictionary of posterior probabilities indexed by action_id.
    """
    # collect likelihoods
    likelihoods = {a.action_id: max(0.0, min(1.0, a.propensity)) for a in actions}
    # compute marginal Z = Σ prior_i * likelihood_i + fp * Σ (1-prior_i)
    marginal = 0.0
    for aid, prior in priors.items():
        marginal += prior * likelihoods.get(aid, 0.0)
    # add false‑positive contribution for the complement mass
    marginal += false_positive * (1.0 - sum(priors.values()))
    if marginal <= 0.0:
        raise ValueError("Marginal probability non‑positive")

    posteriors: Dict[str, float] = {}
    for aid, prior in priors.items():
        lik = likelihoods.get(aid, 0.0)
        post = (prior * lik) / marginal
        posteriors[aid] = post
    # handle actions that were not present in priors (new actions)
    for aid, lik in likelihoods.items():
        if aid not in priors:
            post = (0.0 * lik) / marginal  # remains zero
            posteriors[aid] = post
    return posteriors


def ssim_placeholder(v: np.ndarray, t: np.ndarray) -> float:
    """A lightweight SSIM‑like similarity between two vectors.

    Returns a value in [0,1] where 1 means identical.
    """
    if v.shape != t.shape:
        raise ValueError("Vectors must have the same shape")
    mu_v, mu_t = v.mean(), t.mean()
    sigma_v, sigma_t = v.var(), t.var()
    cov = ((v - mu_v) * (t - mu_t)).mean()
    C1, C2 = 1e-4, 9e-4
    numerator = (2 * mu_v * mu_t + C1) * (2 * cov + C2)
    denominator = (mu_v ** 2 + mu_t ** 2 + C1) * (sigma_v + sigma_t + C2)
    return float(numerator / denominator) if denominator != 0 else 0.0


def hybrid_select_action(
    actions: List[BanditAction],
    pheromones: Dict[str, float],
    priors: Dict[str, float],
    resource_state: np.ndarray,
    target_prototype: np.ndarray,
    decay_rate: float = 0.05,
    eta: float = 0.1,
    w1: float = 0.5,
    w2: float = 0.3,
    w3: float = 0.2,
) -> BanditAction:
    """Unified hybrid decision making.

    Steps:
    1. Decay pheromones.
    2. Bayesian update using propensities as likelihoods.
    3. Reinforce pheromones with posterior (Φ_i ← Φ_i + η·B_i').
    4. Compute SSIM‑like similarity between current resource state and a prototype.
    5. Combine into hybrid score H_i and pick the best action.
    """
    # 1. decay
    pheromones = decay_pheromones(pheromones, decay_rate)

    # 2. Bayesian posterior
    posteriors = bayesian_update_with_propensity(priors, actions)

    # 3. reinforce pheromones
    for aid, post in posteriors.items():
        pheromones[aid] = pheromones.get(aid, 0.0) + eta * post

    # 4. SSIM‑like similarity (same for all actions, but could be per‑action)
    similarity = ssim_placeholder(resource_state, target_prototype)

    # 5. hybrid scoring
    best_score = -math.inf
    best_action = None
    for a in actions:
        post = posteriors.get(a.action_id, 0.0)
        phi = pheromones.get(a.action_id, 0.0)
        # hybrid metric
        score = w1 * post + w2 * phi + w3 * similarity
        if score > best_score:
            best_score = score
            best_action = a

    # Update the selected action's confidence bound with its posterior
    if best_action is not None:
        updated = BanditAction(
            action_id=best_action.action_id,
            propensity=best_action.propensity,
            expected_reward=best_action.expected_reward,
            confidence_bound=posteriors.get(best_action.action_id, 0.0),
            algorithm=best_action.algorithm,
        )
        return updated
    else:
        raise RuntimeError("No action selected – empty action list?")


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny set of actions
    actions = [
        BanditAction("A1", propensity=0.7, expected_reward=1.2, confidence_bound=0.0, algorithm="hybrid"),
        BanditAction("A2", propensity=0.4, expected_reward=0.8, confidence_bound=0.0, algorithm="hybrid"),
        BanditAction("A3", propensity=0.2, expected_reward=0.5, confidence_bound=0.0, algorithm="hybrid"),
    ]

    # Initial pheromone levels (could be learned from previous runs)
    pheromones = {"A1": 0.3, "A2": 0.5, "A3": 0.1}

    # Prior beliefs (simple uniform prior)
    priors = {"A1": 1/3, "A2": 1/3, "A3": 1/3}

    # Simulated resource state vector and a target prototype
    resource_state = np.array([0.6, 0.2, 0.9])
    target_prototype = np.array([0.5, 0.3, 0.8])

    # Run the hybrid selector
    selected = hybrid_select_action(
        actions=actions,
        pheromones=pheromones,
        priors=priors,
        resource_state=resource_state,
        target_prototype=target_prototype,
    )

    print("Selected action:")
    print(selected)
    print("Updated pheromones (post‑selection):")
    print(pheromones)