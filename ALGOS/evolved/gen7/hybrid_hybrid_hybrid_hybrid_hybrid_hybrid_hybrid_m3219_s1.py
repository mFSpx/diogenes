# DARWIN HAMMER — match 3219, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1496_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_ssim_m1265_s0.py (gen6)
# born: 2026-05-29T23:48:44Z

"""
Hybrid Algorithm: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1496_s0 + hybrid_hybrid_hybrid_hybrid_ssim_m1265_s0
Mathematical Bridge
-------------------
The fusion rests on a common *bandit decision* backbone present in both parents.
Parent A supplies a regret‑weighted probability distribution and a differential‑privacy
risk score, while Parent B contributes a geometry‑driven similarity measure
(SSIM) that modulates a sphericity index.  

We construct a **privacy‑aware structural utility**:

    U(structure, reference) = sphericity(structure) × SSIM(features(structure), features(reference))

and blend it with the **privacy‑penalized expected reward**:

    R̂(action) = expected_reward(action) × (1 − reconstruction_risk)

The final hybrid score for an (action, morphology) pair is

    S = λ·R̂(action) + (1‑λ)·U(structure, reference)

where λ∈[0,1] balances reward vs structural similarity.  
Selection uses the bandit’s propensity distribution, while updates employ
the DP‑aggregated regret from Parent A. This mathematically intertwines the
core topologies of both ancestors into a single unified decision system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Sequence, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Shared Data Structures (merged from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Simple reservoir dynamics used in Parent B."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._store_last_delta(delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """A bounded transformation of the last delta."""
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))

    def _store_last_delta(self, delta: float) -> None:
        self._last_delta = delta


# ----------------------------------------------------------------------
# Global policy stores (Parent A)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}
DEFAULT_BUDGET_MB = 1024 * 4


def reset_policy() -> None:
    """Clear all stored statistics."""
    _POLICY.clear()
    _STORE.clear()


def _reward(action_id: str) -> float:
    """Average reward observed for an action."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


# ----------------------------------------------------------------------
# Core mathematical building blocks
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Probability that a record can be re‑identified (Parent A)."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_aggregate(values: List[float], epsilon: float = 1.0) -> float:
    """Differentially private mean of `values` (Parent A)."""
    if not values:
        return 0.0
    sensitivity = 1.0
    laplace_noise = np.random.laplace(0.0, sensitivity / epsilon)
    return sum(values) / len(values) + laplace_noise


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity (Parent B)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def ssim(x: Sequence[float],
         y: Sequence[float],
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index (Parent B)."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mx = x_arr.mean()
    my = y_arr.mean()
    var_x = x_arr.var()
    var_y = y_arr.var()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx ** 2 + my ** 2 + c1) * (var_x + var_y + c2)

    return numerator / denominator if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Hybrid functional layer
# ----------------------------------------------------------------------
def compute_structural_utility(morph: Morphology,
                               reference: Morphology) -> float:
    """
    Combine sphericity and SSIM into a single utility value.
    """
    sph = sphericity_index(morph.length, morph.width, morph.height)
    vec_morph = [morph.length, morph.width, morph.height, morph.mass]
    vec_ref = [reference.length, reference.width, reference.height, reference.mass]
    similarity = ssim(vec_morph, vec_ref)
    return sph * similarity


def privacy_weighted_expected_reward(action: BanditAction,
                                     risk_score: float) -> float:
    """
    Attenuate the expected reward by the reconstruction risk.
    """
    return action.expected_reward * (1.0 - risk_score)


def hybrid_score(action: BanditAction,
                 morph: Morphology,
                 reference: Morphology,
                 risk_score: float,
                 lambda_reward: float = 0.6) -> float:
    """
    Final hybrid score mixing privacy‑aware reward and structural utility.
    """
    reward_part = privacy_weighted_expected_reward(action, risk_score)
    utility_part = compute_structural_utility(morph, reference)
    return lambda_reward * reward_part + (1.0 - lambda_reward) * utility_part


def select_hybrid_action(context: Dict[str, Any],
                         morphologies: List[Morphology],
                         actions: List[BanditAction],
                         reference: Morphology,
                         unique_quasi_identifiers: int,
                         total_records: int,
                         lambda_reward: float = 0.6) -> Tuple[BanditAction, Morphology]:
    """
    Choose the (action, morphology) pair with the highest hybrid score,
    respecting the bandit propensity distribution.
    """
    risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)

    # Compute raw scores for every cross‑product pair
    scores = []
    for morph in morphologies:
        for act in actions:
            sc = hybrid_score(act, morph, reference, risk, lambda_reward)
            # Incorporate propensity as a probabilistic bias
            weighted_sc = sc * act.propensity
            scores.append((weighted_sc, act, morph))

    if not scores:
        raise RuntimeError("No candidate actions or morphologies provided")

    # Select the pair with maximal weighted score
    _, best_action, best_morph = max(scores, key=lambda t: t[0])
    return best_action, best_morph


def record_update(update: BanditUpdate) -> None:
    """
    Record a bandit update using DP‑aggregated regret (Parent A).
    """
    key = update.action_id
    total, n = _POLICY.get(key, [0.0, 0.0])
    total += update.reward
    n += 1
    _POLICY[key] = [total, n]


def estimate_regret(action_id: str, epsilon: float = 1.0) -> float:
    """
    Estimate the regret for an action as the DP‑aggregated average reward.
    """
    rewards = [_reward(action_id)]  # In a real system this would be a list
    return dp_aggregate(rewards, epsilon)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny reference morphology
    reference = Morphology(length=10.0, width=5.0, height=2.0, mass=1.2)

    # Sample morphologies
    morphs = [
        Morphology(9.5, 5.2, 2.1, 1.1),
        Morphology(10.2, 4.8, 1.9, 1.3),
        Morphology(8.0, 5.0, 2.5, 1.0)
    ]

    # Sample actions
    actions = [
        BanditAction("A1", propensity=0.4, expected_reward=1.0, confidence_bound=0.2, algorithm="hybrid"),
        BanditAction("A2", propensity=0.35, expected_reward=0.8, confidence_bound=0.15, algorithm="hybrid"),
        BanditAction("A3", propensity=0.25, expected_reward=1.2, confidence_bound=0.25, algorithm="hybrid")
    ]

    # Context (unused in this minimal demo but kept for API compatibility)
    ctx = {"timestamp": 0}

    # Run selection
    chosen_action, chosen_morph = select_hybrid_action(
        context=ctx,
        morphologies=morphs,
        actions=actions,
        reference=reference,
        unique_quasi_identifiers=3,
        total_records=100,
        lambda_reward=0.6
    )

    print("Chosen Action:", asdict(chosen_action))
    print("Chosen Morphology:", asdict(chosen_morph))

    # Simulate receiving a reward and update the policy
    simulated_reward = random.uniform(0, 2)  # placeholder reward
    upd = BanditUpdate(
        context_id="demo",
        action_id=chosen_action.action_id,
        reward=simulated_reward,
        propensity=chosen_action.propensity,
    )
    record_update(upd)

    # Estimate regret for the chosen action
    regret_est = estimate_regret(chosen_action.action_id, epsilon=1.0)
    print(f"Estimated regret for action {chosen_action.action_id}: {regret_est:.4f}")

    # Demonstrate StoreState dynamics (Parent B component)
    store = StoreState()
    level, delta = store.update(inflow=[regret_est], outflow=[0.0])
    print(f"StoreState level after update: {level:.4f}, delta: {delta:.4f}, dance: {store.dance:.4f}")