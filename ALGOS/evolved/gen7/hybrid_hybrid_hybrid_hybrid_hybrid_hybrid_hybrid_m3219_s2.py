# DARWIN HAMMER — match 3219, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1496_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_ssim_m1265_s0.py (gen6)
# born: 2026-05-29T23:48:44Z

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
    Record a bandit update using differential privacy.
    """
    _POLICY.setdefault(update.action_id, [0.0, 0.0])
    _POLICY[update.action_id][0] += update.reward
    _POLICY[update.action_id][1] += 1.0
    dp_aggregated_regret = dp_aggregate([update.reward], epsilon=1.0)
    _STORE[update.context_id] = dp_aggregated_regret


def calibrated_lambda_reward(unique_quasi_identifiers: int,
                             total_records: int,
                             default_lambda: float = 0.6) -> float:
    """
    Calibrate the lambda parameter based on the re-identification risk.
    """
    risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    return default_lambda * (1.0 - risk) + (1.0 - default_lambda) * risk


def improved_select_hybrid_action(context: Dict[str, Any],
                                 morphologies: List[Morphology],
                                 actions: List[BanditAction],
                                 reference: Morphology,
                                 unique_quasi_identifiers: int,
                                 total_records: int) -> Tuple[BanditAction, Morphology]:
    """
    Improved version of select_hybrid_action with calibrated lambda.
    """
    lambda_reward = calibrated_lambda_reward(unique_quasi_identifiers, total_records)
    return select_hybrid_action(context, morphologies, actions, reference, unique_quasi_identifiers, total_records, lambda_reward)