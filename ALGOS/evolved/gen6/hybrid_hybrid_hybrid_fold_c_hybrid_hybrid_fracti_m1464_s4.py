# DARWIN HAMMER — match 1464, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s0.py (gen4)
# parent_b: hybrid_hybrid_fractional_hd_hybrid_hybrid_hybrid_m401_s0.py (gen5)
# born: 2026-05-29T23:36:35Z

"""Hybrid Fusion Module
Parents:
- hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s0.py (Bandit + pheromone infotaxis + entropy)
- hybrid_hybrid_fractional_hd_hybrid_hybrid_hybrid_m401_s0.py (Hypervector fractional power binding + Gini)

Mathematical Bridge:
The expected reward and log‑count‑ratio from the bandit side are used as
*scalars* that control the fractional power in the hyper‑vector binding.
The resulting bound hyper‑vectors are summed to produce a health‑score
vector whose component magnitudes are fed to a Gini coefficient.
Simultaneously the bandit’s pheromone infotaxis (pheromone × log‑count‑ratio)
yields a Shannon‑entropy term.  The final hybrid decision metric is a
convex combination of the entropy and the Gini‑based health score,
thereby fusing information‑theoretic and high‑dimensional binding
representations in a single unified system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Bandit / Pheromone Subsystem (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridBandit"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


_POLICY: Dict[str, Tuple[float, int]] = {}
_PHEROMONE: Dict[str, float] = defaultdict(float)


def reset_policy() -> None:
    """Reset the bandit policy and pheromone stores."""
    _POLICY.clear()
    _PHEROMONE.clear()


def _reward(action: str) -> float:
    total, n = _POLICY.get(action, (0.0, 0))
    return total / n if n else 0.0


def _count(action: str) -> int:
    return _POLICY.get(action, (0.0, 0))[1]


def update_policy(update: BanditUpdate) -> None:
    """Update policy statistics and pheromone level for an action."""
    total, n = _POLICY.get(update.action_id, (0.0, 0))
    _POLICY[update.action_id] = (total + update.reward, n + 1)
    # Simple pheromone accumulation proportional to received reward
    _PHEROMONE[update.action_id] += max(0.0, update.reward)


def _log_count_ratio(action: str, eps: float = 1e-9) -> float:
    """log(count / (total counts + eps)) for a given action."""
    cnt = _count(action)
    total = sum(_count(a) for a in _POLICY.keys())
    return math.log((cnt + eps) / (total + eps))


def _pheromone_infotaxis(action: str) -> float:
    """pheromone * log‑count‑ratio."""
    pher = _PHEROMONE.get(action, 0.0)
    lcr = _log_count_ratio(action)
    return pher * lcr


def _shannon_entropy(action: str) -> float:
    """Decision hygiene Shannon entropy term."""
    infot = _pheromone_infotaxis(action)
    if infot <= 0.0:
        return 0.0
    return -infot * math.log(infot)


# ----------------------------------------------------------------------
# Hypervector Subsystem (Parent B)
# ----------------------------------------------------------------------
def random_hv(d: int = 1024, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """Generate a random hypervector."""
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * math.pi, size=d)
        return np.exp(1j * theta)
    elif kind == "bipolar":
        return rng.choice([-1, 1], size=d)
    elif kind == "real":
        x = rng.normal(size=d)
        return x / np.linalg.norm(x)
    else:
        raise ValueError("Invalid kind")


def fractional_power_binding(hv1: np.ndarray, hv2: np.ndarray, power: float) -> np.ndarray:
    """Fractional power binding: hv1**power * conj(hv2)."""
    return np.power(hv1, power) * np.conjugate(hv2)


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a 1‑D array."""
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals, dtype=float)
    sum_vals = cumulative[-1]
    if sum_vals == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_vals) / n
    return gini


# ----------------------------------------------------------------------
# Fusion Core
# ----------------------------------------------------------------------
def compute_action_hypervector(action: BanditAction, dim: int = 1024) -> np.ndarray:
    """
    Create a hypervector for an action where the fractional power exponent
    is derived from the action's log‑count‑ratio.
    """
    # Base hypervectors (could be seeded by action_id for reproducibility)
    hv_a = random_hv(d=dim, kind="complex", seed=hash(action.action_id) & 0xffffffff)
    hv_b = random_hv(d=dim, kind="complex", seed=(hash(action.action_id) >> 1) & 0xffffffff)

    # Normalized log‑count‑ratio in [0, 1] to serve as power
    lcr = _log_count_ratio(action.action_id)
    # Scale lcr to a reasonable range for exponentiation
    power = 0.5 + 0.5 * math.tanh(lcr)  # maps to (0,1)
    bound_hv = fractional_power_binding(hv_a, hv_b, power)
    # Weight by expected reward magnitude
    return bound_hv * max(0.0, action.expected_reward)


def aggregate_health_vector(actions: List[BanditAction], dim: int = 1024) -> np.ndarray:
    """
    Sum bound hypervectors of all actions to obtain a health‑score vector.
    """
    agg = np.zeros(dim, dtype=complex)
    for act in actions:
        agg += compute_action_hypervector(act, dim)
    return agg


def fusion_decision_score(actions: List[BanditAction]) -> float:
    """
    Compute the unified decision metric:
        score = α * (average entropy) + (1‑α) * (1 - Gini(health magnitudes))

    where α balances information‑theoretic and high‑dimensional health
    perspectives.
    """
    if not actions:
        return 0.0

    # 1️⃣ Entropy component (average over actions)
    entropies = [_shannon_entropy(act.action_id) for act in actions]
    avg_entropy = sum(entropies) / len(entropies)

    # 2️⃣ Gini component from health vector magnitudes
    health_vec = aggregate_health_vector(actions)
    magnitudes = np.abs(health_vec)
    gini = gini_coefficient(magnitudes)

    # Blend
    alpha = 0.6  # favour entropy slightly
    score = alpha * avg_entropy + (1 - alpha) * (1 - gini)
    return score


def hybrid_select_action(actions: List[BanditAction]) -> BanditAction:
    """
    Select an action using a softmax over the fusion decision scores of each
    individual action (treated as if it were the sole member of the list).
    """
    # Compute a raw score per action
    raw_scores = []
    for act in actions:
        # Temporarily treat the list as containing only this action
        score = fusion_decision_score([act])
        raw_scores.append(score)

    # Stabilize softmax
    max_raw = max(raw_scores)
    exp_vals = [math.exp(s - max_raw) for s in raw_scores]
    total = sum(exp_vals)
    probs = [v / total for v in exp_vals]

    # Sample according to probabilities
    choice = random.choices(actions, weights=probs, k=1)[0]
    return choice


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a tiny policy
    reset_policy()
    actions = [
        BanditAction(action_id="A", propensity=0.2, expected_reward=1.5, confidence_bound=0.1),
        BanditAction(action_id="B", propensity=0.5, expected_reward=0.8, confidence_bound=0.2),
        BanditAction(action_id="C", propensity=0.3, expected_reward=2.0, confidence_bound=0.05),
    ]

    # Simulate a few updates to populate counts and pheromones
    for i in range(20):
        act = random.choice(actions)
        reward = random.uniform(0, 3)
        update_policy(BanditUpdate(context_id=f"c{i}", action_id=act.action_id,
                                   reward=reward, propensity=act.propensity))

    # Compute fused decision score for the whole set
    overall_score = fusion_decision_score(actions)
    print(f"Overall fusion decision score: {overall_score:.4f}")

    # Perform a hybrid selection
    chosen = hybrid_select_action(actions)
    print(f"Chosen action: {chosen.action_id}")

    # Verify that Gini and entropy are sensible numbers
    health_vec = aggregate_health_vector(actions)
    gini_val = gini_coefficient(np.abs(health_vec))
    ent_vals = [_shannon_entropy(a.action_id) for a in actions]
    print(f"Gini of health magnitudes: {gini_val:.4f}")
    print(f"Entropies per action: {[round(e,4) for e in ent_vals]}")