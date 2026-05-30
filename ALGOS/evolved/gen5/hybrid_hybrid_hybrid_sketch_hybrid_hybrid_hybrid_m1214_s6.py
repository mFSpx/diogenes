# DARWIN HAMMER — match 1214, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_sketches_rlct_hybrid_hybrid_bayes__m98_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_label__hybrid_hybrid_bandit_m415_s1.py (gen4)
# born: 2026-05-29T23:34:23Z

"""Hybrid Bayesian‑Bandit‑Sketch Algorithm
Integrates:
- Parent A (hybrid_hybrid_sketches_rlct_hybrid_hybrid_bayes__m98_s1): 
  Count‑Min sketch approximates the empirical log‑likelihood sum; 
  Ollivier‑Ricci curvature computed from a feature vector modulates the estimate.
- Parent B (hybrid_hybrid_hybrid_label__hybrid_hybrid_bandit_m415_s1):
  Contextual bandit selects a labeling function, updates propensity and expected reward,
  and uses a path‑signature‑like statistic to set a recovery priority.

Mathematical bridge:
The bandit’s expected reward **r** is defined as the curvature‑scaled sketch log‑likelihood  

 r = 𝒞(features) · L(sketch)  

where  

 L(sketch) = Σᵢ Σⱼ sketch[i][j]  

 𝒞(features) = Σ_k  f_k · log(f_k)   (Ollivier‑Ricci curvature approximation)

The bandit’s propensity is updated with an exponential‑weight rule using **r**, and the
path‑signature‑derived priority is the normalized product of feature values raised to the
curvature, i.e.  

 π = (∏_k f_k)^{𝒞(features)} / (1 + (∏_k f_k)^{𝒞(features)})

This unified system yields a probabilistic label whose confidence is proportional to
the bandit’s confidence bound, itself a function of the scaled reward and the number of
observations. """

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Standard Count‑Min sketch."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            index = hash(item) % width
            table[d][index] += 1
    return table


def ollivier_ricci_curvature(features: Dict[str, float]) -> float:
    """Approximate curvature: Σ f·log(f). Zero‑valued entries are ignored."""
    curv = 0.0
    for v in features.values():
        if v > 0.0:
            curv += v * math.log(v)
    return curv


def sketch_log_likelihood(sketch: List[List[int]]) -> float:
    """Sum of all counters – serves as a proxy for log‑likelihood."""
    return float(sum(sum(row) for row in sketch))


def hybrid_estimate(sketch: List[List[int]], features: Dict[str, float]) -> float:
    """Curvature‑scaled sketch estimate used as bandit reward."""
    L = sketch_log_likelihood(sketch)
    C = ollivier_ricci_curvature(features)
    return L * C


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int            # binary 0/1
    confidence: float    # in [0, 1]


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float


_POLICY: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0])  # [propensity, cum_reward, count]
_STORE: Dict[str, float] = {}  # placeholder virtual VRAM store


def _confidence_bound(cum_reward: float, count: int, alpha: float = 1.0) -> float:
    """UCB‑style bound."""
    if count == 0:
        return float('inf')
    return cum_reward / count + alpha * math.sqrt(math.log(1 + count) / count)


def select_bandit_action(action_id: str) -> BanditAction:
    """Selects an action using propensity‑weighted sampling and computes its confidence bound."""
    prop, cum_reward, cnt = _POLICY[action_id]
    # Ensure non‑zero propensity for sampling; fall back to uniform if needed
    if prop <= 0.0:
        prop = 1.0
    bound = _confidence_bound(cum_reward, cnt)
    return BanditAction(
        action_id=action_id,
        propensity=prop,
        expected_reward=cum_reward / (cnt if cnt > 0 else 1.0),
        confidence_bound=bound,
    )


def update_bandit(action: BanditAction, reward: float, eta: float = 0.1) -> None:
    """Exponential‑weight update of propensity and reward statistics."""
    prop, cum_reward, cnt = _POLICY[action.action_id]
    # Update propensity multiplicatively
    new_prop = prop * math.exp(eta * reward)
    _POLICY[action.action_id] = [
        new_prop,
        cum_reward + reward,
        cnt + 1,
    ]


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_labeling(
    doc_id: str,
    items: Iterable[str],
    features: Dict[str, float],
    action_id: str,
) -> ProbabilisticLabel:
    """Generates a probabilistic label for a document using the hybrid estimate as bandit reward."""
    sketch = count_min_sketch(items)
    reward = hybrid_estimate(sketch, features)

    # Bandit selection & update
    action = select_bandit_action(action_id)
    update_bandit(action, reward)

    # Decision rule: label 1 if expected reward exceeds median of observed rewards for this action
    _, cum_reward, cnt = _POLICY[action_id]
    median_est = cum_reward / (cnt if cnt > 0 else 1.0)
    label = 1 if reward >= median_est else 0

    # Confidence derived from bandit's confidence bound (scaled to [0,1])
    bound = _confidence_bound(cum_reward, cnt)
    confidence = min(1.0, bound / (bound + 1.0))

    return ProbabilisticLabel(doc_id=doc_id, label=label, confidence=confidence)


def hybrid_recovery_priority(sketch: List[List[int]], features: Dict[str, float]) -> float:
    """Path‑signature‑inspired priority: normalized product of features raised to curvature."""
    prod = 1.0
    for v in features.values():
        prod *= max(v, 1e-9)  # avoid zero
    curvature = ollivier_ricci_curvature(features)
    raw = prod ** curvature
    # Normalization to (0,1)
    priority = raw / (1.0 + raw)
    return priority


def hybrid_error_detection(
    label: ProbabilisticLabel,
    priority: float,
    error_threshold: float = 0.2,
) -> bool:
    """
    Returns True if the label is considered erroneous.
    The threshold is relaxed proportionally to the recovery priority.
    """
    adjusted_threshold = error_threshold * (1.0 - priority)
    return label.confidence < adjusted_threshold


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)

    # synthetic document
    doc_id = "doc-001"
    tokens = [f"word{i}" for i in range(100)]
    features = {
        "operator_visceral_ratio": 0.23,
        "operator_tech_ratio": 0.57,
        "operator_legal_osint_ratio": 0.31,
        "psyche_forensic_shield_ratio": 0.44,
        "psyche_poetic_entropy": 0.68,
        "psyche_dissociative_index": 0.12,
        "resilience_bureaucratic_weaponization_index": 0.05,
        "resilience_resource_exhaustion_metric": 0.79,
        "resilience_swarm_orchestration_density": 0.33,
        "rainmaker_corporate_grit_tension": 0.91,
        "rainmaker_countdown_density": 0.27,
        "rainmaker_asset_structuring_weight": 0.54,
        "telemetry_agent_symmetry_ratio": 0.62,
        "telemetry_protocol_discipline": 0.48,
        "telemetry_manic_velocity": 0.15,
    }

    # initialise a single action in the policy
    _POLICY["label_func_A"] = [1.0, 0.0, 0]

    # run hybrid labeling
    plabel = hybrid_labeling(doc_id, tokens, features, "label_func_A")
    print(f"ProbabilisticLabel: {plabel}")

    # compute recovery priority
    sketch = count_min_sketch(tokens)
    priority = hybrid_recovery_priority(sketch, features)
    print(f"Recovery priority: {priority:.4f}")

    # error detection
    is_error = hybrid_error_detection(plabel, priority)
    print(f"Label error detected? {is_error}")