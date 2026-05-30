# DARWIN HAMMER — match 5256, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m2337_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2698_s1.py (gen5)
# born: 2026-05-30T00:01:01Z

import numpy as np
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Basic action used by the regret component."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome that can be added to the regret estimate."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class BanditAction:
    """Action description used by the bandit component."""
    action_id: str
    propensity: float               # prior probability of being chosen
    expected_reward: float
    confidence_bound: float         # e.g. UCB bonus
    algorithm: str                  # identifier of the underlying bandit algorithm


# ----------------------------------------------------------------------
# Core mathematical utilities
# ----------------------------------------------------------------------
def _softmax(values: Iterable[float], temperature: float = 1.0) -> List[float]:
    """Numerically stable soft‑max with optional temperature."""
    arr = np.array(list(values), dtype=float)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    arr = arr / temperature
    max_val = np.max(arr)
    exp_vals = np.exp(arr - max_val)
    total = np.sum(exp_vals)
    return (exp_vals / total).tolist()


def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    temperature: float = 0.5,
) -> Dict[str, float]:
    """
    Returns a probability distribution over actions derived from a
    regret‑adjusted value function.  The distribution is a soft‑max of
    ``expected_value - cost - risk + counterfactual``.
    """
    if not actions:
        return {}

    # map counterfactual contributions
    cf: Dict[str, float] = {
        c.action_id: c.outcome_value * c.probability for c in counterfactuals
    }

    # raw scores for each action
    scores = {
        a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions
    }

    # soft‑max conversion
    probs = _softmax(scores.values(), temperature)
    return dict(zip(scores.keys(), probs))


def shannon_entropy(probabilities: Iterable[float]) -> float:
    """Standard Shannon entropy (base‑2)."""
    probs = np.array([p for p in probabilities if p > 0], dtype=float)
    if probs.size == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def _entropy_regularizer(base_probs: List[float], entropy_factor: float) -> List[float]:
    """
    Modifies a probability vector by a factor proportional to its entropy.
    ``entropy_factor`` controls the strength of the regularisation (≥0).
    """
    ent = shannon_entropy(base_probs)
    # Scale the whole distribution; larger entropy → more uniform (higher entropy → larger factor)
    scale = 1.0 + entropy_factor * ent
    scaled = np.array(base_probs) * scale
    total = np.sum(scaled)
    return (scaled / total).tolist() if total > 0 else np.ones_like(scaled) / len(scaled)


def hybrid_select_action(
    bandit_actions: List[BanditAction],
    regret_distribution: Dict[str, float],
    rng: Optional[np.random.Generator] = None,
    entropy_factor: float = 0.2,
) -> str:
    """
    Chooses an action by blending:
      * regret‑derived probabilities,
      * the bandit propensity,
      * a confidence‑bound bonus,
    and finally applying an entropy‑based regulariser.
    """
    if not bandit_actions:
        raise ValueError("bandit_actions must contain at least one element")

    rng = rng or np.random.default_rng()

    # Gather raw components
    ids = [a.action_id for a in bandit_actions]
    regret_probs = np.array([regret_distribution.get(i, 0.0) for i in ids], dtype=float)
    propensity = np.array([a.propensity for a in bandit_actions], dtype=float)
    confidence = np.array([a.confidence_bound for a in bandit_actions], dtype=float)

    # Normalise each component safely
    def _safe_norm(v: np.ndarray) -> np.ndarray:
        s = v.sum()
        return v / s if s > 0 else np.ones_like(v) / len(v)

    regret_probs = _safe_norm(regret_probs)
    propensity = _safe_norm(propensity)
    # Confidence bounds are turned into a soft‑max to keep them in [0,1]
    confidence = np.array(_softmax(confidence, temperature=1.0))

    # Blend components (weights can be tuned)
    blended = 0.5 * regret_probs + 0.3 * propensity + 0.2 * confidence
    blended = _safe_norm(blended)

    # Entropy regularisation makes the distribution slightly more exploratory
    blended = np.array(_entropy_regularizer(blended.tolist(), entropy_factor))

    # Final safe normalisation
    blended = _safe_norm(blended)

    chosen = rng.choice(ids, p=blended)
    return chosen


def hybrid_rlct_estimate(
    bandit_actions: List[BanditAction],
    regret_distribution: Dict[str, float],
) -> float:
    """
    Expected reward estimate under the same blended distribution used for selection.
    """
    if not bandit_actions:
        return 0.0

    ids = [a.action_id for a in bandit_actions]
    regret_probs = np.array([regret_distribution.get(i, 0.0) for i in ids], dtype=float)

    # Normalise regret probabilities (fallback to uniform if degenerate)
    if regret_probs.sum() == 0:
        regret_probs = np.ones_like(regret_probs) / len(regret_probs)
    else:
        regret_probs = regret_probs / regret_probs.sum()

    expected = sum(
        a.expected_reward * p for a, p in zip(bandit_actions, regret_probs)
    )
    return expected


def path_signature(sequence: List[List[float]]) -> Tuple[float, np.ndarray]:
    """
    Computes a simple level‑1 and level‑2 signature for a piece‑wise linear path.
    * level‑1: total Euclidean length (as before)
    * level‑2: sum of outer products of successive increments (captures curvature)
    Returns a tuple ``(level1, level2_matrix)``.
    """
    if not sequence:
        return 0.0, np.zeros((0, 0))

    seq = np.asarray(sequence, dtype=float)
    increments = np.diff(seq, axis=0)                     # shape (n-1, d)
    level1 = float(np.linalg.norm(increments, axis=1).sum())

    # level‑2 signature: Σ Δx_i ⊗ Δx_i
    d = seq.shape[1]
    level2 = np.zeros((d, d), dtype=float)
    for inc in increments:
        level2 += np.outer(inc, inc)

    return level1, level2


def build_hybrid_sketch(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    temperature: float = 0.5,
) -> Tuple[Dict[str, float], List[List[float]]]:
    """
    Produces the regret distribution and the raw feature sequence that will
    later be fed to ``path_signature``.
    """
    regret_distribution = compute_regret_weighted_strategy(
        actions, counterfactuals, temperature=temperature
    )
    sequence = [
        [a.expected_value, a.cost, a.risk] for a in actions
    ]
    return regret_distribution, sequence


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic RNG for reproducibility in demos
    rng = np.random.default_rng(seed=42)

    actions = [
        MathAction("action1", expected_value=10.0, cost=1.0, risk=0.5),
        MathAction("action2", expected_value=20.0, cost=2.0, risk=1.0),
        MathAction("action3", expected_value=15.0, cost=0.5, risk=0.2),
    ]

    counterfactuals = [
        MathCounterfactual("action1", outcome_value=5.0, probability=0.8),
        MathCounterfactual("action2", outcome_value=10.0, probability=0.6),
        MathCounterfactual("action3", outcome_value=2.0, probability=0.9),
    ]

    regret_dist, seq = build_hybrid_sketch(actions, counterfactuals, temperature=0.7)
    print("Regret distribution:", regret_dist)

    lvl1, lvl2 = path_signature(seq)
    print("Path signature level‑1 (length):", lvl1)
    print("Path signature level‑2 (matrix):\n", lvl2)

    bandit_actions = [
        BanditAction("action1", propensity=0.4, expected_reward=9.0, confidence_bound=0.8, algorithm="UCB1"),
        BanditAction("action2", propensity=0.4, expected_reward=19.0, confidence_bound=0.5, algorithm="UCB1"),
        BanditAction("action3", propensity=0.2, expected_reward=14.0, confidence_bound=1.2, algorithm="UCB1"),
    ]

    chosen = hybrid_select_action(bandit_actions, regret_dist, rng=rng, entropy_factor=0.3)
    print("Chosen action:", chosen)

    est = hybrid_rlct_estimate(bandit_actions, regret_dist)
    print("Hybrid expected reward estimate:", est)