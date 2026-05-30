# DARWIN HAMMER — match 3189, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1226_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1869_s0.py (gen5)
# born: 2026-05-29T23:48:18Z

"""Hybrid Regret‑Gini‑Confidence Algorithm
================================================

This module fuses the core topologies of the two parent algorithms:

* **Parent A** – *Hybrid Regret‑Bayes* (``hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1226_s4.py``)
* **Parent B** – *Hybrid Regret‑Bandit‑Koopman‑XGBoost with Distributed Leader Election*
  (``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1869_s0.py``)

**Mathematical bridge**  
Both parents generate a *regret‑weighted probability distribution* over a set of
actions.  Parent A measures the inequality of that distribution with the **Gini
coefficient**, while Parent B refines the distribution using **confidence
intervals** (Hoeffding bound) and a **broadcast probability** that drives a
distributed leader‑election style exploration/exploitation trade‑off.

The hybrid algorithm therefore proceeds as:

1. Compute regret‑weighted scores for each action (A‑style).
2. Convert scores to a soft‑max probability vector.
3. Evaluate the Gini coefficient of the vector – a scalar that quantifies
   dispersion.
4. Derive a confidence bound for each action using Hoeffding’s inequality
   (B‑style) and attenuate the probabilities according to a broadcast
   probability that depends on the Gini‑derived “phase”.
5. Renormalise to obtain the final hybrid distribution.

The resulting distribution simultaneously respects the Bayesian‑regret view,
the inequality‑sensitive adjustment from the Gini metric, and the statistically
grounded exploration control from confidence bounds.  This unified system can
drive both decision‑making (bandit‑style) and leader‑election‑style coordination
in a single mathematical framework.
"""

from __future__ import annotations
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from collections.abc import Iterable
import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared between parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action definition used by both parents."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit selection enriched with confidence information."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretGiniConfidence"


# ----------------------------------------------------------------------
# Parent‑A building block – Gini coefficient
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Return the Gini coefficient of a non‑negative value list."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


# ----------------------------------------------------------------------
# Parent‑B building block – Hoeffding confidence bound
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """
    Hoeffding bound for bounded random variable in [0, r].

    Parameters
    ----------
    r : float
        Upper bound of the reward (range width).
    delta : float
        Desired failure probability (e.g., 0.05 for 95% confidence).
    n : int
        Number of independent observations.

    Returns
    -------
    float
        Half‑width of the confidence interval.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return r * math.sqrt(math.log(2.0 / delta) / (2.0 * n))


# ----------------------------------------------------------------------
# Parent‑B building block – broadcast probability (leader election)
# ----------------------------------------------------------------------
def broadcast_probability(phase: int, step: int) -> float:
    """
    Probability of accepting a leader in a distributed election.

    The function decays exponentially with the difference between phase and step.
    """
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def compute_regret_weights(
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual],
) -> dict[str, float]:
    """
    Compute a raw regret‑based score for each action.

    Score_i = expected_value_i - Σ_j (outcome_value_j * probability_j)
    where the sum runs over counterfactuals that refer to action i.
    """
    # Aggregate counterfactual impact per action
    cf_impact: dict[str, float] = {}
    for cf in counterfactuals:
        cf_impact.setdefault(cf.action_id, 0.0)
        cf_impact[cf.action_id] += cf.outcome_value * cf.probability

    scores: dict[str, float] = {}
    for a in actions:
        regret = a.expected_value - cf_impact.get(a.id, 0.0)
        scores[a.id] = regret
    return scores


def softmax_distribution(scores: dict[str, float]) -> dict[str, float]:
    """
    Convert arbitrary scores to a probability distribution via the soft‑max
    transformation (numerically stable).
    """
    ids = list(scores.keys())
    raw = np.array([scores[i] for i in ids], dtype=float)
    # Stabilise by subtracting max
    max_raw = np.max(raw)
    exp_vals = np.exp(raw - max_raw)
    probs = exp_vals / exp_vals.sum()
    return {i: float(p) for i, p in zip(ids, probs)}


def hybrid_regret_gini_confidence(
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual],
    delta: float = 0.05,
    n_observations: int = 30,
    reward_range: float = 1.0,
) -> dict[str, float]:
    """
    Produce a hybrid probability distribution over actions.

    Steps
    -----
    1. Regret‑weighted scores → soft‑max probabilities.
    2. Compute Gini coefficient of the soft‑max vector (dispersion measure).
    3. Map Gini ∈ [0,1] to a discrete “phase” (1‑3) for leader‑election logic.
    4. For each action, compute a Hoeffding confidence bound and attenuate its
       probability by a broadcast probability that depends on (phase, step),
       where *step* is the integer rank of the action after sorting by probability.
    5. Renormalise the attenuated probabilities.
    """
    if not actions:
        return {}

    # 1. Regret scores → soft‑max
    raw_scores = compute_regret_weights(actions, counterfactuals)
    probs = softmax_distribution(raw_scores)

    # 2. Gini coefficient
    gini = gini_coefficient(list(probs.values()))

    # 3. Phase mapping (simple heuristic)
    #    low Gini (uniform) → phase 1 (exploit), high Gini (skewed) → phase 3 (explore)
    if gini < 0.33:
        phase = 1
    elif gini < 0.66:
        phase = 2
    else:
        phase = 3

    # 4. Attenuate with confidence & broadcast
    #    Sort actions by descending probability to assign step numbers.
    sorted_ids = sorted(probs, key=lambda i: probs[i], reverse=True)
    attenuated: dict[str, float] = {}
    for step, aid in enumerate(sorted_ids, start=1):
        base = probs[aid]
        # Hoeffding bound for this action's reward estimate
        cb = hoeffding_bound(reward_range, delta, n_observations)
        # Broadcast probability reduces exploration as phase grows
        bp = broadcast_probability(phase, step)
        # Combine: we lower the probability by the product of (1‑confidence) and (1‑bp)
        adjustment = (1.0 - cb) * bp
        attenuated[aid] = max(0.0, base * adjustment)

    # 5. Renormalise
    total = sum(attenuated.values())
    if total == 0.0:
        # fallback to uniform if everything vanished
        uniform = 1.0 / len(actions)
        return {a.id: uniform for a in actions}
    return {aid: val / total for aid, val in attenuated.items()}


def select_action_hybrid(
    actions: list[MathAction],
    distribution: dict[str, float],
) -> BanditAction:
    """
    Sample an action according to the hybrid distribution and attach confidence
    information.
    """
    ids = list(distribution.keys())
    probs = np.array([distribution[i] for i in ids], dtype=float)
    chosen_id = random.choices(ids, weights=probs, k=1)[0]
    # Expected reward is the original expected value of the chosen action
    expected_reward = next(a.expected_value for a in actions if a.id == chosen_id)
    # Confidence bound using default parameters (can be tuned externally)
    cb = hoeffding_bound(r=1.0, delta=0.05, n=30)
    return BanditAction(
        action_id=chosen_id,
        propensity=distribution[chosen_id],
        expected_reward=expected_reward,
        confidence_bound=cb,
    )


def hybrid_step(
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual],
) -> tuple[dict[str, float], BanditAction]:
    """
    Execute a full hybrid step:
    * compute the hybrid distribution,
    * draw an action,
    * return both for downstream processing.
    """
    dist = hybrid_regret_gini_confidence(actions, counterfactuals)
    chosen = select_action_hybrid(actions, dist)
    return dist, chosen


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny scenario
    actions = [
        MathAction(id="A", expected_value=0.8, cost=0.1),
        MathAction(id="B", expected_value=0.6, cost=0.05),
        MathAction(id="C", expected_value=0.4, cost=0.2),
    ]

    counterfactuals = [
        MathCounterfactual(action_id="A", outcome_value=0.2, probability=0.9),
        MathCounterfactual(action_id="B", outcome_value=0.1, probability=0.7),
        MathCounterfactual(action_id="C", outcome_value=0.3, probability=0.5),
    ]

    distribution, chosen = hybrid_step(actions, counterfactuals)

    print("Hybrid probability distribution:")
    for aid, p in distribution.items():
        print(f"  {aid}: {p:.4f}")

    print("\nChosen action:")
    print(f"  ID: {chosen.action_id}")
    print(f"  Propensity: {chosen.propensity:.4f}")
    print(f"  Expected reward: {chosen.expected_reward:.4f}")
    print(f"  Confidence bound: {chosen.confidence_bound:.4f}")