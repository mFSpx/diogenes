# DARWIN HAMMER — match 5731, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_fisher_locali_m711_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m1568_s0.py (gen6)
# born: 2026-05-30T00:04:24Z

"""Hybrid algorithm combining Fisher information (Parent A) with Gini-modulated bandit regret (Parent B).

Mathematical bridge:
- Compute Fisher scores for a set of θ values using the Gaussian beam (Parent A).
- Treat the collection of Fisher scores as a distribution of “values”; compute its Gini coefficient.
- Use the Gini coefficient to modulate the propensity of each BanditAction (Parent B) via
  `gini_modulated_propensity`.
- Apply a simple Koopman‐type linear operator to forecast the modulated propensities forward
  in time.
- Finally, feed the forecasted propensities into the regret‑weighted strategy from Parent B.

Thus the core topologies (Fisher information ↔ Gini coefficient ↔ Koopman operator ↔
regret weighting) are fused into a single pipeline.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Dict

import numpy as np

# ---------- Parent A components ----------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ---------- Parent B components ----------
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


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


def compute_gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("All values must be non-negative")
    n = len(xs)
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * xs)) / (n * np.sum(xs))
    return gini


def gini_modulated_propensity(action: BanditAction, values: Iterable[float]) -> float:
    """Scale the original propensity by (1 - Gini)."""
    gini = compute_gini_coefficient(values)
    return action.propensity * (1 - gini)


def compute_regret_weighted_strategy(actions: List[MathAction],
                                     counterfactuals: List[MathCounterfactual]) -> Dict[str, float]:
    """Softmax over regret‑adjusted expected values."""
    if not actions:
        return {}
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    vals = {a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions}
    best = max(vals.values())
    w = {k: math.exp(v - best) for k, v in vals.items()}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def koopman_forecast(propensity_dict: Dict[str, float],
                     steps: int = 1,
                     decay: float = 0.95) -> Dict[str, float]:
    """
    A minimal Koopman‑like operator: each step scales the vector by `decay`
    and adds a tiny uniform diffusion term to keep the distribution alive.
    """
    if steps <= 0:
        return propensity_dict.copy()
    vals = np.array(list(propensity_dict.values()), dtype=float)
    for _ in range(steps):
        vals = decay * vals + (1 - decay) / len(vals)
    return dict(zip(propensity_dict.keys(), vals.tolist()))


# ---------- Hybrid functions ----------
def compute_fisher_scores(thetas: Iterable[float],
                          center: float,
                          width: float) -> List[float]:
    """Vectorized Fisher scores for a collection of θ."""
    return [fisher_score(theta, center, width) for theta in thetas]


def modulate_bandit_actions(actions: List[BanditAction],
                            fisher_vals: Iterable[float]) -> List[BanditAction]:
    """
    Return new BanditAction objects whose propensities are scaled by the Gini
    coefficient derived from the Fisher score distribution.
    """
    modulated = []
    for act in actions:
        new_prop = gini_modulated_propensity(act, fisher_vals)
        modulated.append(BanditAction(
            action_id=act.action_id,
            propensity=new_prop,
            expected_reward=act.expected_reward,
            confidence_bound=act.confidence_bound,
            algorithm=act.algorithm
        ))
    return modulated


def hybrid_regret_strategy(actions: List[MathAction],
                           counterfactuals: List[MathCounterfactual],
                           bandit_actions: List[BanditAction],
                           thetas: Iterable[float],
                           center: float,
                           width: float,
                           koopman_steps: int = 2) -> Dict[str, float]:
    """
    Full hybrid pipeline:
    1. Compute Fisher scores over `thetas`.
    2. Modulate bandit propensities with the Gini of those scores.
    3. Forecast the modulated propensities via a Koopman operator.
    4. Combine the forecasted propensities with regret‑weighted probabilities.
    """
    # Step 1
    fisher_vals = compute_fisher_scores(thetas, center, width)

    # Step 2
    modulated_actions = modulate_bandit_actions(bandit_actions, fisher_vals)

    # Step 3 – build a propensity map and forecast
    prop_map = {a.action_id: a.propensity for a in modulated_actions}
    forecasted_prop = koopman_forecast(prop_map, steps=koopman_steps)

    # Step 4 – compute regret‑weighted strategy on MathAction space
    base_strategy = compute_regret_weighted_strategy(actions, counterfactuals)

    # Fuse: weight the base strategy by the forecasted propensities (aligned by id)
    fused = {}
    for aid, prob in base_strategy.items():
        fused_prob = prob * forecasted_prop.get(aid, 0.0)
        fused[aid] = fused_prob
    # renormalize
    total = sum(fused.values()) or 1.0
    return {k: v / total for k, v in fused.items()}


# ---------- Smoke test ----------
if __name__ == "__main__":
    # Dummy data
    thetas = np.linspace(-2.0, 2.0, 9)  # [-2, -1.5, ..., 2]
    center = 0.0
    width = 1.0

    actions = [
        MathAction(id="a1", expected_value=5.0, cost=1.0, risk=0.2),
        MathAction(id="a2", expected_value=4.5, cost=0.5, risk=0.1),
        MathAction(id="a3", expected_value=6.0, cost=1.5, risk=0.3),
    ]

    counterfactuals = [
        MathCounterfactual(action_id="a1", outcome_value=0.8, probability=0.9),
        MathCounterfactual(action_id="a2", outcome_value=0.5, probability=0.6),
    ]

    bandit_actions = [
        BanditAction(action_id="a1", propensity=0.33, expected_reward=4.0,
                     confidence_bound=0.1, algorithm="UCB"),
        BanditAction(action_id="a2", propensity=0.33, expected_reward=3.5,
                     confidence_bound=0.2, algorithm="UCB"),
        BanditAction(action_id="a3", propensity=0.34, expected_reward=5.0,
                     confidence_bound=0.15, algorithm="UCB"),
    ]

    result = hybrid_regret_strategy(
        actions=actions,
        counterfactuals=counterfactuals,
        bandit_actions=bandit_actions,
        thetas=thetas,
        center=center,
        width=width,
        koopman_steps=3,
    )
    print("Hybrid strategy probabilities:", result)