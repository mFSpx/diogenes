# DARWIN HAMMER — match 8, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_hoeffding_tre_m44_s3.py (gen3)
# parent_b: hybrid_regret_engine_hybrid_doomsday_cale_m19_s9.py (gen2)
# born: 2026-05-29T23:26:19Z

"""Hybrid Endpoint‑SSM, Tropical Max‑Plus, Regret & Gini Fusion
================================================================

This module merges the two parent algorithms:

* **Parent A** – ``hybrid_hybrid_endpoint_circ_state_space_duality_m1_s2.py``  
  Provides a linear state‑space model (SSM) built from *endpoints* that yields
  a scalar health score `y_t` per time step.  The health scores are then fed
  to a tropical (max‑plus) network producing *gain* candidates.

* **Parent B** – ``hybrid_regret_engine_hybrid_doomsday_cale_m19_s9.py``  
  Defines actions, regret‑weighted probabilities and the Gini coefficient as
  a measure of distributional fairness.

**Mathematical bridge**

1. The health score vector `h ∈ ℝ^n` (output of the SSM) is interpreted as the
   *expected value* of a set of actions.  Each action `a_j` carries an
   *intrinsic cost* `c_j`.  In the tropical max‑plus layer we compute

   
   g_j = max_i (W_{j,i} + h_i)      with   W_{j,i} = -c_j   (constant row)
   

   i.e. `g_j = max_i (h_i) - c_j`.  This yields a *regret‑adjusted gain*
   candidate for every action.

2. The sequence of gain candidates `g_j(t)` is streamed to a Hoeffding‑bound
   monitor (Parent A) to decide whether a decision‑tree node should be split.
   Simultaneously we evaluate the Gini coefficient of the normalized gain
   distribution (Parent B) to quantify inequality among actions.  The two
   statistics are combined: a split is permitted only if the Hoeffding bound
   certifies statistical significance **and** the Gini coefficient is below a
   user‑defined fairness threshold.

The three core functions below implement this pipeline:

* `compute_health_scores(endpoints)` – builds the SSM matrices from the
  endpoint pool and returns a health‑score vector.
* `tropical_regret_gains(health_scores, actions)` – evaluates the max‑plus
  network and returns a gain per action.
* `update_stats_and_maybe_split(gains, stats, delta, gini_thr)` – updates
  Hoeffding statistics, checks the bound, computes the Gini coefficient and
  decides on a split.

All operations rely only on ``numpy`` and the Python standard library."""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from both parents)
# ----------------------------------------------------------------------


@dataclass
class Endpoint:
    """Endpoint description used by the state‑space model."""
    failures: int
    failure_threshold: int
    righting_time_index: float  # higher ⇒ healthier

    @property
    def failure_rate(self) -> float:
        """Normalized failure rate in [0, 1]."""
        if self.failure_threshold == 0:
            return 0.0
        return min(1.0, self.failures / self.failure_threshold)


@dataclass(frozen=True, slots=True)
class MathAction:
    """Elementary decision element."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


# ----------------------------------------------------------------------
# Parent A – State‑space health scores
# ----------------------------------------------------------------------


def compute_health_scores(endpoints: List[Endpoint]) -> np.ndarray:
    """
    Build a simple linear state‑space model from *endpoints* and return the
    health‑score vector `h ∈ ℝ^m` where `m` is the number of endpoints.

    The SSM is defined as:
        x_{t+1} = A x_t + B u_t
        y_t     = C x_t + D u_t

    For this synthetic fusion we treat each endpoint as an independent
    scalar subsystem with:
        A = 1 - failure_rate
        B = righting_time_index
        u_t = 1   (unit excitation)
        C = 1
        D = 0
    The steady‑state output reduces to:
        h_i = (1 - f_i) * 1 + r_i = 1 - f_i + r_i
    where f_i is the failure_rate and r_i is righting_time_index.
    """
    m = len(endpoints)
    if m == 0:
        return np.array([], dtype=float)

    health = np.empty(m, dtype=float)
    for idx, ep in enumerate(endpoints):
        f = ep.failure_rate
        r = ep.righting_time_index
        health[idx] = 1.0 - f + r
    return health


# ----------------------------------------------------------------------
# Parent B – Gini coefficient (fairness) utilities
# ----------------------------------------------------------------------


def gini_coefficient(values: List[float]) -> float:
    """
    Compute the Gini coefficient for a non‑negative distribution.
    Returns 0 for an empty or all‑zero input.
    """
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0.0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


# ----------------------------------------------------------------------
# Tropical max‑plus regret network (bridge between A and B)
# ----------------------------------------------------------------------


def tropical_regret_gains(health_scores: np.ndarray,
                          actions: List[MathAction]) -> Dict[str, float]:
    """
    Evaluate a two‑layer tropical (max‑plus) network.

    Input vector: `h` = health_scores (size n)
    Weight matrix: each row j corresponds to an action and is a constant
    offset `-cost_j`.  Hence the max‑plus product reduces to:
        g_j = max_i (h_i) - cost_j

    The function returns a mapping from action id to its gain `g_j`.
    """
    if health_scores.size == 0:
        max_h = 0.0
    else:
        max_h = float(np.max(health_scores))

    gains: Dict[str, float] = {}
    for act in actions:
        gains[act.id] = max_h - act.cost
    return gains


# ----------------------------------------------------------------------
# Hoeffding bound utilities (Parent A)
# ----------------------------------------------------------------------


def hoeffding_bound(range_width: float,
                    n: int,
                    delta: float) -> float:
    """
    Compute the Hoeffding bound ε such that with probability `1‑δ`
    the true mean lies within `ε` of the empirical mean.
    """
    if n <= 0:
        return float('inf')
    return math.sqrt((range_width ** 2 * math.log(1.0 / delta)) / (2.0 * n))


# ----------------------------------------------------------------------
# Statistics container for incremental Hoeffding monitoring
# ----------------------------------------------------------------------


@dataclass
class HoeffdingStats:
    """Tracks incremental statistics for a set of gain observations."""
    count: int = 0
    sum_gain: float = 0.0
    min_gain: float = field(default_factory=lambda: float('inf'))
    max_gain: float = field(default_factory=lambda: float('-inf'))

    def update(self, gain: float) -> None:
        self.count += 1
        self.sum_gain += gain
        self.min_gain = min(self.min_gain, gain)
        self.max_gain = max(self.max_gain, gain)

    @property
    def mean(self) -> float:
        return self.sum_gain / self.count if self.count else 0.0

    @property
    def range(self) -> float:
        if self.count == 0:
            return 0.0
        return self.max_gain - self.min_gain


# ----------------------------------------------------------------------
# Hybrid decision logic combining Hoeffding & Gini
# ----------------------------------------------------------------------


def update_stats_and_maybe_split(gains: Dict[str, float],
                                 stats: Dict[str, HoeffdingStats],
                                 delta: float = 0.05,
                                 gini_thr: float = 0.3) -> Tuple[bool, str]:
    """
    Update per‑action Hoeffding statistics with the latest gain, compute the
    Gini coefficient of the current gain distribution and decide whether a
    split is justified.

    Returns `(split_decision, reason)`.
    """
    # Update statistics
    for aid, g in gains.items():
        if aid not in stats:
            stats[aid] = HoeffdingStats()
        stats[aid].update(g)

    # Identify best and second‑best actions by empirical mean gain
    sorted_actions = sorted(stats.items(),
                            key=lambda kv: kv[1].mean,
                            reverse=True)
    if len(sorted_actions) < 2:
        return False, "Not enough actions to compare"

    (best_id, best_stat), (second_id, second_stat) = sorted_actions[:2]

    # Hoeffding bound on the difference of means
    eps_best = hoeffding_bound(best_stat.range, best_stat.count, delta)
    eps_second = hoeffding_bound(second_stat.range, second_stat.count, delta)

    # Decision criterion: with high probability the best mean exceeds the
    # second best by more than the combined uncertainty.
    if best_stat.mean - eps_best > second_stat.mean + eps_second:
        # Fairness check via Gini coefficient of the *current* gain values
        current_gains = list(gains.values())
        gini = gini_coefficient(current_gains)
        if gini <= gini_thr:
            return True, f"Split on action '{best_id}' (Hoeffding OK, Gini={gini:.3f})"
        else:
            return False, f"Gini too high ({gini:.3f} > {gini_thr})"
    else:
        return False, "Hoeffding bound not satisfied"


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------


def _example_endpoints() -> List[Endpoint]:
    return [
        Endpoint(failures=2, failure_threshold=10, righting_time_index=0.8),
        Endpoint(failures=0, failure_threshold=5, righting_time_index=0.6),
        Endpoint(failures=5, failure_threshold=20, righting_time_index=0.9),
    ]


def _example_actions() -> List[MathAction]:
    return [
        MathAction(id="A", expected_value=1.0, cost=0.2),
        MathAction(id="B", expected_value=0.8, cost=0.1),
        MathAction(id="C", expected_value=0.5, cost=0.4),
    ]


if __name__ == "__main__":
    # 1. Compute health scores from endpoints
    eps = _example_endpoints()
    health = compute_health_scores(eps)
    print("Health scores:", health)

    # 2. Map health scores to tropical regret gains for each action
    acts = _example_actions()
    gains = tropical_regret_gains(health, acts)
    print("Tropical regret gains:", gains)

    # 3. Incrementally update statistics and test split decision
    stats: Dict[str, HoeffdingStats] = {}
    split, reason = update_stats_and_maybe_split(gains, stats,
                                                 delta=0.05,
                                                 gini_thr=0.35)
    print("Split decision:", split, "| Reason:", reason)

    # Run a few more iterations to demonstrate convergence
    for step in range(5):
        # Simulate slight variation in health (e.g., due to new data)
        health = health + np.random.normal(scale=0.02, size=health.shape)
        gains = tropical_regret_gains(health, acts)
        split, reason = update_stats_and_maybe_split(gains, stats,
                                                     delta=0.05,
                                                     gini_thr=0.35)
        print(f"Step {step+1:02d} – split:{split} – {reason}")