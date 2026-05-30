# DARWIN HAMMER — match 4723, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s5.py (gen2)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_hybrid_hybrid_m2726_s4.py (gen6)
# born: 2026-05-29T23:57:39Z

"""
This module fuses the core topologies of hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s5.py
and hybrid_hybrid_doomsday_cale_hybrid_hybrid_hybrid_m2726_s4.py. The mathematical bridge between
the two parents lies in the integration of the Hoeffding bound with the adaptive learning rate.
The Hoeffding bound is used to compute the confidence interval for the probability of a node
broadcasting, while the adaptive learning rate is used to adjust the step size based on the
proximity to the next Doomsday.

The fusion combines the probabilistic primitives from Parent A with the Doomsday algorithm
and adaptive learning rate from Parent B. The hybrid algorithm uses the Hoeffding bound to
compute the confidence interval for the probability of a node broadcasting and then adjusts
the step size using the adaptive learning rate based on the proximity to the next Doomsday.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from collections.abc import Mapping, Hashable

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]
NodeId = str
Edge = tuple[NodeId, NodeId, int]  # (src, dst, impedance)

# ----------------------------------------------------------------------
# Parent A – probabilistic primitives and Hoeffding bound
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    if total_phases < 1 or current_phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


# ----------------------------------------------------------------------
# Parent B – Doomsday algorithm and adaptive learning rate
# ----------------------------------------------------------------------
def doomsday_rule(year: int, month: int, day: int) -> int:
    return (date(year, month, day).weekday() + 1) % 7


def days_until_next_doomsday(ref: date = None) -> int:
    if ref is None:
        ref = date.today()
    today_doomsday = doomsday_rule(ref.year, ref.month, ref.day)
    year_doomsday = doomsday_rule(ref.year, 4, 4)
    delta = (year_doomsday - today_doomsday) % 7
    return 7 if delta == 0 else delta


def adaptive_learning_rate(base_mu: float = 0.5) -> float:
    days = days_until_next_doomsday()
    factor = 0.5 + 0.5 * (days - 1) / 6.0
    return base_mu * factor


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_broadcast_probability(total_phases: int, current_phase: int, confidence: float) -> float:
    prob = broadcast_probability(total_phases, current_phase)
    epsilon = hoeffding_bound(1.0, confidence, total_phases)
    return max(0.0, min(1.0, prob - epsilon))


def hybrid_adaptive_step_size(base_mu: float, total_phases: int, current_phase: int) -> float:
    prob = broadcast_probability(total_phases, current_phase)
    adaptive_mu = adaptive_learning_rate(base_mu)
    return adaptive_mu * prob


def hybrid_doomsday_hoeffding(delta: float, n: int, base_mu: float) -> tuple[float, float]:
    r = 1.0
    epsilon = hoeffding_bound(r, delta, n)
    adaptive_mu = adaptive_learning_rate(base_mu)
    return epsilon, adaptive_mu


if __name__ == "__main__":
    total_phases = 10
    current_phase = 5
    confidence = 0.05
    base_mu = 0.5
    delta = 0.05
    n = 100

    print(hybrid_broadcast_probability(total_phases, current_phase, confidence))
    print(hybrid_adaptive_step_size(base_mu, total_phases, current_phase))
    epsilon, adaptive_mu = hybrid_doomsday_hoeffding(delta, n, base_mu)
    print(f"Epsilon: {epsilon}, Adaptive Mu: {adaptive_mu}")