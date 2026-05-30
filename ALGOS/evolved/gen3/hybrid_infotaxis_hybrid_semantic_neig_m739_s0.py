# DARWIN HAMMER — match 739, survivor 0
# gen: 3
# parent_a: infotaxis.py (gen0)
# parent_b: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py (gen2)
# born: 2026-05-29T23:30:39Z

"""
Hybrid Infotaxis-Morphology Algorithm: fusion of infotaxis.py and hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py
The mathematical bridge is established by using the recovery priority as a multiplicative scaling factor for the expected entropy.
This allows the topology of the information space to be modulated by the physical-morphology space.
"""
import math
import random
import sys
from dataclasses import dataclass
import numpy as np
from pathlib import Path

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Maps righting time index to a normalized priority in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def entropy(probabilities: list[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)


def expected_entropy(p_hit: float, hit_state: list[float], miss_state: list[float]) -> float:
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


def hybrid_expected_entropy(p_hit: float, hit_state: list[float], miss_state: list[float], m: Morphology) -> float:
    recovery_p = recovery_priority(m)
    return recovery_p * expected_entropy(p_hit, hit_state, miss_state)


def best_action(actions: dict[str, tuple[float, list[float], list[float]]], m: Morphology) -> str:
    if not actions:
        raise ValueError('actions required')
    return min(actions, key=lambda a: (hybrid_expected_entropy(*actions[a], m), a))


def hybrid_affinity(actions: dict[str, tuple[float, list[float], list[float]]], m: Morphology) -> dict:
    affinity = {}
    for action, (p_hit, hit_state, miss_state) in actions.items():
        affinity[action] = hybrid_expected_entropy(p_hit, hit_state, miss_state, m)
    return affinity


if __name__ == "__main__":
    m = Morphology(length=1.0, width=1.0, height=1.0, mass=1.0)
    actions = {
        'action1': (0.5, [0.4, 0.6], [0.3, 0.7]),
        'action2': (0.6, [0.2, 0.8], [0.1, 0.9]),
    }
    print(best_action(actions, m))
    print(hybrid_affinity(actions, m))