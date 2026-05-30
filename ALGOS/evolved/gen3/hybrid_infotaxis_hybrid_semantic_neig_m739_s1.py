# DARWIN HAMMER — match 739, survivor 1
# gen: 3
# parent_a: infotaxis.py (gen0)
# parent_b: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py (gen2)
# born: 2026-05-29T23:30:39Z

"""
Hybrid Infotaxis-Semantic Neighbor System
Parents:
- infotaxis.py (gradient-free entropy search)
- hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py (morphology-driven recovery priority & circuit breaker)

Mathematical Bridge:
The entropy-based action selection in infotaxis.py is combined with the morphology-driven recovery priority in hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py.
The recovery priority `p ∈ [0,1]` derived from a document's morphology is used as a multiplicative scaling factor for the expected entropy `E ∈ [0, ∞)` between actions.
The hybrid affinity is defined as  

    h = E * p_other  

where `p_other` is the recovery priority of the candidate action.  
Thus the topology of the action space is modulated by the physical-morphology space, and the circuit-breaker logic can suppress actions whose morphology yields low priority.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Morphology & Recovery Priority
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Entropy-based Action Selection
# ----------------------------------------------------------------------
def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


def best_action(actions: Dict[str, Tuple[float, List[float], List[float], Morphology]]) -> str:
    if not actions:
        raise ValueError('actions required')
    return min(actions, key=lambda a: (expected_entropy(*actions[a][:3]) * recovery_priority(actions[a][3]), a))


def hybrid_affinity(actions: Dict[str, Tuple[float, List[float], List[float], Morphology]]) -> Dict[str, float]:
    affinities = {}
    for action, (p_hit, hit_state, miss_state, morphology) in actions.items():
        affinity = expected_entropy(p_hit, hit_state, miss_state) * recovery_priority(morphology)
        affinities[action] = affinity
    return affinities


def normalized_affinities(affinities: Dict[str, float]) -> Dict[str, float]:
    total_affinity = sum(affinities.values())
    if total_affinity <= 0:
        raise ValueError('total affinity must be positive')
    return {action: affinity / total_affinity for action, affinity in affinities.items()}


if __name__ == "__main__":
    actions = {
        'action1': (0.5, [0.2, 0.3, 0.5], [0.1, 0.4, 0.5], Morphology(1.0, 2.0, 3.0, 4.0)),
        'action2': (0.6, [0.3, 0.4, 0.3], [0.2, 0.5, 0.3], Morphology(2.0, 3.0, 4.0, 5.0)),
    }
    best_action_name = best_action(actions)
    affinities = hybrid_affinity(actions)
    normalized_affinities_dict = normalized_affinities(affinities)
    print(best_action_name)
    print(affinities)
    print(normalized_affinities_dict)