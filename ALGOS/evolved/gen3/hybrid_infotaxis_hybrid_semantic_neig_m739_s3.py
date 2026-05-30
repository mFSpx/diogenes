# DARWIN HAMMER — match 739, survivor 3
# gen: 3
# parent_a: infotaxis.py (gen0)
# parent_b: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py (gen2)
# born: 2026-05-29T23:30:39Z

"""Hybrid Entropy-Morphology Search
Parents:
- infotaxis.py (gradient-free entropy search)
- hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py (hybrid semantic-morphology neighbor system)

Mathematical Bridge:
The recovery priority `p ∈ [0,1]` derived from a document's morphology in the hybrid semantic-morphology 
neighbor system is used to modulate the probability distribution in the entropy search. 
The hybrid system defines a new expected entropy calculation that incorporates the morphology-driven 
recovery priority as a scaling factor for the probability distribution.

    h = p * e  

where `p` is the recovery priority and `e` is the entropy.

The governing equations of both parents are integrated through the expected entropy calculation, 
which now depends on both the probability distribution and the morphology-driven recovery priority.
"""

import math
import numpy as np
from dataclasses import dataclass

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

def hybrid_expected_entropy(p_hit: float, hit_state: list[float], miss_state: list[float], 
                              morphology: Morphology) -> float:
    recovery_p = recovery_priority(morphology)
    scaled_hit_state = [p * recovery_p for p in hit_state]
    scaled_miss_state = [p * recovery_p for p in miss_state]
    hit_entropy = entropy(scaled_hit_state)
    miss_entropy = entropy(scaled_miss_state)
    return p_hit * hit_entropy + (1.0 - p_hit) * miss_entropy

def best_action(actions: dict[str, tuple[float, list[float], list[float], Morphology]]) -> str:
    if not actions:
        raise ValueError('actions required')
    return min(actions, key=lambda a: (hybrid_expected_entropy(*actions[a][:3]), a))

def hybrid_affinity(morphology: Morphology, probabilities: list[float]) -> float:
    recovery_p = recovery_priority(morphology)
    return recovery_p * entropy(probabilities)

if __name__ == "__main__":
    morphology = Morphology(1.0, 2.0, 3.0, 4.0)
    hit_state = [0.4, 0.3, 0.3]
    miss_state = [0.1, 0.1, 0.8]
    p_hit = 0.7
    print(hybrid_expected_entropy(p_hit, hit_state, miss_state, morphology))
    actions = {
        'action1': (0.7, [0.4, 0.3, 0.3], [0.1, 0.1, 0.8], Morphology(1.0, 2.0, 3.0, 4.0)),
        'action2': (0.6, [0.2, 0.4, 0.4], [0.3, 0.3, 0.4], Morphology(2.0, 3.0, 4.0, 5.0)),
    }
    print(best_action(actions))
    probabilities = [0.2, 0.3, 0.5]
    print(hybrid_affinity(morphology, probabilities))