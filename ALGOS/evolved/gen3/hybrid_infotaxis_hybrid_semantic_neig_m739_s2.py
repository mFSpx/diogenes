# DARWIN HAMMER — match 739, survivor 2
# gen: 3
# parent_a: infotaxis.py (gen0)
# parent_b: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py (gen2)
# born: 2026-05-29T23:30:39Z

"""
Hybrid Entropy-Morphology Search (HEMS) System
Parents:
- infotaxis.py (gradient-free entropy search)
- hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py (hybrid semantic-morphology neighbor system)

Mathematical Bridge:
The entropy of the hit and miss states from the infotaxis algorithm is used to modulate
the recovery priority of the candidate neighbors in the hybrid semantic-morphology system.
The hybrid affinity is redefined as  

    h = c * p_other * (1 - E_normalized)

where `c` is the cosine similarity, `p_other` is the recovery priority, and `E_normalized`
is the normalized expected entropy.

The governing equations of both parents are integrated through the following steps:
1. Compute the expected entropy of the hit and miss states using the infotaxis algorithm.
2. Normalize the expected entropy to a value in [0,1].
3. Use the normalized expected entropy to modulate the recovery priority of the candidate neighbors.
4. Compute the hybrid affinity using the modulated recovery priority and cosine similarity.
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

def entropy(probabilities: list[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)

def expected_entropy(p_hit: float, hit_state: list[float], miss_state: list[float]) -> float:
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)

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

def hybrid_affinity(cosine_similarity: float, morphology: Morphology, p_hit: float, hit_state: list[float], miss_state: list[float], max_index: float = 10.0) -> float:
    expected_ent = expected_entropy(p_hit, hit_state, miss_state)
    normalized_ent = expected_ent / (1 + expected_ent)  # Normalize to [0,1]
    recovery_prior = recovery_priority(morphology, max_index)
    return cosine_similarity * recovery_prior * (1 - normalized_ent)

def best_action(actions: dict[str, tuple[float, Morphology, list[float], list[float]]]) -> str:
    if not actions:
        raise ValueError('actions required')
    return max(actions, key=lambda a: (hybrid_affinity(*actions[a]), a))

def smoke_test():
    morphology = Morphology(1.0, 2.0, 3.0, 4.0)
    hit_state = [0.5, 0.5]
    miss_state = [0.2, 0.8]
    p_hit = 0.7
    cosine_similarity = 0.8
    max_index = 10.0

    affinity = hybrid_affinity(cosine_similarity, morphology, p_hit, hit_state, miss_state, max_index)
    print(f"Hybrid affinity: {affinity}")

    actions = {
        "action1": (cosine_similarity, morphology, hit_state, miss_state),
        "action2": (0.9, Morphology(2.0, 3.0, 4.0, 5.0), hit_state, miss_state),
    }
    best_action_name = best_action(actions)
    print(f"Best action: {best_action_name}")

if __name__ == "__main__":
    smoke_test()