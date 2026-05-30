# DARWIN HAMMER — match 1222, survivor 8
# gen: 5
# parent_a: hybrid_indy_learning_vector_hybrid_fold_change_d_m38_s0.py (gen4)
# parent_b: hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s2.py (gen2)
# born: 2026-05-29T23:34:39Z

import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Bandit core
# ----------------------------------------------------------------------
class BanditAction:
    """Container for an action in the contextual bandit."""
    def __init__(self, action_id: str, expected_reward: float = 0.0,
                 propensity: float = 0.0, confidence_bound: float = 0.0,
                 algorithm: str = "hybrid"):
        self.action_id = action_id
        self.expected_reward = expected_reward
        self.propensity = propensity
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm

class BanditUpdate:
    """Result of pulling an arm."""
    def __init__(self, context_id: str, action_id: str,
                 reward: float, propensity: float):
        self.context_id = context_id
        self.action_id = action_id
        self.reward = reward
        self.propensity = propensity

# Global policy store: action_id -> [cumulative_reward, count, last_update]
_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n, _ = _POLICY.get(action, [0.0, 0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    """Apply a batch of bandit updates to the global policy."""
    for u in updates:
        total, n, last_update = _POLICY.get(u.action_id, [0.0, 0.0, 0.0])
        _POLICY[u.action_id] = [total + u.reward, n + 1, u.propensity]

# ----------------------------------------------------------------------
# Parent B – Perceptual hash + kinetic score utilities
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """64‑bit perceptual hash based on average comparison."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit hashes."""
    return (a ^ b).bit_count()

def build_graph(elements: List[List[float]]) -> Dict[int, Set[int]]:
    """Connect two nodes if their perceptual hashes differ by ≤4 bits."""
    hashes = [compute_phash(el) for el in elements]
    graph: Dict[int, Set[int]] = {i: set() for i in range(len(elements))}
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if hamming_distance(hashes[i], hashes[j]) <= 4:
                graph[i].add(j)
                graph[j].add(i)
    return graph

def compute_kinetic_score(vector: List[float]) -> float:
    """
    Integrate a “strike” over the vector interpreted as a time‑series of forces.
    Simple Euler integration:
        velocity_{t+1} = velocity_t + force_t * dt
        distance_{t+1} = distance_t + velocity_t * dt
    Returns the peak velocity (scalar kinetic score).
    """
    dt = 1.0  # unit time step
    velocity = 0.0
    distance = 0.0
    peak_velocity = 0.0
    for force in vector:
        velocity += force * dt
        distance += velocity * dt
        if abs(velocity) > abs(peak_velocity):
            peak_velocity = velocity
    # Normalise by vector length to keep scores comparable
    return abs(peak_velocity) / (len(vector) or 1)

# ----------------------------------------------------------------------
# Hybrid core – three public functions
# ----------------------------------------------------------------------
def elect_leaders(elements: List[List[float]],
                 kinetic_scores: List[float]) -> List[int]:
    """
    Perform a maximal‑independent‑set (MIS) election on the perceptual‑hash graph.
    Nodes are processed in descending order of kinetic score, giving energetic
    nodes higher chance to become leaders.
    Returns the list of elected node indices.
    """
    graph = build_graph(elements)
    # Pair (score, node) sorted descending
    ordered = sorted(enumerate(kinetic_scores),
                     key=lambda x: x[1],
                     reverse=True)
    selected: Set[int] = set()
    banned: Set[int] = set()
    for node, _ in ordered:
        if node in banned:
            continue
        selected.add(node)
        banned.update(graph[node])   # neighbours cannot be selected
    return list(selected)

def bandit_choose_action(context_score: float,
                         actions: List[BanditAction],
                         alpha: float = 0.5,
                         beta: float = 1.0) -> BanditAction:
    """
    Compute propensities using a soft‑max over (β·expected_reward + α·context_score)
    and return the sampled action.
    """
    logits = []
    for a in actions:
        logit = beta * a.expected_reward + alpha * context_score
        logits.append(logit)
    # Numerical stability
    max_logit = max(logits) if logits else 0.0
    exp_vals = [math.exp(l - max_logit) for l in logits]
    total = sum(exp_vals) or 1.0
    probs = [v / total for v in exp_vals]

    # Assign propensities back to actions (useful for logging)
    for a, p in zip(actions, probs):
        a.propensity = p

    # Sample according to probabilities
    chosen = random.choices(actions, weights=probs, k=1)[0]
    return chosen

def hybrid_process(elements: List[List[float]],
                   action_ids: List[str],
                   num_updates: int = 10) -> List[Tuple[int, BanditAction]]:
    """
    End‑to‑end hybrid routine:
    1. Compute kinetic scores for all elements.
    2. Elect leaders using MIS biased by kinetic scores.
    3. For each leader, treat its kinetic score as context and select a bandit action.
    Returns a list of (leader_index, chosen_action) pairs.
    """
    # 1. Kinetic scores
    kinetic_scores = [compute_kinetic_score(v) for v in elements]

    # 2. Leader election
    leaders = elect_leaders(elements, kinetic_scores)

    # 3. Initialise bandit actions (shared across leaders)
    actions = [BanditAction(aid, expected_reward=_reward(aid)) for aid in action_ids]

    results: List[Tuple[int, BanditAction]] = []
    updates: List[BanditUpdate] = []
    for lid in leaders:
        ctx = kinetic_scores[lid]
        chosen = bandit_choose_action(ctx, actions)
        results.append((lid, chosen))

        # Simulate a synthetic reward (for demonstration): reward = ctx * random_factor
        reward = ctx * random.random()
        propensity = chosen.propensity
        updates.append(BanditUpdate(context_id=str(lid),
                                   action_id=chosen.action_id,
                                   reward=reward,
                                   propensity=propensity))

    # Apply updates to policy
    update_policy(updates)

    return results