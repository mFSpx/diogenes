# DARWIN HAMMER — match 1222, survivor 7
# gen: 5
# parent_a: hybrid_indy_learning_vector_hybrid_fold_change_d_m38_s0.py (gen4)
# parent_b: hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s2.py (gen2)
# born: 2026-05-29T23:34:39Z

"""Hybrid algorithm merging:
- Parent A: indy_learning_vector / hybrid bandit routing with log‑count statistics.
- Parent B: distributed leader election with kinetic “strike” integration.

Mathematical bridge:
1. Each data element is a float vector `v`.
2. From Parent B we compute a **kinetic score** `k(v)` by integrating a simple
   force‑velocity‑distance dynamics (`v_{t+1}=v_t+f_t·Δt`,
   `d_{t+1}=d_t+v_t·Δt`).  The score is the peak velocity attained.
3. From Parent A we build a perceptual‑hash similarity graph `G` and run a
   **maximal‑independent‑set (MIS) leader election**.  The broadcast
   probability of node *i* is biased by `k(v_i)`:
   `p_i ∝ 1 + α·k(v_i)`.
4. The same kinetic scores weight the **propensity** of actions in a
   contextual bandit: `propensity_a ∝ log(1+cnt_a)·(1+β·k_a)`.
Thus the kinetic physics of the vector drives both clustering/election and
action selection, yielding a unified hybrid system.
"""

import math
import random
import sys
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
from typing import List, Dict, Set, Tuple, Any, Hashable
import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (bandit)
# ----------------------------------------------------------------------
class BanditAction:
    def __init__(self, action_id: str, propensity: float, expected_reward: float,
                 confidence_bound: float, algorithm: str):
        self.action_id = action_id
        self.propensity = propensity
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm

class BanditUpdate:
    def __init__(self, context_id: str, action_id: str, reward: float,
                 propensity: float):
        self.context_id = context_id
        self.action_id = action_id
        self.reward = reward
        self.propensity = propensity

_POLICY: Dict[str, Tuple[float, int]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, (0.0, 0))
    return total / n if n else 0.0

def _count(action: str) -> int:
    return _POLICY.get(action, (0.0, 0))[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        total, n = _POLICY.get(u.action_id, (0.0, 0))
        _POLICY[u.action_id] = (total + u.reward, n + 1)

def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()

# ----------------------------------------------------------------------
# Parent B utilities (leader election & kinetic integration)
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
    return (a ^ b).bit_count()

def build_graph(elements: List[List[float]]) -> Dict[Hashable, Set[Hashable]]:
    """Connect two nodes if their perceptual hashes differ by ≤4 bits."""
    hashes: Dict[int, int] = {i: compute_phash(el) for i, el in enumerate(elements)}
    graph: Dict[int, Set[int]] = {i: set() for i in range(len(elements))}
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if hamming_distance(hashes[i], hashes[j]) <= 4:
                graph[i].add(j)
                graph[j].add(i)
    return graph

def kinetic_score(forces: np.ndarray, dt: float = 0.1) -> float:
    """
    Integrate a simple 1‑D dynamics:
        v_{t+1} = v_t + f_t * dt
        d_{t+1} = d_t + v_t * dt
    Returns the peak velocity encountered (scalar kinetic score).
    """
    v = 0.0
    d = 0.0
    peak_v = 0.0
    for f in forces:
        v += float(f) * dt
        d += v * dt
        if v > peak_v:
            peak_v = v
    return peak_v

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_leader_election(elements: List[List[float]],
                           forces_list: List[np.ndarray],
                           alpha: float = 0.5) -> Set[int]:
    """
    Perform a maximal‑independent‑set (MIS) election where each node's
    broadcast probability is biased by its kinetic score.

    Returns the set of elected leader indices.
    """
    if len(elements) != len(forces_list):
        raise ValueError("Elements and forces_list must have the same length.")
    graph = build_graph(elements)

    # Compute kinetic scores
    kinetic_scores = [kinetic_score(f) for f in forces_list]

    # Compute broadcast weights
    weights = [1.0 + alpha * k for k in kinetic_scores]

    # Randomized order weighted by broadcast probability
    order = list(range(len(elements)))
    random.shuffle(order)
    order.sort(key=lambda i: -weights[i])  # higher weight first

    elected: Set[int] = set()
    removed: Set[int] = set()

    for node in order:
        if node in removed:
            continue
        elected.add(node)
        # Remove node and its neighbours from future consideration
        removed.add(node)
        removed.update(graph[node])
    return elected

def hybrid_bandit_propensity(action_counts: Counter,
                             kinetic_map: Dict[str, float],
                             beta: float = 0.3) -> List[BanditAction]:
    """
    Compute propensities for a set of actions.
    - `action_counts` maps action_id -> observed count.
    - `kinetic_map` maps action_id -> kinetic score (e.g. derived from forces).
    The propensity is proportional to log(1+count) * (1 + beta * kinetic).
    Returns a list of BanditAction objects sorted by descending propensity.
    """
    actions: List[BanditAction] = []
    for aid, cnt in action_counts.items():
        log_cnt = math.log1p(cnt)
        kin = kinetic_map.get(aid, 0.0)
        propensity = log_cnt * (1.0 + beta * kin)
        expected = _reward(aid)
        # Simple confidence bound using sqrt(1/count)
        bound = math.sqrt(1.0 / (cnt + 1))
        actions.append(BanditAction(
            action_id=aid,
            propensity=propensity,
            expected_reward=expected,
            confidence_bound=bound,
            algorithm="hybrid"
        ))
    actions.sort(key=lambda a: -a.propensity)
    return actions

def hybrid_process_context(context: str,
                           action_vocab: List[str],
                           forces_dict: Dict[str, np.ndarray]) -> List[BanditAction]:
    """
    End‑to‑end hybrid pipeline:
    1. Chunk the raw text context into words (simulating the indy_learning_vector chunker).
    2. Count occurrences → log‑count statistics.
    3. Map each word (action) to a kinetic score via `forces_dict`.
    4. Produce propensity‑weighted BanditAction list.
    """
    # Simple whitespace tokenisation
    tokens = context.split()
    counts = Counter(tokens)

    # Ensure every token has a kinetic entry; missing entries get zero vector.
    kinetic_map: Dict[str, float] = {}
    for token in counts:
        forces = forces_dict.get(token, np.zeros(10))
        kinetic_map[token] = kinetic_score(forces)

    return hybrid_bandit_propensity(counts, kinetic_map)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic data
    random.seed(42)
    np.random.seed(42)

    # 10 elements each 20‑dimensional
    elements = [list(np.random.rand(20)) for _ in range(10)]
    forces_list = [np.random.randn(15) for _ in range(10)]

    # Hybrid leader election
    leaders = hybrid_leader_election(elements, forces_list, alpha=0.7)
    print("Elected leaders:", leaders)

    # Build a dummy context and kinetic forces for actions
    context = "alpha beta gamma alpha delta beta epsilon alpha"
    action_vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    forces_dict = {aid: np.random.randn(12) for aid in action_vocab}

    actions = hybrid_process_context(context, action_vocab, forces_dict)
    for a in actions:
        print(f"Action {a.action_id}: propensity={a.propensity:.4f}, reward={a.expected_reward:.4f}")

    # Update policy with mock rewards
    updates = [BanditUpdate(context_id="ctx1", action_id=a.action_id,
                           reward=random.random(), propensity=a.propensity) for a in actions[:3]]
    update_policy(updates)
    print("Policy after updates:", _POLICY)