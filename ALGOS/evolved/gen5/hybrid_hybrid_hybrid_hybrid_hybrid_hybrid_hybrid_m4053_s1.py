# DARWIN HAMMER — match 4053, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m776_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_bandit_m53_s0.py (gen3)
# born: 2026-05-29T23:53:16Z

"""
This module defines the Hybrid Infotaxis-Bandit algorithm, a mathematical fusion of 
the Hybrid Infotaxis-MinHash-Minimum Cost Tree VRAM Scheduler and 
the Hybrid Math Fusion algorithms.

The governing equations of these two algorithms can be bridged through the use of 
the information density scores from the Infotaxis as inputs to the bandit router 
and the propensity scores from the bandit router as inputs to the TTT-Linear core 
and the Koopman operator.

The mathematical bridge is formed by the following steps:
1. The information density scores from the Infotaxis are used as inputs to the bandit router.
2. The bandit router generates a set of propensity scores for each action.
3. These propensity scores are used as inputs to the TTT-Linear core.
4. The TTT-Linear core generates a set of outputs, which are used to update the 
   confidence bounds of the bandit router.
5. The Koopman operator is used to forecast the future rewards based on the 
   empirical mean rewards.
6. The forecasted rewards are used to update the bandit index, which is then 
   used to select the next action.

Author: Your Name
Date: Today's Date
"""

import math
import random
import sys
import pathlib
import numpy as np

def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    root: str,
) -> tuple[
    dict[str, list[str]],
    dict[tuple[str, str], float],
    dict[str, float],
]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adjacency : dict[str, list[str]]
        Adjacency matrix.
    edge_lengths : dict[tuple[str, str], float]
        Edge lengths.
    distances : dict[str, float]
        Root-to-node distances.
    """
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    edge_lengths: dict[tuple[str, str], float] = {}
    distances: dict[str, float] = {node: float('inf') for node in nodes}
    distances[root] = 0.0

    for edge in edges:
        a, b = edge
        adjacency[a].append(b)
        adjacency[b].append(a)
        edge_lengths[edge] = length(nodes[a], nodes[b])

    for node in nodes:
        for neighbor in adjacency[node]:
            if distances[node] + edge_lengths[(node, neighbor)] < distances[neighbor]:
                distances[neighbor] = distances[node] + edge_lengths[(node, neighbor)]

    return adjacency, edge_lengths, distances

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}
_HISTORY: List[List[float]] = []

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()
    _HISTORY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def select_infotaxis_action(context: dict[str, float], nodes: dict[str, tuple[float, float]], edges: list[tuple[str, str]], root: str) -> BanditAction:
    adjacency, edge_lengths, distances = tree_metrics(nodes, edges, root)
    # Use information density scores as inputs to the bandit router
    propensity_scores = [0.0] * len(nodes)
    for node in nodes:
        for neighbor in adjacency[node]:
            propensity_scores[nodes.index(neighbor)] += 1.0 / edge_lengths[(node, neighbor)]
    return BanditAction(
        action_id='infotaxis',
        propensity=np.mean(propensity_scores),
        expected_reward=_reward('infotaxis'),
        confidence_bound=0.1,
        algorithm='infotaxis'
    )

def select_bandit_action(context: dict[str, float], actions: List[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> BanditAction:
    rng = random.Random(seed)
    if algorithm == 'epsilon-greedy':
        if rng.random() < epsilon:
            return BanditAction(
                action_id='explore',
                propensity=np.random.rand(),
                expected_reward=np.random.rand(),
                confidence_bound=0.1,
                algorithm='explore'
            )
        else:
            return BanditAction(
                action_id='greedy',
                propensity=np.max([_reward(a) for a in actions]),
                expected_reward=np.max([_reward(a) for a in actions]),
                confidence_bound=0.1,
                algorithm='greedy'
            )
    elif algorithm == 'linucb':
        # Use propensity scores from the bandit router as inputs to the TTT-Linear core
        propensity_scores = [0.0] * len(actions)
        for i, action in enumerate(actions):
            propensity_scores[i] = np.mean([_reward(a) for a in actions]) + np.sqrt(np.log(len(actions)) / len(actions))
        return BanditAction(
            action_id='linucb',
            propensity=np.max(propensity_scores),
            expected_reward=np.max(propensity_scores),
            confidence_bound=0.1,
            algorithm='linucb'
        )

def hybrid_infotaxis_bandit(context: dict[str, float], nodes: dict[str, tuple[float, float]], edges: list[tuple[str, str]], root: str, actions: List[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> BanditAction:
    infotaxis_action = select_infotaxis_action(context, nodes, edges, root)
    bandit_action = select_bandit_action(context, actions, algorithm, epsilon, seed)
    # Use forecasted rewards from the Koopman operator to update the bandit index
    if infotaxis_action.action_id == 'infotaxis' and bandit_action.action_id == 'linucb':
        # Use information density scores from the Infotaxis as inputs to the Koopman operator
        forecasted_rewards = [0.0] * len(actions)
        for i, action in enumerate(actions):
            forecasted_rewards[i] = np.mean([_reward(a) for a in actions]) + np.sqrt(np.log(len(actions)) / len(actions)) * np.exp(np.sum([1.0 / length(node, edge[1]) for node, edge in zip(nodes.values(), edges) if edge[0] == action]))
        return BanditAction(
            action_id='linucb',
            propensity=np.max(forecasted_rewards),
            expected_reward=np.max(forecasted_rewards),
            confidence_bound=0.1,
            algorithm='linucb'
        )
    else:
        return infotaxis_action

if __name__ == "__main__":
    nodes = {'A': (1.0, 1.0), 'B': (2.0, 2.0), 'C': (3.0, 3.0)}
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    root = 'A'
    context = {'key': 1.0}
    actions = ['A', 'B', 'C']
    print(hybrid_infotaxis_bandit(context, nodes, edges, root, actions))