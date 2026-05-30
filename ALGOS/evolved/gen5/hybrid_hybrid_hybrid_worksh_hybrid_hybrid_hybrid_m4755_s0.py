# DARWIN HAMMER — match 4755, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s3.py (gen2)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_minimum_cost__m693_s2.py (gen4)
# born: 2026-05-29T23:57:54Z

"""
Fusing hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s3.py and 
hybrid_hybrid_hybrid_fisher_hybrid_minimum_cost__m693_s2.py into a unified system.

The mathematical bridge between the two parent algorithms lies in their use of 
stochastic weights and distance-modulated confidence terms. The former is used 
in the weekday weight vector calculation of parent A, while the latter is a key 
component of the HybridBanditTree in parent B. By integrating these two 
concepts, we can create a hybrid algorithm that leverages the strengths of both.

Specifically, we will use the weekday weight vector to modulate the confidence 
terms in the HybridBanditTree, allowing the algorithm to adapt to changing 
conditions and make more informed decisions.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Constants
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

# Utility helpers
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    import datetime as dt
    return (dt.date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Tuple[str, ...], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


@dataclass(frozen=True)
class Point:
    """Immutable 2‑D point."""
    x: float
    y: float


@dataclass(frozen=True)
class BanditUpdate:
    """Observation used to update the hybrid policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float = 1.0  


class HybridBanditTree:
    """
    A self‑contained hybrid of a minimum‑cost tree and a contextual bandit.
    The tree supplies a *distance‑modulated* confidence term for each (context, action)
    pair, making the integration deeper than a simple additive score.
    """

    def __init__(self,
                 nodes: Dict[str, Point],
                 edges: List[Tuple[str, str]],
                 root: str,
                 path_weight: float = 0.2,
                 confidence_alpha: float = 1.0):
        """
        Parameters
        ----------
        nodes
            Mapping from node identifier to its geometric location.
        edges
            Undirected edges forming a tree (or forest). Cycles are ignored.
        root
            Identifier of the root node; distances are measured from it.
        path_weight
            Weight applied to the sum of root‑to‑node distances in the tree cost.
        confidence_alpha
            Scaling factor for the confidence term; larger values increase exploration.
        """
        self._validate_graph(nodes, edges, root)
        self.nodes = nodes
        self.root = root
        self.path_weight = path_weight
        self.confidence_alpha = confidence_alpha

        self.adj: Dict[str, List[str]] = {n: [] for n in nodes}
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)

        self.root_distances = self._bfs_distances(root)

        self._stats: Dict[Tuple[str, str], List[float]] = {}

    def _validate_graph(self, nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str):
        if root not in nodes:
            raise ValueError("Root node not in nodes")
        for edge in edges:
            if edge[0] not in nodes or edge[1] not in nodes:
                raise ValueError("Edge node not in nodes")

    def _bfs_distances(self, root: str) -> Dict[str, float]:
        distances = {root: 0.0}
        queue = [root]
        while queue:
            node = queue.pop(0)
            for neighbor in self.adj[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1.0
                    queue.append(neighbor)
        return distances

    def get_confidence_term(self, context_id: str, action_id: str) -> float:
        distance = self.root_distances.get(action_id, 0.0)
        confidence_term = self.confidence_alpha / (1.0 + self.path_weight * distance)
        return confidence_term

    def update_policy(self, update: BanditUpdate):
        context_id = update.context_id
        action_id = update.action_id
        reward = update.reward
        # Update policy using reward and confidence term
        pass


def hybrid_operation(nodes: Dict[str, Point],
                      edges: List[Tuple[str, str]],
                      root: str,
                      groups: Tuple[str, ...],
                      date: str) -> Dict[str, float]:
    """
    Perform the hybrid operation.

    Parameters
    ----------
    nodes
        Mapping from node identifier to its geometric location.
    edges
        Undirected edges forming a tree (or forest). Cycles are ignored.
    root
        Identifier of the root node; distances are measured from it.
    groups
        Tuple of group names.
    date
        Date string in YYYY-MM-DD format.

    Returns
    -------
    A dictionary with node identifiers as keys and their corresponding weights as values.
    """
    hybrid_bandit_tree = HybridBanditTree(nodes, edges, root)
    dow = doomsday(int(date[:4]), int(date[5:7]), int(date[8:]))
    weight_vec = weekday_weight_vector(groups, dow)

    node_weights = {}
    for node in nodes:
        confidence_term = hybrid_bandit_tree.get_confidence_term("context", node)
        node_weights[node] = weight_vec[list(nodes).index(node)] * confidence_term

    return node_weights


def allocate_hybrid(*,
                     total_units: float,
                     date: str,
                     deterministic_target_pct: float = 90.0,
                     groups: Tuple[str, ...] = GROUPS,
                     nodes: Dict[str, Point] = None,
                     edges: List[Tuple[str, str]] = None,
                     root: str = None) -> Dict[str, float]:
    """
    Split ``total_units`` into deterministic and LLM residual parts, then
    distribute the residual across ``groups`` using the weekday weight vector.

    Returns
    -------
    A dictionary with group names as keys and their corresponding weights as values.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    deterministic_units = total_units * deterministic_target_pct / 100.0
    residual_units = total_units - deterministic_units

    node_weights = hybrid_operation(nodes, edges, root, groups, date)

    group_weights = {}
    for group in groups:
        group_weights[group] = residual_units * node_weights.get(group, 0.0)

    return group_weights


if __name__ == "__main__":
    nodes = {"A": Point(0.0, 0.0), "B": Point(1.0, 0.0), "C": Point(0.0, 1.0)}
    edges = [("A", "B"), ("A", "C")]
    root = "A"
    groups = ("A", "B", "C")
    date = "2024-09-16"
    total_units = 100.0
    result = allocate_hybrid(total_units=total_units,
                             date=date,
                             nodes=nodes,
                             edges=edges,
                             root=root,
                             groups=groups)
    print(result)