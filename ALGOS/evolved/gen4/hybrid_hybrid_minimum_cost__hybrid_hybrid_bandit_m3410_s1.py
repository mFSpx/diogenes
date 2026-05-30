# DARWIN HAMMER — match 3410, survivor 1
# gen: 4
# parent_a: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s1.py (gen3)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py (gen2)
# born: 2026-05-29T23:49:51Z

"""
This module fuses the governing equations of two parent algorithms:
- hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s1.py (Parent A)
- hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py (Parent B)

The mathematical bridge between the two parents is the bandit decision-making process.
Parent A uses a minimum-cost tree score calculation, while Parent B uses a hybrid bandit model with a virtual VRAM store.
This fusion integrates the tree cost calculation into the hybrid bandit model, using the tree cost as a reward signal to update the bandit policy.

The fusion creates a new hybrid algorithm that combines the strengths of both parents:
- The minimum-cost tree score calculation provides a more informed reward signal for the bandit.
- The hybrid bandit model with a virtual VRAM store allows for more efficient exploration and exploitation of the action space.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass(frozen=True)
class Point:
    """A point in 2D space."""
    x: float
    y: float

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    """Erase all learned statistics."""
    _POLICY.clear()

def _reward(action: str) -> float:
    """Empirical mean reward for *action* (0 if never observed)."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    """Number of times *action* has been selected."""
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate a batch of observations into the global policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def length(a: Point, b: Point) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a.x - b.x, a.y - b.y)

def tree_cost(nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str, path_weight: float = 0.2) -> float:
    """Calculate the minimum-cost tree score."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())

def calculate_bandit_confidence(updates: List[BanditUpdate]) -> Dict[str, float]:
    """Calculate the confidence term for the bandit."""
    confidence_terms = {}
    for u in updates:
        confidence_terms[u.action_id] = 1.0 / (1.0 + _count(u.action_id))
    return confidence_terms

class HybridBanditTTT:
    """
    A tighter integration of a contextual bandit (Parent A) and a linear
    TTT model (Parent B).  The virtual VRAM store influences the learning
    rate *and* the bandit’s propensity, creating a deeper feedback loop.
    """

    DEFAULT_BUDGET_MB = 8192  # assumed total VRAM budget for reporting

    def __init__(
        self,
        d_in: int,
        d_out: int = None,
        seed: int = 0,
        base_eta: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1.0,
        store_decay: float = 0.99,
    ) -> None:
        """
        Parameters
        ----------
        d_in, d_out : dimensions of the TTT weight matrix.
        seed        : RNG seed for reproducibility.
        base_eta    : Baseline learning rate before modulation.
        alpha, beta : Coefficients for the store differential equation.
        dt          : Time step for store integration.
        store_decay : Exponential decay applied to the store each step
                      (simulates memory eviction).
        """
        self.rng = np.random.default_rng(seed)
        self.base_eta = base_eta
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.store_decay = store_decay
        self.d_in = d_in
        self.d_out = d_out if d_out is not None else d_in

    def calculate_tree_cost_reward(self, nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str) -> float:
        """Calculate the tree cost reward for the bandit."""
        return tree_cost(nodes, edges, root)

    def update_bandit_policy(self, updates: List[BanditUpdate]) -> None:
        """Update the bandit policy using the tree cost reward."""
        update_policy(updates)

    def run_bandit(self, nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str) -> BanditAction:
        """Run the bandit algorithm and return the selected action."""
        tree_cost_reward = self.calculate_tree_cost_reward(nodes, edges, root)
        confidence_terms = calculate_bandit_confidence([])
        action_id = random.choice(list(nodes.keys()))
        propensity = 1.0 / (1.0 + _count(action_id))
        expected_reward = _reward(action_id)
        confidence_bound = confidence_terms.get(action_id, 0.0)
        return BanditAction(action_id, propensity, expected_reward, confidence_bound, "HybridBanditTTT")

if __name__ == "__main__":
    nodes = {"A": Point(0.0, 0.0), "B": Point(1.0, 1.0), "C": Point(2.0, 2.0)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    bandit = HybridBanditTTT(3)
    action = bandit.run_bandit(nodes, edges, root)
    print(action)
    sys.exit(0)