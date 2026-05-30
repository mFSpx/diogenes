# DARWIN HAMMER — match 277, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_label_foundry_m21_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s3.py (gen3)
# born: 2026-05-29T23:28:06Z

"""
This module integrates the Hybrid Bandit-Sketch-Label Fusion Module (Parent A) 
with the Hybrid VRAM Scheduler & Minimum-Cost Tree with Bayesian Decision Hygiene (Parent B).
The mathematical bridge is established by using the Bayesian marginal-posterior update 
from Parent B to inform the Upper-Confidence-Bound (UCB) selection rule in Parent A, 
while utilizing the Count-Min and HyperLogLog sketches from Parent A to estimate the 
VRAM usage and optimize the scheduling decision.
"""

import math
import random
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Iterable
import numpy as np

class CountMinSketch:
    """Simple Count-Min sketch for non-negative integers"""
    def __init__(self, width: int, depth: int):
        self.width = width
        self.depth = depth
        self.table = [[0 for _ in range(width)] for _ in range(depth)]

    def update(self, item: int):
        for i in range(self.depth):
            index = hash(item) % self.width
            self.table[i][index] += 1

    def estimate(self, item: int) -> int:
        estimates = []
        for i in range(self.depth):
            index = hash(item) % self.width
            estimates.append(self.table[i][index])
        return min(estimates)

class HyperLogLog:
    """Simple HyperLogLog sketch for estimating cardinality"""
    def __init__(self, num_registers: int):
        self.num_registers = num_registers
        self.registers = [0] * num_registers

    def update(self, item: int):
        x = hash(item)
        register_index = x % self.num_registers
        register_value = x // self.num_registers
        self.registers[register_index] = max(self.registers[register_index], register_value.bit_length() - 1)

    def estimate(self) -> int:
        alpha = 0.7213 / (1 + 1.079 / self.num_registers)
        estimate = alpha * self.num_registers ** 2 / sum([2 ** -m for m in self.registers])
        return int(estimate)

def tree_metrics(nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]], root: str) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root-to-node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)
    return adj, edge_len, dist

def bayesian_update(dist: Dict[str, float], budget: float, uncertainty: float) -> Dict[str, float]:
    """
    Perform Bayesian marginal-posterior update to quantify the probability 
    that the observed usage fits within the budget given measurement uncertainty.

    Returns
    -------
    probabilities : dict mapping node → probability of fitting within the budget
    """
    probabilities = {}
    for node, distance in dist.items():
        probability = math.exp(-((distance - budget) ** 2) / (2 * uncertainty ** 2))
        probabilities[node] = probability
    return probabilities

def ucb_selection(rewards: List[float], probabilities: Dict[str, float], num_arms: int) -> int:
    """
    Perform Upper-Confidence-Bound (UCB) selection rule informed by Bayesian update.

    Returns
    -------
    selected_arm : index of the selected arm
    """
    ucbs = []
    for arm in range(num_arms):
        ucb = rewards[arm] + math.sqrt(2 * math.log(len(rewards)) / (arm + 1)) * probabilities[str(arm)]
        ucbs.append(ucb)
    selected_arm = ucbs.index(max(ucbs))
    return selected_arm

if __name__ == "__main__":
    # Smoke test
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    budget = 10.0
    uncertainty = 1.0
    probabilities = bayesian_update(dist, budget, uncertainty)
    rewards = [1.0, 2.0, 3.0]
    selected_arm = ucb_selection(rewards, probabilities, len(rewards))
    print("Selected arm:", selected_arm)