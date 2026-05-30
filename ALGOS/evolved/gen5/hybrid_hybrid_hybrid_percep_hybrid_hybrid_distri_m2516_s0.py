# DARWIN HAMMER — match 2516, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_perceptual_de_hybrid_label_foundry_m1030_s1.py (gen4)
# parent_b: hybrid_hybrid_distributed_l_hybrid_physarum_netw_m110_s5.py (gen4)
# born: 2026-05-29T23:42:34Z

"""
Hybrid Perceptual-Physarum Algorithm
=================================
This module fuses the perceptual hashing utilities and RBF surrogate modeling 
from *hybrid_hybrid_perceptual_de_hybrid_label_foundry_m1030_s1.py* (Parent A)
with the Physarum-inspired flow network and leader election from 
*hybrid_hybrid_distributed_l_hybrid_physarum_netw_m110_s5.py* (Parent B).

The mathematical bridge is established by using the perceptual hash as a 
clustering key for the Physarum conductance update and as a feature for the 
leader election. The temperature from Parent B scales the Physarum conductance 
update and is also used as the Metropolis temperature for the leader-selection 
acceptance probability.

"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Tuple, Dict

Vector = Sequence[float]

# ---------- Parent A: perceptual hashing utilities ----------
def compute_dhash(values: List[float]) -> int:
    """Difference hash: 1 bit per adjacent pair, 1 if decreasing."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: List[float]) -> int:
    """Average hash limited to first 64 values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count('1')

# ---------- Parent B: Physarum and leader election ----------
def broadcast_probability(phases: int, phase: int) -> float:
    """Original A: p = 1 / 2^{phases‑phase}, clamped to [0,1]."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Original B: exponential cooling schedule."""
    if k < 0 or t0 < 0 or not (0.0 <= alpha <= 1.0):
        raise ValueError("k, t0, and alpha must be valid")
    return t0 * (alpha ** k)

# ---------- Hybrid Perceptual-Physarum Algorithm ----------
@dataclass(frozen=True)
class Node:
    id: int
    values: List[float]

def perceptual_physarum_step(nodes: List[Node], phase: int, phases: int) -> Dict[Tuple[int, int], float]:
    temperature = cooling_temperature(phase) * broadcast_probability(phases, phase)
    conductances = {}
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            node_i = nodes[i]
            node_j = nodes[j]
            dhash_i = compute_dhash(node_i.values)
            dhash_j = compute_dhash(node_j.values)
            conductance = math.exp(-(hamming_distance(dhash_i, dhash_j) ** 2) / (2 * temperature ** 2))
            conductances[(node_i.id, node_j.id)] = conductance
            conductances[(node_j.id, node_i.id)] = conductance
    return conductances

def leader_election_step(nodes: List[Node], phase: int, phases: int) -> int:
    temperature = cooling_temperature(phase) * broadcast_probability(phases, phase)
    current_leader = random.choice(nodes)
    for node in nodes:
        dhash_current = compute_dhash(current_leader.values)
        dhash_node = compute_dhash(node.values)
        if random.random() < math.exp(-(hamming_distance(dhash_current, dhash_node) ** 2) / (2 * temperature ** 2)):
            current_leader = node
    return current_leader.id

def hybrid_perceptual_physarum_smoke_test():
    np.random.seed(42)
    random.seed(42)
    nodes = [Node(i, np.random.rand(100).tolist()) for i in range(10)]
    phases = 10
    for phase in range(phases):
        conductances = perceptual_physarum_step(nodes, phase, phases)
        leader = leader_election_step(nodes, phase, phases)
        print(f"Phase {phase+1}, Leader: {leader}")

if __name__ == "__main__":
    hybrid_perceptual_physarum_smoke_test()