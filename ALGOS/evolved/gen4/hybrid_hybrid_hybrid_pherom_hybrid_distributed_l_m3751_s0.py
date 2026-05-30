# DARWIN HAMMER — match 3751, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_pheromone_hyb_hybrid_minimum_cost__m792_s2.py (gen3)
# parent_b: hybrid_distributed_leader_e_thanatosis_m65_s2.py (gen1)
# born: 2026-05-29T23:51:23Z

"""
Hybrid Distributed Pheromone Leader Election Algorithm
Parents:
- hybrid_pheromone_hybrid_distributed_l_m41_s0.py (perceptual hashing of pheromone signals)
- hybrid_distributed_leader_e_thanatosis_m65_s2.py (hybrid leader election via simulated annealing)

Mathematical Bridge:
A pheromone signal vector is reduced to a 64-bit perceptual hash.
The Hamming similarity between two node hashes provides a data-driven likelihood
that an edge between those nodes is “relevant”.  This likelihood is fed into
the cooling temperature equation of the leader election algorithm, allowing the
temperature to decay with both pheromone evidence and phase.  The resulting
temperature modulates the acceptance probability of the Metropolis rule, yielding
a hybrid stochastic process that respects both the locality of pheromone signals
and the annealing dynamics of the leader election algorithm.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np
from typing import Set, Tuple

# Type aliases
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Perceptual hashing utilities (from Parent A)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Return a 64-bit perceptual hash of a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    # Pad remaining bits with zeros if fewer than 64 values
    bits <<= max(0, 64 - len(values))
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64-bit integers."""
    return (a ^ b).bit_count()

def broadcast_probability(phase: int, steps: int) -> float:
    """Probability that a pheromone broadcast succeeds at a given phase/step."""
    if phase < 1 or steps < 1:
        raise ValueError('phase and steps must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - steps)))

# ----------------------------------------------------------------------
# Leader election utilities (from Parent B)
# ----------------------------------------------------------------------
def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Original B: exponential cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)

def hybrid_temperature(phases: int, phase: int, steps: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """
    Combine the decay of broadcast probability and annealing temperature.

    temperature = cooling_temperature(phase‑1) * broadcast_probability(...)
    """
    p = broadcast_probability(phases, steps)
    return cooling_temperature(phase - 1, t0, alpha) * p

# ----------------------------------------------------------------------
# Hybrid Distributed Pheromone Leader Election Algorithm
# ----------------------------------------------------------------------
def pheromone_leader_election(graph: Set[Edge], pheromones: Dict[Edge, List[float]], temperature: float) -> Edge:
    """Elect a leader node in a graph with pheromone signals and temperature."""
    max_conflict = 0
    leader = None
    for node in graph:
        conflict = 0
        for neighbor in graph:
            if node != neighbor and neighbor in pheromones and node in pheromones[neighbor]:
                conflict += hamming_distance(pheromones[neighbor][node], pheromones[node][neighbor])
        if conflict > max_conflict:
            max_conflict = conflict
            leader = node
    if random.random() < np.exp(-max_conflict / temperature):
        return leader
    return None

def simulated_annealing(graph: Set[Edge], pheromones: Dict[Edge, List[float]], temperature: float, steps: int) -> Edge:
    """Simulated annealing to elect a leader node in a graph with pheromone signals and temperature."""
    candidate = random.choice(list(graph))
    for _ in range(steps):
        neighbor = random.choice(list(graph))
        if neighbor != candidate and candidate in pheromones and neighbor in pheromones[candidate]:
            pheromone = pheromones[candidate][neighbor]
            new_pheromone = compute_phash(pheromones[neighbor][candidate])
            if hamming_distance(pheromone, new_pheromone) < hamming_distance(pheromones[neighbor][candidate], pheromone):
                pheromones[candidate][neighbor] = new_pheromone
        if random.random() < np.exp(-np.sum([hamming_distance(pheromones[u][v], pheromones[v][u]) for u, v in graph]) / temperature):
            return candidate
    return None

def hybrid_election(graph: Set[Edge], pheromones: Dict[Edge, List[float]], steps: int, phases: int) -> Edge:
    """Hybrid distributed pheromone leader election algorithm."""
    temperature = hybrid_temperature(phases, 1, steps)
    for phase in range(2, phases + 1):
        temperature = hybrid_temperature(phases, phase, steps)
        leader = pheromone_leader_election(graph, pheromones, temperature)
        if leader is not None:
            return leader
    return None

if __name__ == "__main__":
    graph = {('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D'), ('D', 'E'), ('E', 'A')}
    pheromones = {
        ('A', 'B'): [compute_phash([0.5, 0.5])],
        ('B', 'C'): [compute_phash([0.3, 0.7])],
        ('C', 'A'): [compute_phash([0.8, 0.2])],
        ('A', 'D'): [compute_phash([0.1, 0.9])],
        ('D', 'E'): [compute_phash([0.6, 0.4])],
        ('E', 'A'): [compute_phash([0.4, 0.6])]
    }
    steps = 10
    phases = 10
    leader = hybrid_election(graph, pheromones, steps, phases)
    print(leader)