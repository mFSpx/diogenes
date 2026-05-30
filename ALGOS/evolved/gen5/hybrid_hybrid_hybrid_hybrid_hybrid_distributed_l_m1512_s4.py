# DARWIN HAMMER — match 1512, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_pherom_krampus_stickers_m507_s0.py (gen4)
# parent_b: hybrid_distributed_leader_e_thanatosis_m65_s1.py (gen1)
# born: 2026-05-29T23:37:09Z

"""Hybrid Leader Election with Perceptual Hash, MinHash, and Simulated Annealing

This module fuses the core topologies of two parent algorithms:

* **PARENT A** – combines pheromone‑based maximal independent set selection with
  MinHash perceptual similarity (functions ``compute_phash`` and ``minhash_signature``).
* **PARENT B** – distributed leader election driven by a broadcast probability
  and a simulated‑annealing acceptance rule (functions ``broadcast_probability``,
  ``acceptance_probability`` and ``cooling_temperature``).

**Mathematical bridge**

Both parents manipulate a *probabilistic* state that can be expressed as an
energy‑like quantity:

* In A the pheromone level (derived from entropy) modulates the likelihood that a
  node will join the independent set.
* In B the annealing temperature modulates the acceptance of a change in the
  leader set.

We map the pheromone‑derived *entropy weight* to the annealing temperature and
use the **similarity** between two nodes (computed from the Hamming distance of
their perceptual hash and the Jaccard overlap of their MinHash signatures) as
the *energy difference* ΔE.  The hybrid leader election therefore proceeds with
the broadcast probability of B, but the decision to keep a newly elected leader
uses the acceptance probability of B evaluated on ΔE derived from A’s similarity
measure and a temperature that decays like the pheromone cooling schedule.

The result is a single unified algorithm that simultaneously respects
graph‑topology (independent‑set constraints), perceptual similarity, and a
temperature‑driven stochastic acceptance criterion.
"""

import sys
import random
import math
import hashlib
from pathlib import Path
from collections import Counter
from typing import Mapping, Hashable, Set, List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
ValuesMap = Mapping[Node, List[float]]
TokensMap = Mapping[Node, Set[str]]

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Perceptual hash: 1 bit per value indicating >= average (max 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return bin(a ^ b).count("1")


def minhash_signature(tokens: Set[str], num_hashes: int = 7) -> List[int]:
    """MinHash signature for a set of tokens."""
    signatures: List[int] = []
    for seed in range(num_hashes):
        # deterministic hash function per seed
        def hash_fn(x: str) -> int:
            return int(hashlib.md5((x + str(seed)).encode()).hexdigest(), 16)

        hashed = [hash_fn(tok) for tok in tokens]
        signatures.append(min(hashed) if hashed else 0)
    return signatures


def jaccard_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity for MinHash signatures."""
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def node_similarity(
    phash_a: int,
    phash_b: int,
    sig_a: List[int],
    sig_b: List[int],
) -> float:
    """
    Combined similarity:
        - Normalised Hamming similarity (1 - distance/64)
        - MinHash Jaccard similarity
    Weighted equally and averaged.
    """
    ham_sim = 1.0 - hamming_distance(phash_a, phash_b) / 64.0
    mh_sim = jaccard_similarity(sig_a, sig_b)
    return 0.5 * ham_sim + 0.5 * mh_sim


# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def broadcast_probability(phases: int, phase: int) -> float:
    """Probability that a node broadcasts in the given phase."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance rule."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0.0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Exponential cooling schedule."""
    if k < 0 or t0 < 0 or not (0.0 <= alpha <= 1.0):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def precompute_node_features(
    values_map: ValuesMap, tokens_map: TokensMap
) -> Tuple[Dict[Node, int], Dict[Node, List[int]]]:
    """
    Compute and cache perceptual hashes and MinHash signatures for all nodes.
    Returns two dictionaries indexed by node.
    """
    phash_dict: Dict[Node, int] = {}
    sig_dict: Dict[Node, List[int]] = {}
    for n in values_map:
        phash_dict[n] = compute_phash(values_map[n])
        sig_dict[n] = minhash_signature(tokens_map.get(n, set()))
    return phash_dict, sig_dict


def hybrid_leader_election(
    graph: Graph,
    values_map: ValuesMap,
    tokens_map: TokensMap,
    phases: int = 6,
    seed: int | str | None = None,
    t0: float = 1.0,
    alpha: float = 0.9,
) -> Set[Node]:
    """
    Hybrid leader election:

    1. Nodes broadcast with probability derived from ``broadcast_probability``.
    2. Among broadcasters, a node becomes a *candidate leader* if none of its
       neighbours also broadcast (maximal independent set constraint).
    3. The set of new candidates is evaluated against the current leader set
       using a similarity‑based energy ΔE.
    4. Acceptance of the new candidates follows the Metropolis rule with a
       temperature that follows the cooling schedule (simulated annealing).

    Returns the final set of elected leaders.
    """
    rng = random.Random(seed)
    phash_dict, sig_dict = precompute_node_features(values_map, tokens_map)

    undecided: Set[Node] = set(graph)
    leaders: Set[Node] = set()

    for phase in range(1, phases + 1):
        if not undecided:
            break

        # 1. Broadcast step
        p = broadcast_probability(phases, phase)
        broadcasts = {n for n in undecided if rng.random() < p}

        # 2. Maximal independent set filter
        candidates = {
            n
            for n in broadcasts
            if not (graph.get(n, set()) & broadcasts)  # no neighbour also broadcasting
        }

        if not candidates:
            continue

        # 3. Energy evaluation (ΔE)
        # Energy is defined as the total similarity between the candidate set and
        # the existing leaders (higher similarity → higher energy, i.e. less
        # desirable because leaders become redundant).
        def total_similarity(nodes: Set[Node]) -> float:
            total = 0.0
            for a in nodes:
                for b in leaders:
                    total += node_similarity(
                        phash_dict[a], phash_dict[b], sig_dict[a], sig_dict[b]
                    )
            return total

        current_energy = total_similarity(leaders)
        new_energy = total_similarity(leaders | candidates)
        delta_e = new_energy - current_energy

        # 4. Acceptance step
        temperature = cooling_temperature(phase, t0, alpha)
        accept_prob = acceptance_probability(delta_e, temperature)

        if rng.random() < accept_prob:
            # Accept the new candidates
            leaders.update(candidates)
            undecided.difference_update(candidates)

    return leaders


def hybrid_maximal_independent_set(
    graph: Graph,
    values_map: ValuesMap,
    tokens_map: TokensMap,
    seed: int | str | None = None,
) -> Set[Node]:
    """
    A stand‑alone maximal independent set algorithm that uses the same similarity
    and temperature machinery as ``hybrid_leader_election`` but runs until a
    stable independent set is reached (no further accepted improvements).
    """
    rng = random.Random(seed)
    phash_dict, sig_dict = precompute_node_features(values_map, tokens_map)

    mis: Set[Node] = set()
    undecided: Set[Node] = set(graph)

    iteration = 0
    while undecided:
        iteration += 1
        # Randomly shuffle undecided nodes to avoid bias
        shuffled = list(undecided)
        rng.shuffle(shuffled)

        # Greedy independent set construction
        for n in shuffled:
            if not (graph.get(n, set()) & mis):
                mis.add(n)
                undecided.remove(n)

        # Evaluate whether to keep the new nodes using annealing
        # Energy: sum of pairwise similarities inside the set (lower is better)
        def intra_similarity(nodes: Set[Node]) -> float:
            total = 0.0
            nodes_list = list(nodes)
            for i in range(len(nodes_list)):
                for j in range(i + 1, len(nodes_list)):
                    a, b = nodes_list[i], nodes_list[j]
                    total += node_similarity(
                        phash_dict[a], phash_dict[b], sig_dict[a], sig_dict[b]
                    )
            return total

        current_energy = intra_similarity(mis)
        # Propose a random removal of a fraction of nodes to explore
        removal_fraction = 0.1
        to_remove = {n for n in mis if rng.random() < removal_fraction}
        proposed = mis - to_remove
        new_energy = intra_similarity(proposed)

        delta_e = new_energy - current_energy
        temperature = cooling_temperature(iteration, t0=1.0, alpha=0.93)
        accept = acceptance_probability(delta_e, temperature)

        if rng.random() < accept:
            mis = proposed
            # Re‑add removed nodes to undecided for possible re‑selection
            undecided.update(to_remove)

    return mis


def hybrid_similarity_matrix(
    nodes: List[Node],
    values_map: ValuesMap,
    tokens_map: TokensMap,
) -> np.ndarray:
    """
    Returns a symmetric matrix S where S[i, j] = node_similarity(i, j).
    Useful for external analysis or visualisation.
    """
    phash_dict, sig_dict = precompute_node_features(values_map, tokens_map)
    n = len(nodes)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            sim = node_similarity(
                phash_dict[nodes[i]],
                phash_dict[nodes[j]],
                sig_dict[nodes[i]],
                sig_dict[nodes[j]],
            )
            mat[i, j] = mat[j, i] = sim
    return mat


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a small random graph
    rng = random.Random(42)
    num_nodes = 12
    nodes = [f"N{i}" for i in range(num_nodes)]
    graph: Dict[Node, Set[Node]] = {n: set() for n in nodes}
    # Random undirected edges
    for _ in range(num_nodes * 2):
        a, b = rng.sample(nodes, 2)
        graph[a].add(b)
        graph[b].add(a)

    # Random values for perceptual hashing (10 floats per node)
    values_map: Dict[Node, List[float]] = {
        n: [rng.random() for _ in range(10)] for n in nodes
    }

    # Random token sets for MinHash (5‑10 tokens per node)
    token_pool = [f"tok{i}" for i in range(30)]
    tokens_map: Dict[Node, Set[str]] = {
        n: set(rng.sample(token_pool, rng.randint(5, 10))) for n in nodes
    }

    leaders = hybrid_leader_election(
        graph,
        values_map,
        tokens_map,
        phases=5,
        seed=123,
        t0=1.0,
        alpha=0.9,
    )
    print("Hybrid leaders:", leaders)

    mis = hybrid_maximal_independent_set(
        graph, values_map, tokens_map, seed=456
    )
    print("Hybrid MIS size:", len(mis), "nodes:", mis)

    sim_matrix = hybrid_similarity_matrix(nodes, values_map, tokens_map)
    print("Similarity matrix shape:", sim_matrix.shape)
    # Verify matrix is symmetric and values lie in [0,1]
    assert np.allclose(sim_matrix, sim_matrix.T, atol=1e-9)
    assert np.all((sim_matrix >= 0.0) & (sim_matrix <= 1.0))
    print("Smoke test completed successfully.")