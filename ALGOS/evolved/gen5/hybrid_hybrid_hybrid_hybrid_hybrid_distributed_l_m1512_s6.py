# DARWIN HAMMER — match 1512, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_pherom_krampus_stickers_m507_s0.py (gen4)
# parent_b: hybrid_distributed_leader_e_thanatosis_m65_s1.py (gen1)
# born: 2026-05-29T23:37:09Z

import sys
import random
import math
import hashlib
from pathlib import Path
from collections import Counter
from typing import Mapping, Hashable, Set, List, Dict, Tuple, Iterable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
ValuesMap = Mapping[Node, List[float]]
TokensMap = Mapping[Node, Set[str]]

# ----------------------------------------------------------------------
# Utilities from Parent A (perceptual similarity)
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
    """Deterministic MinHash signature for a set of tokens."""
    signatures: List[int] = []
    for seed in range(num_hashes):
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
    weight_phash: float = 0.5,
) -> float:
    """
    Combined similarity:
        - Normalised Hamming similarity (1 - distance/64)
        - MinHash Jaccard similarity
    Weighted by ``weight_phash`` (rest goes to MinHash).
    """
    ham_sim = 1.0 - hamming_distance(phash_a, phash_b) / 64.0
    mh_sim = jaccard_similarity(sig_a, sig_b)
    return weight_phash * ham_sim + (1.0 - weight_phash) * mh_sim


# ----------------------------------------------------------------------
# Utilities from Parent B (annealing & broadcast)
# ----------------------------------------------------------------------
def broadcast_probability(phases: int, phase: int) -> float:
    """Probability that a node broadcasts in the given phase (geometric decay)."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    # decay factor ensures early phases are more aggressive
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
# Additional pheromone / entropy utilities (new)
# ----------------------------------------------------------------------
def shannon_entropy(values: List[float], bins: int = 16) -> float:
    """Compute Shannon entropy of a numeric list using a fixed number of bins."""
    if not values:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, density=True)
    # avoid log(0) by filtering zero probabilities
    probs = hist[hist > 0]
    return -np.sum(probs * np.log2(probs))


def compute_pheromone_weights(values_map: ValuesMap) -> Dict[Node, float]:
    """
    Derive a pheromone‑like weight from the entropy of each node's value vector.
    Higher entropy → stronger pheromone → higher influence on temperature.
    Normalised to [0, 1].
    """
    entropies = {n: shannon_entropy(v) for n, v in values_map.items()}
    if not entropies:
        return {}
    max_e = max(entropies.values())
    min_e = min(entropies.values())
    span = max_e - min_e if max_e != min_e else 1.0
    return {n: (e - min_e) / span for n, e in entropies.items()}


# ----------------------------------------------------------------------
# Core preprocessing
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


# ----------------------------------------------------------------------
# Improved Hybrid Leader Election
# ----------------------------------------------------------------------
def hybrid_leader_election(
    graph: Graph,
    values_map: ValuesMap,
    tokens_map: TokensMap,
    phases: int = 6,
    seed: Optional[int | str] = None,
    t0: float = 1.0,
    alpha: float = 0.9,
    weight_phash: float = 0.5,
) -> Set[Node]:
    """
    Improved hybrid leader election.

    Key enhancements over the original design:
        • Normalised energy (average similarity) removes bias toward larger
          leader sets.
        • Global temperature is modulated by the average pheromone weight
          (entropy‑derived) to keep the annealing schedule sensitive to the
          underlying data distribution.
        • Energy computation is cached where possible to avoid O(|L|²) blow‑up.
        • The similarity weight between perceptual hash and MinHash is
          configurable via ``weight_phash``.
    """
    rng = random.Random(seed)
    phash_dict, sig_dict = precompute_node_features(values_map, tokens_map)
    pheromone_weights = compute_pheromone_weights(values_map)

    undecided: Set[Node] = set(graph)
    leaders: Set[Node] = set()

    # cache similarity between any two nodes to avoid recomputation
    _sim_cache: Dict[Tuple[Node, Node], float] = {}

    def similarity(a: Node, b: Node) -> float:
        key = (a, b) if a <= b else (b, a)  # unordered pair
        if key not in _sim_cache:
            _sim_cache[key] = node_similarity(
                phash_dict[a], phash_dict[b], sig_dict[a], sig_dict[b], weight_phash
            )
        return _sim_cache[key]

    def average_similarity(group: Set[Node], reference: Set[Node]) -> float:
        """
        Compute the mean similarity between every node in *group* and every node
        in *reference*.  Returns 0.0 when *reference* is empty.
        """
        if not reference:
            return 0.0
        total = 0.0
        for a in group:
            for b in reference:
                total += similarity(a, b)
        return total / (len(group) * len(reference))

    for phase in range(1, phases + 1):
        if not undecided:
            break

        # ------------------------------------------------------------------
        # 1. Broadcast step (geometric decay)
        # ------------------------------------------------------------------
        p = broadcast_probability(phases, phase)
        broadcasts = {n for n in undecided if rng.random() < p}

        # ------------------------------------------------------------------
        # 2. Maximal independent set filter (MIS)
        # ------------------------------------------------------------------
        candidates = {
            n
            for n in broadcasts
            if not (graph.get(n, set()) & broadcasts)  # no neighbour also broadcasting
        }

        if not candidates:
            continue

        # ------------------------------------------------------------------
        # 3. Energy evaluation (average similarity)
        # ------------------------------------------------------------------
        current_avg = average_similarity(leaders, leaders)
        new_avg = average_similarity(leaders | candidates, leaders | candidates)
        delta_e = new_avg - current_avg  # positive → more redundant (higher energy)

        # ------------------------------------------------------------------
        # 4. Temperature modulation (annealing + pheromone)
        # ------------------------------------------------------------------
        base_temp = cooling_temperature(phase - 1, t0, alpha)
        # average pheromone weight of nodes involved in this phase
        involved = leaders | candidates
        avg_pheromone = (
            sum(pheromone_weights.get(n, 0.0) for n in involved) / len(involved)
            if involved
            else 0.0
        )
        temperature = base_temp * (1.0 + avg_pheromone)  # pheromone lifts temperature

        # ------------------------------------------------------------------
        # 5. Acceptance step
        # ------------------------------------------------------------------
        accept_prob = acceptance_probability(delta_e, temperature)
        if rng.random() < accept_prob:
            leaders.update(candidates)
            undecided.difference_update(candidates)

    return leaders

# ----------------------------------------------------------------------
# Example stub (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic test to ensure the module runs
    example_graph: Graph = {
        0: {1, 2},
        1: {0, 3},
        2: {0, 3},
        3: {1, 2, 4},
        4: {3},
    }
    example_values: ValuesMap = {
        i: [random.random() for _ in range(10)] for i in example_graph
    }
    example_tokens: TokensMap = {
        i: {f"tok{j}" for j in range(random.randint(1, 5))} for i in example_graph
    }

    leaders = hybrid_leader_election(
        example_graph,
        example_values,
        example_tokens,
        phases=5,
        seed=42,
        t0=1.0,
        alpha=0.85,
        weight_phash=0.6,
    )
    print("Elected leaders:", leaders)