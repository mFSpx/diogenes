# DARWIN HAMMER — match 1512, survivor 5
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
    ham_sim = 1.0 - hamming_distance(phash_a, phash_b) / 64.0
    mh_sim = jaccard_similarity(sig_a, sig_b)
    return 0.5 * ham_sim + 0.5 * mh_sim


# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def broadcast_probability(phases: int, phase: int) -> float:
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))


def acceptance_probability(delta_e: float, temperature: float) -> float:
    if delta_e < 0:
        return 1.0
    if temperature <= 0.0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    if k < 0 or t0 < 0 or not (0.0 <= alpha <= 1.0):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def precompute_node_features(
    values_map: ValuesMap, tokens_map: TokensMap
) -> Tuple[Dict[Node, int], Dict[Node, List[int]]]:
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
    rng = random.Random(seed)
    phash_dict, sig_dict = precompute_node_features(values_map, tokens_map)

    undecided: Set[Node] = set(graph)
    leaders: Set[Node] = set()

    for phase in range(1, phases + 1):
        if not undecided:
            break

        p = broadcast_probability(phases, phase)
        broadcasts = {n for n in undecided if rng.random() < p}

        candidates = {
            n
            for n in broadcasts
            if not (graph.get(n, set()) & broadcasts)  
        }

        if not candidates:
            continue

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

        temperature = cooling_temperature(phase, t0, alpha)
        accept_prob = acceptance_probability(-delta_e, temperature)

        if rng.random() < accept_prob:
            leaders = leaders | candidates
            undecided -= candidates

    return leaders