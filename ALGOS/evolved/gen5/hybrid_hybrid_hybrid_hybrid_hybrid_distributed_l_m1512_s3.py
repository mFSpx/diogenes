# DARWIN HAMMER — match 1512, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_pherom_krampus_stickers_m507_s0.py (gen4)
# parent_b: hybrid_distributed_leader_e_thanatosis_m65_s1.py (gen1)
# born: 2026-05-29T23:37:09Z

"""Hybrid Leader Election via Perceptual Hashing, MinHash Similarity, and Simulated Annealing.

Parents:
- **hybrid_hybrid_hybrid_pherom_krampus_stickers_m507_s0.py** – provides
  perceptual hashing, MinHash signatures and entropy‑weighted pheromone ideas.
- **hybrid_distributed_leader_e_thanatosis_m65_s1.py** – provides a
  probabilistic broadcast leader election combined with simulated‑annealing
  acceptance.

Mathematical Bridge:
The bridge is the *probabilistic acceptance* concept that appears in both
parents.  In the DARWIN‑HAMMER side the pheromone update is weighted by an
entropy term; in the THANATOSIS side the acceptance of a new leader set is
governed by an annealing probability.  We fuse them by:
1. Using MinHash‑derived similarity between node signatures to modulate the
   broadcast probability (the “broadcast‑probability” function becomes a
   similarity‑aware function).
2. Computing an entropy weight from each node’s MinHash signature and feeding
   it into the annealing acceptance probability, thus turning the acceptance
   probability into an *entropy‑aware* function.
The resulting algorithm performs a distributed leader election that respects
both structural similarity (via perceptual hashing) and thermodynamic‑style
exploration (via simulated annealing).

The module implements three core hybrid operations:
- `node_signature` – builds a MinHash signature from a perceptual hash.
- `similarity_adjusted_broadcast` – adapts broadcast probability using
  pair‑wise Hamming similarity of signatures.
- `entropy_weighted_acceptance` – adjusts the annealing acceptance probability
  with an entropy factor derived from the signature token distribution.
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
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
Features = Mapping[Node, List[float]]  # raw numeric attributes per node

# ----------------------------------------------------------------------
# Perceptual hashing utilities (from Parent A)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash: 1 bit per value indicating >= average."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:  # limit to 64 bits
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return bin(a ^ b).count('1')

# ----------------------------------------------------------------------
# MinHash utilities (from Parent A)
# ----------------------------------------------------------------------
def minhash_signature(tokens: Set[str], num_hashes: int = 7) -> List[int]:
    """Compute a MinHash signature for a set of string tokens."""
    signatures: List[int] = []
    for seed in range(num_hashes):
        # deterministic hash per seed
        def hash_fn(x: str) -> int:
            h = hashlib.md5((x + str(seed)).encode()).hexdigest()
            return int(h, 16)
        min_hash = min(hash_fn(tok) for tok in tokens)
        signatures.append(min_hash)
    return signatures

# ----------------------------------------------------------------------
# Entropy utility (derived from KRAMPUS side)
# ----------------------------------------------------------------------
def entropy_from_signature(sig: List[int]) -> float:
    """Shannon entropy of the distribution of bits across the MinHash signature."""
    # flatten bits of all integers in the signature
    bits = []
    for val in sig:
        bits.extend([(val >> i) & 1 for i in range(64)])  # 64 bits per hash
    total = len(bits)
    if total == 0:
        return 0.0
    counts = Counter(bits)
    ent = 0.0
    for cnt in counts.values():
        p = cnt / total
        ent -= p * math.log(p, 2)
    return ent

# ----------------------------------------------------------------------
# Broadcast & acceptance utilities (from Parent B)
# ----------------------------------------------------------------------
def broadcast_probability(phases: int, phase: int) -> float:
    """Base broadcast probability decreasing with phase."""
    if phases < 1 or phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))

def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance for energy change delta_e at given temperature."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Geometric cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)

# ----------------------------------------------------------------------
# Hybrid building blocks
# ----------------------------------------------------------------------
def node_signature(values: List[float]) -> Tuple[int, List[int], float]:
    """
    Compute a node's perceptual hash, its MinHash signature and the entropy
    of that signature.
    Returns (phash, signature, entropy).
    """
    ph = compute_phash(values)
    # use each bit as a token string for MinHash
    tokens = {f'bit{i}_{(ph >> i) & 1}' for i in range(64)}
    sig = minhash_signature(tokens)
    ent = entropy_from_signature(sig)
    return ph, sig, ent

def similarity_adjusted_broadcast(base_p: float,
                                  node_ph: int,
                                  neighbor_phs: List[int]) -> float:
    """
    Reduce broadcast probability proportionally to the average Hamming
    similarity with neighbours.  More similar neighbourhoods broadcast less
    often, encouraging diversity.
    """
    if not neighbor_phs:
        return base_p
    avg_ham = sum(hamming_distance(node_ph, nb) for nb in neighbor_phs) / len(neighbor_phs)
    # similarity = 1 - normalized Hamming distance
    similarity = 1.0 - (avg_ham / 64.0)
    # damp the probability: highly similar neighbourhoods (similarity≈1) get a
    # smaller factor.
    factor = 1.0 - similarity * 0.5  # clamp to [0.5,1.0]
    return max(0.0, min(1.0, base_p * factor))

def entropy_weighted_acceptance(delta_e: float,
                                temperature: float,
                                avg_entropy: float,
                                entropy_scale: float = 0.3) -> float:
    """
    Combine Metropolis acceptance with an entropy multiplier.
    Higher entropy (more diverse signatures) increases the chance to accept
    a new leader set.
    """
    base = acceptance_probability(delta_e, temperature)
    # map entropy (0..~6 bits) to a multiplier in [1, 1+entropy_scale]
    mult = 1.0 + entropy_scale * (avg_entropy / 6.0)  # 6 is approx max entropy for 64 bits
    return max(0.0, min(1.0, base * mult))

# ----------------------------------------------------------------------
# Full hybrid leader election
# ----------------------------------------------------------------------
def hybrid_leader_election(graph: Graph,
                           node_features: Features,
                           phases: int = 8,
                           seed: int | str | None = None,
                           t0: float = 1.0,
                           alpha: float = 0.95,
                           dormancy_floor: float = 0.05) -> Set[Node]:
    """
    Distributed leader election that fuses:
    * Perceptual‑hash / MinHash similarity (Parent A)
    * Simulated‑annealing acceptance with entropy weighting (Parent B)

    Returns the set of elected leaders.
    """
    rng = random.Random(seed)
    # Pre‑compute per‑node hashes, signatures and entropies
    phash_map: Dict[Node, int] = {}
    entropy_map: Dict[Node, float] = {}
    for n, vals in node_features.items():
        ph, _, ent = node_signature(vals)
        phash_map[n] = ph
        entropy_map[n] = ent

    undecided: Set[Node] = set(graph)
    leaders: Set[Node] = set()
    for phase in range(1, phases + 1):
        if not undecided:
            break

        base_p = broadcast_probability(phases, phase)

        # decide which undecided nodes broadcast, using similarity‑adjusted prob.
        broadcasts: Set[Node] = set()
        for n in undecided:
            neighbor_phs = [phash_map[nb] for nb in graph.get(n, set()) if nb in phash_map]
            p_adj = similarity_adjusted_broadcast(base_p, phash_map[n], neighbor_phs)
            if rng.random() < p_adj:
                broadcasts.add(n)

        # nodes that broadcast and have no broadcasting neighbour become candidates
        candidates: Set[Node] = {
            n for n in broadcasts
            if not (graph.get(n, set()) & broadcasts)
        }

        # compute energy change (delta_e) as change in leader count
        delta_e = len(candidates) - len(leaders)

        # temperature for this phase
        temp = cooling_temperature(phase, t0, alpha)

        # average entropy of the candidate set (fallback to overall avg)
        if candidates:
            avg_ent = sum(entropy_map[n] for n in candidates) / len(candidates)
        else:
            avg_ent = sum(entropy_map.values()) / len(entropy_map) if entropy_map else 0.0

        accept_prob = entropy_weighted_acceptance(delta_e, temp, avg_ent)

        # accept the new candidates with probability accept_prob
        if rng.random() < accept_prob:
            leaders.update(candidates)
            undecided.difference_update(candidates)

        # optional dormancy: nodes that have not broadcast for many phases may
        # be forced to broadcast with a minimal floor probability.
        if dormancy_floor > 0 and rng.random() < dormancy_floor:
            # pick a random undecided node to become a leader as a safety net
            if undecided:
                forced = rng.choice(list(undecided))
                leaders.add(forced)
                undecided.remove(forced)

    return leaders

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic graph: a ring of 6 nodes with random feature vectors
    rng = np.random.default_rng(42)
    nodes = [f"n{i}" for i in range(6)]
    graph: Graph = {n: set() for n in nodes}
    for i, n in enumerate(nodes):
        nxt = nodes[(i + 1) % len(nodes)]
        prev = nodes[(i - 1) % len(nodes)]
        graph[n].update({nxt, prev})

    # Random float vectors (length 10) as node features
    node_features: Features = {
        n: rng.random(10).tolist() for n in nodes
    }

    leaders = hybrid_leader_election(
        graph,
        node_features,
        phases=5,
        seed=123,
        t0=1.0,
        alpha=0.9,
        dormancy_floor=0.02,
    )
    print("Elected leaders:", leaders)