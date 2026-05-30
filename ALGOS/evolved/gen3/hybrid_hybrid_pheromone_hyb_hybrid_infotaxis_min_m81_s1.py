# DARWIN HAMMER — match 81, survivor 1
# gen: 3
# parent_a: hybrid_pheromone_hybrid_distributed_l_m41_s1.py (gen2)
# parent_b: hybrid_infotaxis_minhash_m63_s5.py (gen1)
# born: 2026-05-29T23:25:36Z

"""Hybrid algorithm combining pheromone‑based maximal independent set selection
(PARENT A) with MinHash‑based perceptual similarity and entropy weighting
(PARENT B).

Mathematical bridge:
- Each node’s neighbourhood values are reduced to a perceptual hash
  (binary fingerprint) via `compute_phash` (A).
- The binary fingerprint is interpreted as a token set; a MinHash signature
  is built from these tokens (`signature` from B).  The signature approximates
  Jaccard similarity between neighbourhoods.
- Leader election (maximal independent set) uses broadcast probability as in
  A, but a node may become a leader only if its MinHash signature is
  sufficiently dissimilar from other broadcasting neighbours.  The similarity
  threshold couples the two topologies.
- After leaders are chosen, their pheromone values are updated proportionally
  to the Shannon entropy of their MinHash signature, merging the entropy
  computation of B with the pheromone update of A.

The module provides three core hybrid operations:
1. `node_neighbour_phash` – compute perceptual hash per node.
2. `node_signature` – obtain a MinHash signature from the hash‑derived tokens.
3. `hybrid_maximal_independent_set` – leader election that fuses broadcast
   probability, MinHash similarity, and entropy‑driven pheromone update.
"""

import sys
import random
import math
import hashlib
from pathlib import Path
from collections import Counter
from typing import Mapping, Hashable, Set, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]

# ----------------------------------------------------------------------
# Pheromone / perceptual hashing utilities (from Parent A)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Perceptual hash: 1 bit per value indicating >= average."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:  # limit to 64 bits
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()

def broadcast_probability(phases: int, phase: int) -> float:
    """p = 1 / 2^(phases‑phase), clamped to [0,1]."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))

# ----------------------------------------------------------------------
# MinHash utilities (from Parent B)
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1
DEFAULT_K = 128
_EPS = 1e-12

def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash for (seed, token)."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: List[str], k: int = DEFAULT_K) -> List[int]:
    """MinHash signature of length k for a token collection."""
    if k <= 0:
        raise ValueError("k must be positive")
    token_set = {t for t in tokens if t}
    if not token_set:
        return [MAX64] * k
    return [min(_hash(i, t) for t in token_set) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Approximate Jaccard similarity via equal components."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def _normalize(probs: List[float]) -> List[float]:
    total = sum(probs)
    if total <= 0:
        raise ValueError("probability mass must be positive")
    return [p / total for p in probs]

def entropy_from_counts(counts: List[int]) -> float:
    """Shannon entropy of a discrete distribution given raw counts."""
    if not counts:
        raise ValueError("counts must not be empty")
    probs = _normalize([float(c) for c in counts])
    return -sum(p * math.log(max(p, _EPS)) for p in probs)

def signature_entropy(sig: List[int]) -> float:
    """Entropy of a MinHash signature interpreted as a bit‑frequency distribution."""
    # Convert each 64‑bit integer to its bit count (0‑64)
    bit_counts = [bin(x).count("1") for x in sig]
    # Histogram over possible counts (0‑64)
    hist = [0] * 65
    for c in bit_counts:
        hist[c] += 1
    return entropy_from_counts(hist)

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def node_neighbour_phash(node: Node, graph: Graph, values: Dict[Node, float]) -> int:
    """Compute a perceptual hash of a node together with its neighbours' values."""
    neighbourhood = [values.get(node, 0.0)]
    neighbourhood.extend(values.get(n, 0.0) for n in graph.get(node, []))
    return compute_phash(neighbourhood)

def node_signature(node: Node, graph: Graph, values: Dict[Node, float],
                  k: int = DEFAULT_K) -> List[int]:
    """
    Produce a MinHash signature for a node.
    Tokens are the binary representation of the node's neighbourhood perceptual hash.
    """
    ph = node_neighbour_phash(node, graph, values)
    # Turn the 64‑bit hash into a list of '0'/'1' tokens
    token_str = f"{ph:064b}"
    tokens = list(token_str)  # each token is a character '0' or '1'
    return signature(tokens, k)

def hybrid_maximal_independent_set(
    graph: Graph,
    values: Dict[Node, float],
    phases: int = 8,
    k: int = DEFAULT_K,
    similarity_thresh: float = 0.7,
    seed: int | str | None = None,
) -> Set[Node]:
    """
    Hybrid leader election:
    - Broadcast probability follows Parent A.
    - A broadcasting node becomes a leader only if:
        * No neighbour is also broadcasting (independence).
        * Its MinHash signature is not too similar to any other broadcasting node
          (similarity < threshold).
    - After leader selection, pheromone values of leaders are boosted by
      `1 + entropy(signature)`.
    Returns the set of leader nodes.
    """
    rng = random.Random(seed)
    undecided: Set[Node] = set(graph)
    leaders: Set[Node] = set()
    # Pre‑compute signatures for all nodes (cost O(|V|·k))
    sigs: Dict[Node, List[int]] = {
        n: node_signature(n, graph, values, k) for n in graph
    }

    for phase in range(1, phases + 1):
        if not undecided:
            break
        p = broadcast_probability(phases, phase)
        broadcasts = {n for n in undecided if rng.random() < p}
        if not broadcasts:
            continue

        # Determine which broadcasts are too similar to each other
        too_similar: Set[Node] = set()
        broadcast_list = list(broadcasts)
        for i, a in enumerate(broadcast_list):
            for b in broadcast_list[i + 1 :]:
                if similarity(sigs[a], sigs[b]) >= similarity_thresh:
                    too_similar.add(a)
                    too_similar.add(b)

        # Independent set condition + similarity filter
        new_leaders = {
            n
            for n in broadcasts
            if n not in too_similar
            and not (graph.get(n, set()) & broadcasts)  # no neighbour also broadcasting
        }

        leaders.update(new_leaders)
        # Remove chosen leaders and their neighbours from further consideration
        removed = set(new_leaders)
        for n in new_leaders:
            removed.update(graph.get(n, set()))
        undecided.difference_update(removed)

    # Pheromone update based on signature entropy
    for leader in leaders:
        ent = signature_entropy(sigs[leader])
        values[leader] = values.get(leader, 0.0) * (1.0 + ent)

    return leaders

# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny random graph with 10 nodes
    rng = random.Random(42)
    nodes = list(range(10))
    graph: Dict[int, Set[int]] = {n: set() for n in nodes}
    for n in nodes:
        # each node connects to up to 3 random other nodes
        neighbours = rng.sample([m for m in nodes if m != n], k=rng.randint(0, 3))
        graph[n].update(neighbours)
        for m in neighbours:
            graph[m].add(n)  # ensure undirected

    # Random initial pheromone values
    values = {n: rng.random() for n in nodes}

    leaders = hybrid_maximal_independent_set(
        graph,
        values,
        phases=6,
        k=64,
        similarity_thresh=0.6,
        seed=123,
    )
    print("Leaders selected:", sorted(leaders))
    print("Updated pheromone values for leaders:")
    for l in sorted(leaders):
        print(f"  Node {l}: {values[l]:.4f}")