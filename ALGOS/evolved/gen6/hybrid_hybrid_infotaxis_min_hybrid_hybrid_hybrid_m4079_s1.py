# DARWIN HAMMER — match 4079, survivor 1
# gen: 6
# parent_a: hybrid_infotaxis_minhash_m63_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_distributed_leader_e_m1525_s1.py (gen5)
# born: 2026-05-29T23:53:36Z

"""Hybrid Entropic MinHash & Pheromone‑Weighted Leader Election

Parents:
- hybrid_infotaxis_minhash_m63_s0.py (Entropic MinHash)
- hybrid_hybrid_hybrid_distributed_leader_e_m1525_s1.py (Pheromone probabilities & entropy‑based leader election)

Mathematical bridge:
Both parents manipulate probability distributions and their entropy.  
We first turn a pheromone probability vector into a MinHash signature (Parent A’s
`signature` applied to the stringified probabilities).  The signature provides a
compact, Jaccard‑like similarity measure between two pheromone distributions.
These pairwise similarities are then combined with the entropy of each
distribution (Parent B’s entropy) to produce a weighted score that drives a
distributed leader election.  Thus the core topologies – MinHash similarity
and entropy‑weighted decision making – are fused into a single unified system.
"""

import math
import hashlib
import random
import sys
import pathlib
import re
from typing import Iterable, List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities (entropy, hashing, MinHash signature)
# ----------------------------------------------------------------------
def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities)


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on Blake2b."""
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity between two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Pheromone handling (Parent B)
# ----------------------------------------------------------------------
def calculate_pheromone_probabilities(surface_key: str, limit: int, db_url: str) -> List[float]:
    """
    Placeholder implementation – in production this queries a PostgreSQL DB.
    For offline testing we return a random Dirichlet sample.
    """
    # The real implementation would use psycopg as in the original source.
    # Here we synthesize a valid probability vector.
    if limit <= 0:
        raise ValueError('limit must be positive')
    # Dirichlet(1,…,1) gives a uniform random distribution.
    raw = np.random.dirichlet([1.0] * limit).tolist()
    return raw


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def pheromone_minhash(probabilities: List[float], k: int = 128) -> List[int]:
    """
    Convert a pheromone probability vector into a MinHash signature.
    Each probability is stringified to become a token; the resulting
    signature can be compared with other distributions.
    """
    tokens = [f"{p:.12g}" for p in probabilities]  # preserve numeric information
    return signature(tokens, k)


def similarity_matrix(signatures: List[List[int]]) -> np.ndarray:
    """
    Compute the pairwise MinHash similarity matrix for a list of signatures.
    """
    n = len(signatures)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            sim = similarity(signatures[i], signatures[j])
            mat[i, j] = mat[j, i] = sim
    return mat


def leader_election(
    node_distributions: Dict[str, List[float]],
    k: int = 128,
    entropy_weight: float = 0.5
) -> Tuple[str, Dict[str, float]]:
    """
    Perform a leader election among nodes whose pheromone distributions are known.

    For each node:
      * Compute its MinHash signature.
      * Compute average similarity to all other nodes.
      * Compute normalized entropy (entropy / log(len(dist))).

    The final score is:
        score = (1 - entropy_weight) * avg_similarity + entropy_weight * (1 - norm_entropy)

    The node with the highest score is elected leader.
    Returns (leader_id, scores_dict).
    """
    if not node_distributions:
        raise ValueError('no nodes provided')

    node_ids = list(node_distributions.keys())
    signatures = [pheromone_minhash(node_distributions[nid], k) for nid in node_ids]

    # Pairwise similarities
    sim_mat = similarity_matrix(signatures)

    scores: Dict[str, float] = {}
    for idx, nid in enumerate(node_ids):
        # average similarity to others (exclude self)
        avg_sim = (sim_mat[idx].sum() - 1.0) / (len(node_ids) - 1) if len(node_ids) > 1 else 1.0

        # normalized entropy
        ent = entropy(node_distributions[nid])
        max_ent = math.log(len(node_distributions[nid])) if len(node_distributions[nid]) > 1 else 0.0
        norm_ent = ent / max_ent if max_ent > 0 else 0.0

        # combine
        score = (1 - entropy_weight) * avg_sim + entropy_weight * (1 - norm_ent)
        scores[nid] = score

    leader = max(scores, key=scores.get)
    return leader, scores


# ----------------------------------------------------------------------
# Additional illustrative function (expected entropy after a binary observation)
# ----------------------------------------------------------------------
def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """
    Expected entropy after a binary observation (hit / miss) on a distribution.
    Mirrors the truncated function from Parent A.
    """
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic pheromone distributions for three nodes
    random.seed(42)
    np.random.seed(42)

    nodes = {
        "node_A": np.random.dirichlet([0.5, 0.5, 0.5, 0.5]).tolist(),
        "node_B": np.random.dirichlet([1.0, 1.0, 1.0, 1.0]).tolist(),
        "node_C": np.random.dirichlet([2.0, 2.0, 2.0, 2.0]).tolist(),
    }

    leader, scores = leader_election(nodes, k=64, entropy_weight=0.4)

    print("Pheromone distributions:")
    for nid, dist in nodes.items():
        print(f"  {nid}: {['{:.3f}'.format(p) for p in dist]}")
    print("\nLeader election scores:")
    for nid, sc in scores.items():
        print(f"  {nid}: {sc:.4f}")
    print(f"\nElected leader: {leader}")

    # Demonstrate expected_entropy utility
    hit = [0.9, 0.05, 0.05]
    miss = [0.3, 0.4, 0.3]
    exp_ent = expected_entropy(0.7, hit, miss)
    print(f"\nExpected entropy after observation: {exp_ent:.6f}")