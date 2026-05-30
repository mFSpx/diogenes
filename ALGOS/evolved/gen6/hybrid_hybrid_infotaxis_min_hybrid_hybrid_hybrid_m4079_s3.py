# DARWIN HAMMER — match 4079, survivor 3
# gen: 6
# parent_a: hybrid_infotaxis_minhash_m63_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_distributed_leader_e_m1525_s1.py (gen5)
# born: 2026-05-29T23:53:36Z

import math
import hashlib
import random
from typing import Iterable, List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core mathematical utilities
# ----------------------------------------------------------------------


def _hash(seed: int, token: str) -> int:
    """
    Deterministic 64‑bit hash using Blake2b.
    The seed distinguishes different hash functions.
    """
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """
    Shannon entropy of a probability distribution.
    The distribution is normalised internally.
    """
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    probs = [(p / total) for p in probabilities]
    return -sum(p * math.log(max(p, eps)) for p in probs)


def normalized_entropy(probabilities: List[float]) -> float:
    """
    Entropy divided by the maximum possible entropy for the support size.
    This yields a value in [0, 1].
    """
    ent = entropy(probabilities)
    support = sum(1 for p in probabilities if p > 0)
    if support <= 1:
        return 0.0
    max_ent = math.log(support)
    return ent / max_ent


def jensen_shannon(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    """
    Symmetric Jensen‑Shannon divergence (in bits). Returns a similarity
    measure in [0, 1] where 1 means identical distributions.
    """
    if len(p) != len(q):
        raise ValueError("distributions must have equal length")
    total_p, total_q = sum(p), sum(q)
    if total_p <= 0 or total_q <= 0:
        raise ValueError("positive probability mass required")
    p_norm = [(x / total_p) for x in p]
    q_norm = [(x / total_q) for x in q]
    m = [(pi + qi) / 2 for pi, qi in zip(p_norm, q_norm)]
    def _kl(a, b):
        return sum(ai * math.log(max(ai, eps) / max(bi, eps)) for ai, bi in zip(a, b))
    js = (_kl(p_norm, m) + _kl(q_norm, m)) / 2.0
    # Convert divergence to similarity
    return 1.0 - (js / math.log(2))


# ----------------------------------------------------------------------
# Weighted MinHash (Consistent Weighted Sampling) utilities
# ----------------------------------------------------------------------


def weighted_signature(weights: List[float], k: int = 128) -> List[int]:
    """
    Consistent Weighted Sampling (CWS) signature for a non‑negative weight vector.
    For each hash function `i` we compute:
        sig[i] = min_{j | w_j > 0}  hash(i, j) / w_j
    The division makes larger weights more likely to produce smaller hash values,
    preserving the weighted Jaccard similarity.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    n = len(weights)
    if n == 0:
        return [2**64 - 1] * k

    # Pre‑compute token identifiers for each dimension
    tokens = [str(idx) for idx in range(n)]

    signature = []
    for seed in range(k):
        best = 2**64 - 1
        for idx, w in enumerate(weights):
            if w <= 0:
                continue
            h = _hash(seed, tokens[idx])
            # Use integer division with rounding up to avoid float precision issues
            candidate = h // max(w, 1e-12)
            if candidate < best:
                best = candidate
        signature.append(best)
    return signature


def weighted_jaccard(sig_a: List[int], sig_b: List[int]) -> float:
    """
    Estimate weighted Jaccard similarity from two CWS signatures.
    The estimator is the fraction of equal components.
    """
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Pheromone handling (synthetic placeholder)
# ----------------------------------------------------------------------


def calculate_pheromone_probabilities(limit: int) -> List[float]:
    """
    Return a random Dirichlet sample of dimension `limit`.
    In production this would query a real data source.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    return np.random.dirichlet([1.0] * limit).tolist()


# ----------------------------------------------------------------------
# Hybrid leader election with deeper integration
# ----------------------------------------------------------------------


def leader_election(
    node_distributions: Dict[str, List[float]],
    k: int = 128,
    weight_minhash: float = 0.4,
    weight_js: float = 0.4,
    weight_entropy: float = 0.2,
) -> Tuple[str, Dict[str, float]]:
    """
    Distributed leader election that fuses three information‑theoretic cues:

    * **Weighted MinHash similarity** – captures overlap of high‑weight pheromone
      components while being sub‑linear in dimensionality.
    * **Jensen‑Shannon similarity** – a true divergence on the original probability
      vectors.
    * **Normalized entropy** – rewards nodes whose distribution is more
      deterministic (lower entropy).

    The final score for a node `i` is

        score_i = w_m * avg_minhash_i + w_js * avg_js_i + w_e * (1 - norm_entropy_i)

    where the weights sum to 1.  The node with the highest score becomes the leader.
    """
    if not node_distributions:
        raise ValueError("no nodes provided")
    total_weight = weight_minhash + weight_js + weight_entropy
    if not math.isclose(total_weight, 1.0, rel_tol=1e-9):
        raise ValueError("weights must sum to 1")

    node_ids = list(node_distributions.keys())
    n = len(node_ids)

    # Compute weighted MinHash signatures once
    signatures = [weighted_signature(node_distributions[nid], k) for nid in node_ids]

    # Pre‑compute pairwise similarities
    minhash_sim = np.zeros((n, n), dtype=float)
    js_sim = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i, n):
            mh = weighted_jaccard(signatures[i], signatures[j])
            minhash_sim[i, j] = minhash_sim[j, i] = mh

            js = jensen_shannon(node_distributions[node_ids[i]], node_distributions[node_ids[j]])
            js_sim[i, j] = js_sim[j, i] = js

    scores: Dict[str, float] = {}
    for idx, nid in enumerate(node_ids):
        # Average similarity to all *other* nodes (self similarity excluded)
        if n > 1:
            avg_mh = (minhash_sim[idx].sum() - 1.0) / (n - 1)
            avg_js = (js_sim[idx].sum() - 1.0) / (n - 1)
        else:
            avg_mh = avg_js = 1.0

        norm_ent = normalized_entropy(node_distributions[nid])

        score = (
            weight_minhash * avg_mh
            + weight_js * avg_js
            + weight_entropy * (1.0 - norm_ent)
        )
        scores[nid] = score

    leader = max(scores, key=scores.get)
    return leader, scores


# ----------------------------------------------------------------------
# Illustrative utility (expected entropy after binary observation)
# ----------------------------------------------------------------------


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """
    Expected Shannon entropy after a binary observation with probability `p_hit`.
    """
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError("p_hit must be in [0, 1]")
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Synthetic pheromone vectors for three nodes
    nodes = {
        "node_A": calculate_pheromone_probabilities(4),
        "node_B": calculate_pheromone_probabilities(4),
        "node_C": calculate_pheromone_probabilities(4),
    }

    leader, scores = leader_election(
        nodes,
        k=64,
        weight_minhash=0.35,
        weight_js=0.45,
        weight_entropy=0.20,
    )

    print("Pheromone distributions:")
    for nid, dist in nodes.items():
        print(f"  {nid}: {[f'{p:.3f}' for p in dist]}")
    print("\nLeader election scores:")
    for nid, sc in scores.items():
        print(f"  {nid}: {sc:.4f}")
    print(f"\nElected leader: {leader}")

    # Demonstrate expected_entropy utility
    hit = [0.9, 0.05, 0.05]
    miss = [0.3, 0.4, 0.3]
    print("\nExpected entropy after observation:",
          f"{expected_entropy(0.7, hit, miss):.6f}")