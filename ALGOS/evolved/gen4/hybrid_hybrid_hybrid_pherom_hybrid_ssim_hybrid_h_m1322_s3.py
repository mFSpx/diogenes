# DARWIN HAMMER — match 1322, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_pheromone_hyb_hybrid_infotaxis_min_m81_s1.py (gen3)
# parent_b: hybrid_ssim_hybrid_hybrid_fracti_m934_s2.py (gen3)
# born: 2026-05-29T23:35:15Z

"""Hybrid Algorithm Fusion of:
- PARENT A: pheromone‑based maximal independent set with perceptual hashing & MinHash (hybrid_hybrid_pheromone_hyb_hybrid_infotaxis_min_m81_s1.py)
- PARENT B: SSIM similarity combined with fractional‑Hoeffding bound (hybrid_ssim_hybrid_hybrid_fracti_m934_s2.py)

Mathematical Bridge
-------------------
1. Each node’s neighbourhood values are turned into a perceptual hash (A) and
   simultaneously into a random hyper‑vector (B).  
2. The structural similarity index (SSIM) between a node and each neighbour
   (B) is used as a *weight* α in a fractional‑power transform of the hyper‑vector:
   `V' = fractional_power(V, α)`.  This couples the continuous similarity measure
   with the high‑dimensional binding space.
3. The transformed hyper‑vector is tokenised (indices of positive components) and
   merged with the 1‑bits of the perceptual hash to form a token set.
   A MinHash signature (A) is built from this set, providing an efficient Jaccard
   approximation of neighbourhood similarity.
4. Leader election (maximal independent set) follows the pheromone broadcast rule
   of A, but a node may become a leader only if its MinHash signature is
   sufficiently *dissimilar* (low Jaccard) from already‑selected neighbours.
5. After selection, the node’s pheromone is updated by the product of
   – the Shannon entropy of its MinHash signature (A) and
   – a Hoeffding bound where the success probability *r* is the average SSIM
     to its neighbours (B).  
   This merges the uncertainty quantification of B with the pheromone dynamics of A.

The module implements three core hybrid operations:
`node_feature_vector`, `minhash_signature`, and `hybrid_maximal_independent_set`. """

import sys
import random
import math
import hashlib
from pathlib import Path
from collections import Counter
from typing import Mapping, Hashable, Set, List, Dict, Tuple, Optional

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
ValueMap = Mapping[Node, List[float]]

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
    """Hamming distance between two integers."""
    return bin(a ^ b).count("1")

def shannon_entropy(counter: Counter) -> float:
    """Entropy of a discrete distribution given as Counter of occurrences."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for cnt in counter.values():
        p = cnt / total
        ent -= p * math.log2(p)
    return ent

# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def ssim(x: List[float], y: List[float],
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural similarity index between two equal‑length sequences."""
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x:
        raise ValueError('samples must not be empty')
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x) / n
    vy = sum((v - my) ** 2 for v in y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))

def random_hv(d: int = 256, kind: str = "bipolar", seed: Optional[int] = None) -> np.ndarray:
    """Generate a random hyper‑vector."""
    rng = np.random.default_rng(seed)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        return v / (np.linalg.norm(v) + 1e-30)
    raise ValueError(f"Unsupported kind {kind!r}")

def fractional_power(X: np.ndarray, alpha: float) -> np.ndarray:
    """Element‑wise fractional power preserving sign."""
    return np.abs(X) ** alpha * np.sign(X)

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for Bernoulli trials."""
    if not (0 < r < 1):
        raise ValueError("r must be in (0,1)")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt(math.log(2 / delta) / (2 * n)) + r

# ----------------------------------------------------------------------
# Hybrid building blocks
# ----------------------------------------------------------------------
def node_feature_vector(node: Node,
                        graph: Graph,
                        values: ValueMap,
                        hv_dim: int = 256) -> Tuple[int, np.ndarray, float]:
    """
    Produce a tuple (phash, transformed_hv, avg_ssim) for ``node``.
    - ``phash``: perceptual hash of the node's own neighbourhood values.
    - ``transformed_hv``: hyper‑vector after fractional‑power scaling by
      the average SSIM to neighbours.
    - ``avg_ssim``: mean SSIM between the node and each neighbour (used later).
    """
    neigh_vals = []
    for nbr in graph.get(node, []):
        neigh_vals.extend(values.get(nbr, []))
    own_vals = values.get(node, [])
    # perceptual hash on the concatenated neighbourhood (including own)
    phash = compute_phash(own_vals + neigh_vals)

    # generate deterministic hyper‑vector from node identifier
    seed = int(hashlib.sha256(str(node).encode()).hexdigest(), 16) % (2 ** 32)
    hv = random_hv(d=hv_dim, kind="bipolar", seed=seed)

    # compute average SSIM to neighbours (if any)
    sims = []
    for nbr in graph.get(node, []):
        nbr_vals = values.get(nbr, [])
        if nbr_vals and own_vals:
            sims.append(ssim(own_vals, nbr_vals))
    avg_ssim = sum(sims) / len(sims) if sims else 0.0

    # fractional power with α = avg_ssim (clamped to [0,1])
    alpha = max(0.0, min(1.0, avg_ssim))
    transformed_hv = fractional_power(hv, alpha)

    return phash, transformed_hv, avg_ssim

def minhash_signature(phash: int,
                      hv: np.ndarray,
                      num_perm: int = 10,
                      max_hash: int = (1 << 31) - 1) -> List[int]:
    """
    Build a MinHash signature from the union of:
    - indices of 1‑bits in ``phash`` (0‑63)
    - indices of positive components in ``hv`` (0‑dim‑1)
    The signature is a list of ``num_perm`` minimum hash values.
    """
    token_set = set()
    # phash tokens
    for i in range(64):
        if (phash >> (63 - i)) & 1:
            token_set.add(i)
    # hv tokens (positive entries)
    for idx, val in enumerate(hv):
        if val > 0:
            token_set.add(64 + idx)  # offset to avoid collision with phash indices

    # generate random hash functions a*x + b mod prime
    rng = np.random.default_rng(42)
    a = rng.integers(1, max_hash, size=num_perm, endpoint=True, dtype=np.int64)
    b = rng.integers(0, max_hash, size=num_perm, endpoint=True, dtype=np.int64)

    signature = []
    for a_i, b_i in zip(a, b):
        min_val = max_hash
        for token in token_set:
            hv_val = (a_i * token + b_i) % max_hash
            if hv_val < min_val:
                min_val = hv_val
        signature.append(min_val)
    return signature

def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("signatures must have equal length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

def hybrid_maximal_independent_set(graph: Graph,
                                   values: ValueMap,
                                   broadcast_prob: float = 0.5,
                                   similarity_thresh: float = 0.3,
                                   delta: float = 0.05) -> Set[Node]:
    """
    Hybrid leader election:
    1. Nodes are processed in random order.
    2. A node broadcasts with probability ``broadcast_prob``.
    3. If broadcasting, it becomes a leader iff none of its already‑selected
       neighbours have a MinHash Jaccard similarity > ``similarity_thresh``.
    4. After selection, its pheromone is updated:
          pheromone *= entropy(signature) * hoeffding_bound(avg_ssim, delta, degree)
    Returns the set of selected leader nodes.
    """
    pheromone: Dict[Node, float] = {node: 1.0 for node in graph}
    signatures: Dict[Node, List[int]] = {}
    avg_ssims: Dict[Node, float] = {}
    leaders: Set[Node] = set()

    nodes = list(graph.keys())
    random.shuffle(nodes)

    for node in nodes:
        # Step 1: compute features once
        phash, hv, avg_ssim = node_feature_vector(node, graph, values)
        sig = minhash_signature(phash, hv)
        signatures[node] = sig
        avg_ssims[node] = avg_ssim

        if random.random() > broadcast_prob * pheromone[node]:
            continue  # node stays silent this round

        # Step 2: similarity check against already chosen neighbours
        conflict = False
        for nbr in graph.get(node, []):
            if nbr in leaders:
                sim = jaccard_similarity(sig, signatures[nbr])
                if sim > similarity_thresh:
                    conflict = True
                    break
        if conflict:
            continue

        # Node becomes a leader
        leaders.add(node)

        # Step 3: pheromone update using entropy and Hoeffding bound
        ent = shannon_entropy(Counter(sig))
        deg = len(graph.get(node, []))
        hoeff = hoeffding_bound(avg_ssim if avg_ssim > 0 else 0.5, delta, max(1, deg))
        pheromone[node] *= max(1e-6, ent * hoeff)  # avoid zero

    return leaders

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny random graph with 5 nodes
    random.seed(0)
    nodes = [f"N{i}" for i in range(5)]
    graph: Dict[Node, Set[Node]] = {n: set() for n in nodes}
    # add undirected edges randomly
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if random.random() < 0.4:
                graph[nodes[i]].add(nodes[j])
                graph[nodes[j]].add(nodes[i])

    # assign each node a random vector of 10 floats (0‑255)
    values: Dict[Node, List[float]] = {}
    for n in nodes:
        values[n] = [random.uniform(0, 255) for _ in range(10)]

    leaders = hybrid_maximal_independent_set(graph, values)
    print("Graph adjacency:")
    for n, nbrs in graph.items():
        print(f"  {n}: {sorted(nbrs)}")
    print("\nSelected leaders:", sorted(leaders))
    sys.exit(0)