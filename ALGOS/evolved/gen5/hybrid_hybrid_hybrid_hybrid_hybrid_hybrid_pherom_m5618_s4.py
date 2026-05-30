# DARWIN HAMMER — match 5618, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_ssim_hybrid_h_m1322_s3.py (gen4)
# parent_b: hybrid_hybrid_pheromone_inf_hybrid_hybrid_sheaf__m42_s0.py (gen3)
# born: 2026-05-30T00:03:30Z

"""Hybrid Algorithm Fusion of:
- Parent A: pheromone‑based maximal independent set with perceptual hashing, MinHash,
  SSIM similarity and fractional power hyper‑vector transform.
- Parent B: pheromone system with entropy‑based pruning, Hoeffding bound and sheaf‑cohomology
  matrix operations.

Mathematical Bridge
------------------
1. For each node we build a *perceptual hash* (bit‑vector) and a *random hyper‑vector*.
2. Pairwise SSIM‑like similarity between nodes yields a symmetric matrix **S**.
   **S** is interpreted as the *sheaf coboundary matrix* that propagates
   hyper‑vectors across the graph (sheaf‑cohomology view).
3. Each hyper‑vector **vᵢ** is transformed by a fractional power whose exponent
   αᵢⱼ = Sᵢⱼ (the similarity to neighbour j):
       vᵢ′ = Πⱼ (vⱼ)^{αᵢⱼ}   (implemented as element‑wise power then product).
   This couples the continuous SSIM similarity with the high‑dimensional binding
   space of the sheaf.
4. The transformed vector is tokenised (indices of positive components) and merged
   with the 1‑bits of the perceptual hash; a MinHash signature approximates the
   Jaccard similarity of these token sets.
5. Leader election follows a pheromone‑broadcast maximal‑independent‑set rule.
   A node may become a leader only if its MinHash signature is *sufficiently
   dissimilar* (low Jaccard) from already chosen leaders and if its pruning
   probability  p = exp(‑entropy) exceeds a random threshold.
6. After selection the node’s pheromone τ is updated by
       τ ← τ × (H(sig) * ε_Hoeffding)
   where H(sig) is the Shannon entropy of the MinHash signature and
   ε_Hoeffding is a Hoeffding bound computed from the average SSIM to neighbours.
"""

import sys
import math
import random
import hashlib
from pathlib import Path
from collections import Counter
from typing import Mapping, Hashable, Set, List, Dict, Tuple, Optional

import numpy as np
import datetime

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
ValueMap = Mapping[Node, List[float]]

# ----------------------------------------------------------------------
# Helper utilities (shared by both parents)
# ----------------------------------------------------------------------
def shannon_entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Entropy H = - Σ p·log(p) for a probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("Positive probability mass required")
    probs = [max(p / total, eps) for p in probabilities]
    return -sum(p * math.log(p) for p in probs)

def hoeffding_bound(success_prob: float, n: int, delta: float = 0.05) -> float:
    """Hoeffding bound ε = sqrt( ln(2/δ) / (2n) ) for Bernoulli trials."""
    if n <= 0:
        return 0.0
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))

def perceptual_hash(values: List[float], bits: int = 64) -> int:
    """Simple perceptual hash: hash of concatenated rounded values."""
    data = ''.join(f"{round(v,4):.4f}" for v in values).encode()
    digest = hashlib.sha256(data).digest()
    # Take first `bits` bits
    int_val = int.from_bytes(digest[:bits // 8], byteorder='big')
    return int_val

def random_hypervector(dim: int = 256) -> np.ndarray:
    """Random bipolar hyper‑vector (+1 / -1)."""
    return np.where(np.random.rand(dim) > 0.5, 1.0, -1.0)

def minhash_signature(tokens: Set[int], num_perm: int = 64) -> Tuple[int, ...]:
    """Compute a MinHash signature using simple universal hash functions."""
    # deterministic random seeds for reproducibility
    max_hash = (1 << 32) - 1
    signature = []
    for i in range(num_perm):
        a = random.Random(i).randint(1, max_hash)
        b = random.Random(i + 1000).randint(0, max_hash)
        min_val = max_hash
        for t in tokens:
            hv = (a * t + b) % max_hash
            if hv < min_val:
                min_val = hv
        signature.append(min_val)
    return tuple(signature)

def jaccard_distance(sig1: Tuple[int, ...], sig2: Tuple[int, ...]) -> float:
    """Approximate Jaccard distance between two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("Signatures must have same length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return 1.0 - matches / len(sig1)

# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def node_feature_vectors(
    graph: Graph,
    values: ValueMap,
    hash_bits: int = 64,
    hyper_dim: int = 256
) -> Dict[Node, Tuple[int, np.ndarray]]:
    """
    For each node produce:
        - perceptual hash (int)
        - random hyper‑vector (np.ndarray)
    """
    feats = {}
    for node, neigh in graph.items():
        # aggregate neighbour values for perceptual hash
        agg_vals = values.get(node, [])
        for nb in neigh:
            agg_vals += values.get(nb, [])
        phash = perceptual_hash(agg_vals, bits=hash_bits)
        hv = random_hypervector(dim=hyper_dim)
        feats[node] = (phash, hv)
    return feats

def ssim_like_similarity(vec_a: np.ndarray, vec_b: np.ndarray, C: float = 1e-6) -> float:
    """A simplified SSIM between two bipolar vectors."""
    mu_a = np.mean(vec_a)
    mu_b = np.mean(vec_b)
    sigma_a = np.var(vec_a)
    sigma_b = np.var(vec_b)
    sigma_ab = np.mean((vec_a - mu_a) * (vec_b - mu_b))
    numerator = (2 * mu_a * mu_b + C) * (2 * sigma_ab + C)
    denominator = (mu_a**2 + mu_b**2 + C) * (sigma_a + sigma_b + C)
    return float(numerator / denominator)

def build_similarity_matrix(
    feats: Dict[Node, Tuple[int, np.ndarray]]
) -> Tuple[np.ndarray, List[Node]]:
    """
    Returns a symmetric matrix S where S[i,j] = SSIM similarity between node i and j.
    Also returns the ordered list of nodes matching matrix rows/cols.
    """
    nodes = list(feats.keys())
    n = len(nodes)
    S = np.zeros((n, n), dtype=float)
    for i in range(n):
        _, hv_i = feats[nodes[i]]
        for j in range(i, n):
            _, hv_j = feats[nodes[j]]
            sim = ssim_like_similarity(hv_i, hv_j)
            S[i, j] = S[j, i] = sim
    return S, nodes

def fractional_sheaf_transform(
    S: np.ndarray,
    hv_dict: Dict[Node, np.ndarray],
    node_order: List[Node]
) -> Dict[Node, np.ndarray]:
    """
    Implements the sheaf‑cohomology inspired propagation:
    v_i' = Π_j (v_j)^{S_ij}
    where exponentiation is element‑wise fractional power.
    """
    transformed = {}
    dim = next(iter(hv_dict.values())).shape[0]
    for idx, node in enumerate(node_order):
        # start with ones vector (log‑space additive)
        log_vec = np.zeros(dim, dtype=float)
        for jdx, other in enumerate(node_order):
            alpha = S[idx, jdx]
            if alpha == 0:
                continue
            # fractional power via exp(alpha * log(|v|)) preserving sign
            v = hv_dict[other]
            sign = np.sign(v)
            magnitude = np.abs(v)
            # avoid log(0)
            magnitude = np.where(magnitude == 0, 1e-12, magnitude)
            log_vec += alpha * np.log(magnitude)
            # accumulate sign separately (odd/even powers)
            sign = sign ** alpha  # works because sign is ±1
        # reconstruct vector
        transformed_vec = sign * np.exp(log_vec)
        transformed[node] = transformed_vec
    return transformed

def hybrid_minhash_signature(
    phash: int,
    transformed_vec: np.ndarray,
    num_perm: int = 64
) -> Tuple[int, ...]:
    """
    Combine perceptual hash bits with positive indices of the transformed vector,
    then compute a MinHash signature.
    """
    tokens = {i for i, val in enumerate(transformed_vec) if val > 0}
    # add hash bit positions as tokens
    for bit in range(phash.bit_length()):
        if (phash >> bit) & 1:
            tokens.add(bit + 1000)  # offset to avoid collision with vector indices
    return minhash_signature(tokens, num_perm=num_perm)

def prune_candidates(
    signatures: Dict[Node, Tuple[int, ...]],
    entropy_thresh: float = 0.5
) -> Set[Node]:
    """
    Remove nodes whose signature entropy is higher than `entropy_thresh`
    (i.e., more uncertain). Entropy is computed on the distribution of hash
    values inside the signature.
    """
    kept = set()
    for node, sig in signatures.items():
        probs = Counter(sig).values()
        ent = shannon_entropy(list(probs))
        # normalise by log(num_perm) to obtain value in [0,1]
        norm_ent = ent / math.log(len(sig))
        if norm_ent <= entropy_thresh:
            kept.add(node)
    return kept

def hybrid_maximal_independent_set(
    graph: Graph,
    signatures: Dict[Node, Tuple[int, ...]],
    pheromones: Dict[Node, float],
    S: np.ndarray,
    node_order: List[Node],
    jaccard_thresh: float = 0.3
) -> Set[Node]:
    """
    Greedy maximal independent set guided by pheromone strength.
    A node can be added if:
        - its pheromone is above the mean pheromone,
        - its MinHash Jaccard distance to every already selected node exceeds `jaccard_thresh`,
        - it survived pruning (handled before calling this function).
    After selection, pheromone is updated with entropy × Hoeffding bound.
    """
    selected: Set[Node] = set()
    mean_pheromone = np.mean(list(pheromones.values())) if pheromones else 0.0

    # sort nodes by decreasing pheromone to bias selection
    sorted_nodes = sorted(graph.keys(), key=lambda n: pheromones.get(n, 0.0), reverse=True)

    for node in sorted_nodes:
        if pheromones.get(node, 0.0) < mean_pheromone:
            continue
        # check Jaccard distance to already selected leaders
        conflict = False
        for leader in selected:
            dist = jaccard_distance(signatures[node], signatures[leader])
            if dist < jaccard_thresh:
                conflict = True
                break
        if conflict:
            continue
        # accept node
        selected.add(node)

        # pheromone update
        idx = node_order.index(node)
        avg_ssim = np.mean(S[idx, :])  # average similarity to all nodes
        n_trials = len(graph[node]) if graph[node] else 1
        epsilon = hoeffding_bound(avg_ssim, n_trials)
        sig_entropy = shannon_entropy(list(Counter(signatures[node]).values()))
        pheromones[node] = pheromones.get(node, 1.0) * (sig_entropy * epsilon)

    return selected

# ----------------------------------------------------------------------
# End‑to‑end hybrid pipeline
# ----------------------------------------------------------------------
def run_hybrid_algorithm(
    graph: Graph,
    values: ValueMap,
    hash_bits: int = 64,
    hyper_dim: int = 256,
    num_perm: int = 64,
    jaccard_thresh: float = 0.3,
    entropy_thresh: float = 0.5
) -> Set[Node]:
    """
    Executes the full hybrid workflow and returns the selected leader set.
    """
    # 1. Feature extraction
    feats = node_feature_vectors(graph, values, hash_bits=hash_bits, hyper_dim=hyper_dim)

    # 2. Similarity matrix (SSIM) – also serves as sheaf coboundary
    S, node_order = build_similarity_matrix(feats)

    # 3. Sheaf‑inspired fractional transform of hyper‑vectors
    hv_dict = {n: hv for n, (_, hv) in feats.items()}
    transformed = fractional_sheaf_transform(S, hv_dict, node_order)

    # 4. Hybrid MinHash signatures
    signatures = {}
    for node in graph:
        phash, _ = feats[node]
        signatures[node] = hybrid_minhash_signature(phash, transformed[node], num_perm=num_perm)

    # 5. Prune by signature entropy
    candidate_nodes = prune_candidates(signatures, entropy_thresh=entropy_thresh)

    # 6. Initialise pheromones (uniform)
    pheromones = {n: 1.0 for n in graph}

    # 7. Maximal independent set selection
    selected = hybrid_maximal_independent_set(
        graph={n: graph[n] & candidate_nodes for n in candidate_nodes},
        signatures={n: signatures[n] for n in candidate_nodes},
        pheromones=pheromones,
        S=S,
        node_order=node_order,
        jaccard_thresh=jaccard_thresh,
    )
    return selected

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small random graph
    random.seed(42)
    np.random.seed(42)

    N = 10
    nodes = [f"n{i}" for i in range(N)]
    graph: Dict[str, Set[str]] = {n: set() for n in nodes}
    for i in range(N):
        for j in range(i + 1, N):
            if random.random() < 0.3:  # sparse connectivity
                graph[nodes[i]].add(nodes[j])
                graph[nodes[j]].add(nodes[i])

    # Random float values per node (simulating feature vectors)
    values: Dict[str, List[float]] = {n: list(np.random.rand(5)) for n in nodes}

    leaders = run_hybrid_algorithm(
        graph=graph,
        values=values,
        hash_bits=64,
        hyper_dim=256,
        num_perm=64,
        jaccard_thresh=0.25,
        entropy_thresh=0.6,
    )
    print("Selected leader nodes:", leaders)