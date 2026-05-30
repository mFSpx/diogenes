# DARWIN HAMMER — match 4834, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1512_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hoeffd_hybrid_hybrid_hybrid_m1646_s2.py (gen6)
# born: 2026-05-29T23:58:23Z

"""Hybrid Leader Election and Adaptive Splitting via Entropy‑Weighted Simulated Annealing
and Gini‑Regularized Hoeffding Bounds.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1512_s3.py (A)
- hybrid_hybrid_hybrid_hoeffd_hybrid_hybrid_hybrid_m1646_s2.py (B)

Mathematical Bridge:
Both parents employ a *probabilistic acceptance* mechanism driven by a
statistical bound.  A uses an entropy‑weighted acceptance probability in a
simulated‑annealing schedule, while B uses a Hoeffding bound regularised by the
Gini coefficient to decide whether to split a decision node.  The fusion
creates a unified acceptance function where the annealing temperature is
inflated by the Hoeffding‑Gini bound and the raw acceptance probability is
scaled by the signature entropy.  This yields a single decision metric that
simultaneously respects structural similarity (via MinHash signatures) and
statistical confidence (via Hoeffding‑Gini regularisation)."""

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
# Parent A utilities (perceptual hashing, MinHash, entropy)
# ----------------------------------------------------------------------
def node_signature(features: List[float], num_hashes: int = 10) -> List[int]:
    """Create a simple MinHash‑style signature from a list of numeric features.

    Each hash function is simulated by hashing the concatenation of the feature
    value with a seed.  The resulting integer is reduced modulo a large prime.
    """
    signature = []
    prime = 2_147_483_647  # a large 31‑bit prime
    for seed in range(num_hashes):
        hasher = hashlib.sha256()
        for f in features:
            hasher.update(f"{f:.6f}_{seed}".encode())
        # Convert to int and take modulo prime
        sig_val = int(hasher.hexdigest(), 16) % prime
        signature.append(sig_val)
    return signature

def hamming_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Fraction of equal components between two signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("Signatures must have equal length")
    equal = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return equal / len(sig1)

def similarity_adjusted_broadcast(
    node: Node,
    neighbors: Set[Node],
    signatures: Dict[Node, List[int]],
    base_prob: float = 0.5,
) -> Dict[Node, float]:
    """Return a broadcast probability for each neighbour adjusted by signature similarity."""
    probs = {}
    sig_node = signatures[node]
    for nb in neighbors:
        sim = hamming_similarity(sig_node, signatures[nb])
        # Linear blend: more similar → higher probability
        probs[nb] = base_prob * (0.5 + 0.5 * sim)
    return probs

def signature_entropy(sig: List[int]) -> float:
    """Shannon entropy of the distribution of hash values within a signature."""
    cnt = Counter(sig)
    total = len(sig)
    ent = -sum((c / total) * math.log(c / total + 1e-12) for c in cnt.values())
    return ent

def max_possible_entropy(num_hashes: int) -> float:
    """Maximum entropy achievable with `num_hashes` distinct values."""
    if num_hashes <= 0:
        return 0.0
    p = 1.0 / num_hashes
    return -num_hashes * p * math.log(p)

# ----------------------------------------------------------------------
# Parent B utilities (Hoeffding bound with Gini regularisation, split decision)
# ----------------------------------------------------------------------
def gini_coefficient(values: List[float]) -> float:
    """Compute the Gini coefficient of a non‑negative list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(values)
    cumulative = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return gini

def hoeffding_bound_with_gini(r: float, delta: float, n: int, gini_coeff: float) -> float:
    """Hoeffding bound regularised by a Gini term (parent B)."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    regularization_term = gini_coeff * math.pi / 6
    return math.sqrt((r * r * math.log(1.0 / delta) + regularization_term) / (2.0 * n))

class SplitDecision:
    __slots__ = ("should_split", "epsilon", "gain_gap", "reason")
    def __init__(self, should_split: bool, epsilon: float, gain_gap: float, reason: str):
        self.should_split = should_split
        self.epsilon = epsilon
        self.gain_gap = gain_gap
        self.reason = reason
    def __repr__(self):
        return (f"SplitDecision(should_split={self.should_split}, epsilon={self.epsilon:.4f}, "
                f"gain_gap={self.gain_gap:.4f}, reason='{self.reason}')")

def should_split_with_gini(
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
    gini_coeff: float = 0.5,
) -> SplitDecision:
    eps = hoeffding_bound_with_gini(r, delta, n, gini_coeff)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

# ----------------------------------------------------------------------
# Hybrid core: combine entropy‑aware annealing with Hoeffding‑Gini bound
# ----------------------------------------------------------------------
def hybrid_annealing_acceptance(
    delta_energy: float,
    temperature: float,
    entropy: float,
    max_entropy: float,
    bound: float,
) -> float:
    """Entropy‑scaled acceptance probability where temperature is inflated by a Hoeffding‑Gini bound.

    The raw Metropolis probability is `exp(-ΔE / T)`.  We first increase the
    temperature by a factor `1 + bound` (bound ≥ 0) and then multiply by the
    normalized entropy term `entropy / max_entropy` to favour more diverse
    signatures.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    effective_T = temperature * (1.0 + bound)
    base_prob = math.exp(-delta_energy / effective_T)
    entropy_factor = entropy / (max_entropy + 1e-12)
    return min(1.0, base_prob * entropy_factor)

def compute_hybrid_acceptance_matrix(
    signatures: Dict[Node, List[int]],
    delta_energies: Dict[Tuple[Node, Node], float],
    temperature: float = 1.0,
    r: float = 1.0,
    delta: float = 0.05,
    n_samples: int = 30,
) -> Dict[Tuple[Node, Node], float]:
    """For every ordered pair (i, j) compute the hybrid acceptance probability.

    The Hoeffding‑Gini bound uses the distribution of signature entropies as
    the value set for Gini calculation.
    """
    # Gather entropies
    entropies = {node: signature_entropy(sig) for node, sig in signatures.items()}
    max_ent = max_possible_entropy(len(next(iter(signatures.values()))))
    entropy_vals = list(entropies.values())
    gini = gini_coefficient(entropy_vals)

    # Compute a single bound that will be shared (could be pair‑wise but simplified)
    bound = hoeffding_bound_with_gini(r, delta, n_samples, gini)

    acceptance = {}
    for (i, j), dE in delta_energies.items():
        prob = hybrid_annealing_acceptance(
            delta_energy=dE,
            temperature=temperature,
            entropy=entropies[i],
            max_entropy=max_ent,
            bound=bound,
        )
        acceptance[(i, j)] = prob
    return acceptance

def hybrid_split_decision(
    node: Node,
    signatures: Dict[Node, List[int]],
    best_gain: float,
    second_best_gain: float,
    r: float = 1.0,
    delta: float = 0.05,
    n_samples: int = 30,
    tie_threshold: float = 0.05,
    gini_coeff: float = 0.5,
) -> SplitDecision:
    """Combine entropy‑aware acceptance with Hoeffding‑Gini split logic.

    The entropy of the node's signature modulates `r` (the range of reward)
    to reflect how informative the node's feature distribution is.
    """
    entropy = signature_entropy(signatures[node])
    max_ent = max_possible_entropy(len(signatures[node]))
    # Scale r proportionally to normalized entropy (more entropy → larger effective range)
    r_eff = r * (1.0 + entropy / (max_ent + 1e-12))
    return should_split_with_gini(
        best_gain=best_gain,
        second_best_gain=second_best_gain,
        r=r_eff,
        delta=delta,
        n=n_samples,
        tie_threshold=tie_threshold,
        gini_coeff=gini_coeff,
    )

# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic graph with random features
    random.seed(42)
    nodes = [f"n{i}" for i in range(5)]
    graph: Graph = {node: set(nodes) - {node} for node in nodes}
    features: Features = {
        node: [random.random() for _ in range(8)] for node in nodes
    }

    # Compute signatures
    signatures = {node: node_signature(feat, num_hashes=12) for node, feat in features.items()}

    # Example: broadcast probabilities from node n0
    probs = similarity_adjusted_broadcast("n0", graph["n0"], signatures, base_prob=0.6)
    print("Broadcast probabilities from n0:", probs)

    # Simulate some delta energies for each ordered pair
    delta_energies = {
        (i, j): random.uniform(-1.0, 1.0) for i in nodes for j in nodes if i != j
    }

    # Compute hybrid acceptance matrix
    accept = compute_hybrid_acceptance_matrix(
        signatures,
        delta_energies,
        temperature=1.0,
        r=1.0,
        delta=0.05,
        n_samples=30,
    )
    sample_key = next(iter(accept))
    print(f"Hybrid acceptance probability for {sample_key}: {accept[sample_key]:.4f}")

    # Perform a hybrid split decision for node n2
    split_dec = hybrid_split_decision(
        node="n2",
        signatures=signatures,
        best_gain=0.12,
        second_best_gain=0.07,
        r=1.0,
        delta=0.05,
        n_samples=30,
    )
    print("Hybrid split decision for n2:", split_dec)