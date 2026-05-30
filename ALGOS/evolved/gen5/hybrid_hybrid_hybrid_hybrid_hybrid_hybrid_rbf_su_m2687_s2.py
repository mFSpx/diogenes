# DARWIN HAMMER — match 2687, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m296_s0.py (gen4)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hybrid_fisher_m317_s0.py (gen3)
# born: 2026-05-29T23:44:53Z

"""Hybrid Algorithm: Fusion of Hybrid Workshare Allocator with Liquid Time Constant &
Geometric Product (Parent A) and Gaussian‑Based Similarity / Fisher Scoring (Parent B).

Mathematical Bridge:
- Parent B produces a similarity matrix S(i,j) between feature vectors using hash‑based
  Hamming similarity and a Gaussian kernel on the Hamming distance.
- Parent A represents weight matrices as multivectors in a Clifford algebra.
- In this hybrid we embed the similarity values S(i,j) as grade‑1 components of a
  multivector w_i = Σ_j S(i,j) e_j.  The multivector inner product ⟨w_i·w_j⟩ yields a
  similarity‑aware scalar that can be used in the Hoeffding bound of a streaming
  decision tree.  Gaussian uncertainty from Parent B modulates the multivector
  components, while the Hoeffding bound (Parent A) decides when a split is
  statistically significant.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple, List, Set

# ----------------------------------------------------------------------
# Multivector implementation (simplified Clifford algebra Cl(n,0))
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return sorted blade and sign after bubble‑sorting, removing duplicate indices."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)
        elif lst[i] == lst[i + 1]:
            # duplicate index cancels (e_i ∧ e_i = 0)
            lst.pop(i)
            lst.pop(i)  # second element now at same position
            n -= 2
            i = max(i - 1, 0)
        else:
            i += 1
    return lst, sign

def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    """Sparse representation of a multivector in Cl(n,0)."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        self.n = int(n)
        # store only non‑zero components
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-12}

    def __add__(self, other: "Multivector") -> "Multivector":
        assert self.n == other.n
        new = self.components.copy()
        for k, v in other.components.items():
            new[k] = new.get(k, 0.0) + v
            if abs(new[k]) < 1e-12:
                del new[k]
        return Multivector(new, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({k: -v for k, v in self.components.items()}, self.n)

    def __mul__(self, other):
        """Geometric product with another multivector or scalar."""
        if isinstance(other, (int, float)):
            return Multivector({k: v * other for k, v in self.components.items()}, self.n)
        assert isinstance(other, Multivector) and self.n == other.n
        result: Dict[frozenset, float] = {}
        for a_blade, a_val in self.components.items():
            for b_blade, b_val in other.components.items():
                blade, sign = _multiply_blades(a_blade, b_blade)
                result[blade] = result.get(blade, 0.0) + sign * a_val * b_val
        return Multivector(result, self.n)

    __rmul__ = __mul__

    def inner(self, other: "Multivector") -> float:
        """Scalar inner product ⟨A·B⟩ = sum over matching blades."""
        assert self.n == other.n
        total = 0.0
        for blade, a_val in self.components.items():
            b_val = other.components.get(blade, 0.0)
            total += a_val * b_val
        return total

    def norm(self) -> float:
        """Euclidean norm derived from inner product."""
        return math.sqrt(self.inner(self))

    def grade(self, k: int) -> "Multivector":
        """Return a multivector containing only grade‑k blades."""
        new = {b: v for b, v in self.components.items() if len(b) == k}
        return Multivector(new, self.n)

    def __repr__(self):
        return f"Multivector({self.components}, n={self.n})"

# ----------------------------------------------------------------------
# Parent B utilities – similarity & Gaussian modelling
# ----------------------------------------------------------------------
Node = int
FeatureVec = Tuple[float, ...]
Graph = Dict[Node, Set[Node]]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel with bandwidth epsilon."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: 1 bit per value compared to mean."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def similarity_matrix(features: Dict[Node, FeatureVec]) -> Tuple[np.ndarray, List[Node]]:
    """Hash‑based similarity (1 – normalized Hamming distance)."""
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = compute_phash(list(features[ni]))
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                hj = compute_phash(list(features[nj]))
                d = hamming_distance(hi, hj)
                S[i, j] = 1.0 - d / 64.0
    return S, nodes

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width ** 2))
    return derivative / (intensity + eps)

# ----------------------------------------------------------------------
# Hybrid core – embedding similarity into multivectors and Hoeffding bound
# ----------------------------------------------------------------------
def build_multivector_weights(features: Dict[Node, FeatureVec],
                              epsilon: float = 1.0) -> Dict[Node, Multivector]:
    """
    For each node i, construct a grade‑1 multivector:
        w_i = Σ_j  w_ij * e_j
    where w_ij = Gaussian( HammingSimilarity(i,j) ) .
    The Gaussian kernel (Parent B) injects uncertainty into the Clifford‑algebra
    weight representation (Parent A).
    """
    S, nodes = similarity_matrix(features)
    n = len(nodes)
    index_map = {node: idx for idx, node in enumerate(nodes)}
    weights: Dict[Node, Multivector] = {}
    for i, node_i in enumerate(nodes):
        comps: Dict[frozenset, float] = {}
        for j, node_j in enumerate(nodes):
            sim = S[i, j]                     # similarity ∈ [0,1]
            w = gaussian(sim, epsilon)       # Gaussian‑modulated weight
            if w > 1e-12 and i != j:
                blade = frozenset({j})        # basis vector e_j
                comps[blade] = w
        weights[node_i] = Multivector(comps, n)
    return weights

def hoeffding_bound(range_r: float, delta: float, n_samples: int) -> float:
    """
    Classic Hoeffding bound:
        ε = sqrt( (R^2 * ln(1/δ)) / (2 * n) )
    """
    if n_samples <= 0:
        return float('inf')
    return math.sqrt((range_r ** 2 * math.log(1.0 / delta)) / (2.0 * n_samples))

def gini_impurity(class_counts: Counter) -> float:
    """Standard Gini impurity for a set of class counts."""
    total = sum(class_counts.values())
    if total == 0:
        return 0.0
    prob_sq_sum = sum((c / total) ** 2 for c in class_counts.values())
    return 1.0 - prob_sq_sum

def evaluate_split(node: Node,
                   weights: Dict[Node, Multivector],
                   labels: Dict[Node, int],
                   delta: float = 1e-7) -> Tuple[bool, float]:
    """
    Decide whether to split on `node` using a Hoeffding‑test on Gini impurity.
    The impurity reduction is weighted by inner products of multivectors,
    i.e. similarity‑aware counts.
    Returns (should_split, bound).
    """
    # Gather weighted class counts for left/right partitions based on sign of inner product
    w_i = weights[node]
    left_counts = Counter()
    right_counts = Counter()
    for other, w_j in weights.items():
        if other == node:
            continue
        similarity = w_i.inner(w_j)          # scalar similarity from multivectors
        # Use Gaussian of similarity as probability of belonging to left partition
        prob_left = gaussian(similarity, epsilon=0.5)
        if random.random() < prob_left:
            left_counts[labels[other]] += 1
        else:
            right_counts[labels[other]] += 1

    # Compute impurity before and after split
    parent_counts = left_counts + right_counts
    impurity_parent = gini_impurity(parent_counts)
    impurity_left = gini_impurity(left_counts)
    impurity_right = gini_impurity(right_counts)
    weighted_impurity = (sum(left_counts.values()) * impurity_left +
                         sum(right_counts.values()) * impurity_right) / max(1, sum(parent_counts.values()))
    impurity_gain = impurity_parent - weighted_impurity

    # Hoeffding bound on gain (range of Gini is [0,0.5])
    bound = hoeffding_bound(range_r=0.5, delta=delta, n_samples=sum(parent_counts.values()))
    should_split = impurity_gain > bound
    return should_split, bound

def train_hybrid_tree(features: Dict[Node, FeatureVec],
                      labels: Dict[Node, int],
                      max_splits: int = 5) -> List[Node]:
    """
    Very lightweight streaming tree trainer.
    Returns the list of nodes chosen as split points.
    """
    weights = build_multivector_weights(features, epsilon=1.0)
    split_nodes: List[Node] = []
    candidates = list(features.keys())
    random.shuffle(candidates)

    for node in candidates:
        if len(split_nodes) >= max_splits:
            break
        should_split, bound = evaluate_split(node, weights, labels)
        if should_split:
            split_nodes.append(node)
    return split_nodes

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic dataset
    random.seed(42)
    features: Dict[Node, FeatureVec] = {
        0: (0.1, 0.2),
        1: (0.9, 0.8),
        2: (0.4, 0.5),
        3: (0.6, 0.55),
        4: (0.2, 0.1),
    }
    # Random binary labels
    labels: Dict[Node, int] = {node: random.randint(0, 1) for node in features}

    # Build multivector weights and display one example
    mv_weights = build_multivector_weights(features, epsilon=0.8)
    print("Example multivector weight for node 0:", mv_weights[0])

    # Attempt a split decision on node 0
    split, eps = evaluate_split(0, mv_weights, labels)
    print(f"Node 0 split decision: {split} (Hoeffding bound ε={eps:.6f})")

    # Train a tiny hybrid tree
    splits = train_hybrid_tree(features, labels, max_splits=3)
    print("Selected split nodes:", splits)