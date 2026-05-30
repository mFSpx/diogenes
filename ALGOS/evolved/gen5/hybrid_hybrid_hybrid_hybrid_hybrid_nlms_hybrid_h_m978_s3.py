# DARWIN HAMMER — match 978, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_model_vram_sc_m61_s0.py (gen4)
# parent_b: hybrid_nlms_hybrid_hybrid_rbf_su_m223_s2.py (gen4)
# born: 2026-05-29T23:32:08Z

import numpy as np
import math
import random
from collections import defaultdict
from typing import Hashable, Sequence, List, Dict, Set, Tuple, Any

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Node = Hashable
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Radial basis function with a tunable bandwidth."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    """Simple locality‑sensitive hash based on median threshold."""
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for v in values[:64]:                     # limit to 64 bits for a 64‑bit int
        bits = (bits << 1) | int(v >= median)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()

# ----------------------------------------------------------------------
# Core mathematical building blocks
# ----------------------------------------------------------------------
def _epsilon_from_vram(vram_budget_mb: int, min_eps: float = 0.01, max_eps: float = 5.0) -> float:
    """
    Convert a VRAM budget (MiB) into an RBF bandwidth epsilon.
    The mapping is logarithmic to avoid extreme values for very large or very small budgets.
    """
    if vram_budget_mb <= 0:
        raise ValueError("VRAM budget must be positive")
    # Normalise to a 0‑1 range assuming typical budgets between 64 MiB and 16384 MiB
    norm = (math.log2(vram_budget_mb) - math.log2(64)) / (math.log2(16384) - math.log2(64))
    norm = max(0.0, min(1.0, norm))
    # Map to the epsilon interval [min_eps, max_eps] on a log‑scale
    return math.exp(math.log(min_eps) + norm * (math.log(max_eps) - math.log(min_eps)))

def similarity_matrix(features: Dict[Node, FeatureVec],
                      vram_budget_mb: int) -> Tuple[np.ndarray, List[Node]]:
    """
    Build a similarity matrix S where S_ij = exp(- (epsilon * d_ij)^2 )
    with epsilon derived from the available VRAM.
    """
    nodes = list(features.keys())
    n = len(nodes)
    epsilon = _epsilon_from_vram(vram_budget_mb)

    S = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        S[i, i] = 1.0
        for j in range(i + 1, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            sim = gaussian(dist, epsilon)
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes

def rbf_kernel_matrix(features: Dict[Node, FeatureVec],
                      epsilon: float = 1.0) -> Tuple[np.ndarray, List[Node]]:
    """
    Classic RBF kernel K_ij = exp(- (epsilon * d_ij)^2 ).
    """
    nodes = list(features.keys())
    n = len(nodes)

    K = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K, nodes

def hybrid_rbf_similarity(features: Dict[Node, FeatureVec],
                          vram_budget_mb: int,
                          kernel_eps: float = 1.0) -> Tuple[np.ndarray, List[Node]]:
    """
    Fuse the VRAM‑aware similarity matrix with a static RBF kernel.
    The element‑wise product emphasises pairs that are both close in feature space
    and affordable given the current memory budget.
    """
    S, nodes = similarity_matrix(features, vram_budget_mb)
    K, _ = rbf_kernel_matrix(features, epsilon=kernel_eps)
    combined = np.multiply(S, K)
    return combined, nodes

# ----------------------------------------------------------------------
# Hoeffding tree components
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Classic Hoeffding bound for a bounded random variable."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

class HoeffdingTreeNode:
    """
    Minimal Hoeffding tree node that stores sufficient statistics for a
    regression task (variance reduction) and can evaluate split candidates.
    """
    __slots__ = ("samples", "sum_y", "sum_y2", "children", "split_feature", "split_value")

    def __init__(self) -> None:
        self.samples: List[Tuple[FeatureVec, float]] = []   # (x, y)
        self.sum_y: float = 0.0
        self.sum_y2: float = 0.0
        self.children: Dict[int, "HoeffdingTreeNode"] = {}
        self.split_feature: int | None = None
        self.split_value: float | None = None

    def update(self, x: FeatureVec, y: float) -> None:
        self.samples.append((x, y))
        self.sum_y += y
        self.sum_y2 += y * y

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def variance(self) -> float:
        if self.n == 0:
            return 0.0
        mean = self.sum_y / self.n
        return (self.sum_y2 / self.n) - mean * mean

    def best_split(self, similarity: np.ndarray, node_index: int,
                   min_samples: int = 5) -> Tuple[float, float, int, float]:
        """
        Scan all features and candidate thresholds (derived from tropical
        polynomials) and return the best gain, second best gain, feature index,
        and threshold value.
        """
        if self.n < min_samples:
            return -math.inf, -math.inf, -1, math.nan

        # Convert samples to a NumPy array for vectorised operations
        X = np.array([x for x, _ in self.samples])
        y = np.array([y for _, y in self.samples])

        # Weight each sample by its similarity to the current node (row of the matrix)
        weights = similarity[node_index, :]  # shape (N_nodes,)
        # Restrict to the nodes that actually contributed samples
        # (in practice we keep the full vector; zeros have no effect)
        weighted_y = y * weights[:self.n]  # broadcast safely because weights may be longer

        # Base impurity (variance) using weighted targets
        base_impurity = np.var(weighted_y, ddof=0)

        best_gain = -math.inf
        second_gain = -math.inf
        best_feat = -1
        best_thr = math.nan

        # Tropical polynomial: for each feature we consider max‑plus linear pieces.
        # Here we approximate by evaluating at each unique value.
        for f_idx in range(X.shape[1]):
            values = np.unique(X[:, f_idx])
            if len(values) < 2:
                continue
            for thr in values[:-1]:  # last value cannot be a split point
                left_mask = X[:, f_idx] <= thr
                right_mask = ~left_mask

                if left_mask.sum() < min_samples or right_mask.sum() < min_samples:
                    continue

                left_var = np.var(weighted_y[left_mask], ddof=0)
                right_var = np.var(weighted_y[right_mask], ddof=0)

                # Weighted variance reduction (gain)
                gain = base_impurity - (
                    left_mask.sum() / self.n) * left_var - (right_mask.sum() / self.n) * right_var

                if gain > best_gain:
                    second_gain = best_gain
                    best_gain = gain
                    best_feat = f_idx
                    best_thr = thr
                elif gain > second_gain:
                    second_gain = gain

        return best_gain, second_gain, best_feat, best_thr

    def split(self, feature: int, threshold: float) -> None:
        """Create two child nodes and store split information."""
        self.split_feature = feature
        self.split_value = threshold
        self.children = {0: HoeffdingTreeNode(), 1: HoeffdingTreeNode()}

    def route(self, x: FeatureVec) -> "HoeffdingTreeNode":
        """Follow the stored split to the appropriate child."""
        if self.split_feature is None:
            return self
        direction = int(x[self.split_feature] > self.split_value)
        return self.children[direction]

# ----------------------------------------------------------------------
# Hybrid Hoeffding tree algorithm
# ----------------------------------------------------------------------
def should_split(best_gain: float, second_gain: float,
                 r: float, delta: float, n: int,
                 tie_threshold: float = 0.05) -> bool:
    """
    Decide whether to split based on Hoeffding bound and a relative‑gain tie threshold.
    """
    eps = hoeffding_bound(r, delta, n)
    gain_gap = best_gain - second_gain
    # If the observed gap exceeds the bound *and* is not a tie, split.
    return (gain_gap > eps) and (gain_gap > tie_threshold * max(abs(best_gain), abs(second_gain)))

def hybrid_hoeffding_tree(features: Dict[Node, FeatureVec],
                          targets: Dict[Node, float],
                          vram_budget_mb: int,
                          delta: float = 0.05,
                          r: float = 1.0,
                          min_samples_leaf: int = 10) -> HoeffdingTreeNode:
    """
    Build a Hoeffding tree where each node's split decision is weighted by a
    hybrid RBF similarity matrix. The tree grows incrementally as new
    (feature, target) pairs are streamed.
    """
    # Pre‑compute the similarity matrix once (static for the whole stream)
    similarity, node_order = hybrid_rbf_similarity(features, vram_budget_mb)

    # Map node identifiers to their index in the similarity matrix
    node_to_idx = {node: idx for idx, node in enumerate(node_order)}

    root = HoeffdingTreeNode()

    # Stream over the data (order is arbitrary; in practice this would be a true stream)
    for node, x in features.items():
        y = targets.get(node, 0.0)
        # Traverse the tree to find the leaf
        leaf = root
        while leaf.split_feature is not None:
            leaf = leaf.route(x)

        leaf.update(x, y)

        # Attempt a split if enough samples have been seen
        if leaf.n >= min_samples_leaf:
            idx = node_to_idx[node]
            best_gain, second_gain, feat, thr = leaf.best_split(similarity, idx, min_samples=min_samples_leaf)
            if feat != -1 and should_split(best_gain, second_gain, r, delta, leaf.n):
                leaf.split(feat, thr)

    return root

# ----------------------------------------------------------------------
# Simple sanity‑check / demonstration
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Generate synthetic data: 20 nodes, 5‑dimensional features, and a noisy linear target
    N = 20
    D = 5
    features: Dict[int, List[float]] = {
        i: [random.random() for _ in range(D)] for i in range(N)
    }
    true_w = np.random.randn(D)
    targets: Dict[int, float] = {
        i: float(np.dot(features[i], true_w) + random.gauss(0, 0.1)) for i in range(N)
    }

    vram_budget_mb = 1024  # typical desktop GPU budget

    tree = hybrid_hoeffding_tree(features, targets, vram_budget_mb)

    # Quick inspection: print the depth of the tree
    def depth(node: HoeffdingTreeNode) -> int:
        if not node.children:
            return 0
        return 1 + max(depth(child) for child in node.children.values())

    print("Tree depth:", depth(tree))
    print("Root split feature:", tree.split_feature, "threshold:", tree.split_value)