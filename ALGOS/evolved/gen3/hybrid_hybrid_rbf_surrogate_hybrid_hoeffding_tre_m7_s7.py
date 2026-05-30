# DARWIN HAMMER — match 7, survivor 7
# gen: 3
# parent_a: hybrid_rbf_surrogate_hybrid_distributed_l_m58_s0.py (gen2)
# parent_b: hybrid_hoeffding_tree_tropical_maxplus_m18_s1.py (gen1)
# born: 2026-05-29T23:25:20Z

import math
import random
from dataclasses import dataclass
from typing import Dict, Hashable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Basic Types
# ----------------------------------------------------------------------
Node = Hashable
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------
def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def compute_phash(values: List[float]) -> int:
    """
    Simple perceptual hash: 1‑bit per value relative to the median.
    Uses up to 64 bits; remaining values are ignored.
    """
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers interpreted as bit strings."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Parent A – RBF kernel & perceptual similarity
# ----------------------------------------------------------------------
def rbf_kernel_matrix(
    features: Dict[Node, FeatureVec], epsilon: float = 1.0
) -> Tuple[np.ndarray, List[Node]]:
    """
    Dense RBF kernel K where K[i, j] = exp(-ε² * ||f_i - f_j||²).
    Returns the matrix and the node ordering.
    """
    nodes = list(features.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        K[i, i] = 1.0  # distance zero → kernel 1
        for j in range(i + 1, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K, nodes


def perceptual_similarity_matrix(
    features: Dict[Node, FeatureVec]
) -> Tuple[np.ndarray, List[Node]]:
    """
    Perceptual similarity matrix S where
        S[i, j] = 1 - Hamming(hash_i, hash_j) / 64 .
    Returns the matrix and the node ordering.
    """
    nodes = list(features.keys())
    n = len(nodes)
    hashes = [compute_phash(list(features[n])) for n in nodes]
    S = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        S[i, i] = 1.0
        for j in range(i + 1, n):
            d = hamming_distance(hashes[i], hashes[j])
            sim = 1.0 - d / 64.0
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes


# ----------------------------------------------------------------------
# Tropical Algebra (max‑plus) utilities
# ----------------------------------------------------------------------
def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical (max‑plus) matrix product:
        (A ⊗ B)_{ij} = max_k ( A_{ik} + B_{kj} )
    Operates on dense float matrices.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.shape[1] != B.shape[0]:
        raise ValueError("Inner dimensions must agree for tropical multiplication")

    # Compute all pairwise sums A[i,k] + B[k,j] and take max over k.
    # Shape after broadcasting: (i, k, j)
    sums = A[:, :, np.newaxis] + B[np.newaxis, :, :]
    return np.max(sums, axis=1)


# ----------------------------------------------------------------------
# Hoeffding bound utilities
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """
    Hoeffding bound for a range r, confidence 1‑δ, after n independent samples.
    """
    if r <= 0:
        raise ValueError("range r must be positive")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    if n <= 0:
        raise ValueError("sample count n must be positive")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split(
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> SplitDecision:
    """
    Hoeffding‑based decision whether a candidate split is statistically justified.
    """
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    if gap > eps:
        reason = "gap_exceeds_bound"
    elif eps < tie_threshold:
        reason = "tight_bound"
    else:
        reason = "insufficient_gap"
    return SplitDecision(split, eps, gap, reason)


# ----------------------------------------------------------------------
# Core Fusion – deeper integration via log‑domain tropical product
# ----------------------------------------------------------------------
def log_safe(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Element‑wise log with a small epsilon to avoid log(0)."""
    return np.log(np.clip(M, eps, None))


def exp_safe(L: np.ndarray) -> np.ndarray:
    """Exponentiate a log‑domain matrix and clip to [0,1]."""
    return np.clip(np.exp(L), 0.0, 1.0)


def combined_similarity(
    features: Dict[Node, FeatureVec],
    epsilon: float = 1.0,
    log_eps: float = 1e-12,
) -> Tuple[np.ndarray, List[Node]]:
    """
    Compute a mathematically consistent fusion of the RBF kernel K and the
    perceptual similarity S using a log‑domain tropical product:

        L = log(K) ⊗ log(S)          (max‑plus on logs)
        C = exp(L)                   (back to similarity space)

    Returns the fused matrix C and the node ordering.
    """
    K, nodes_k = rbf_kernel_matrix(features, epsilon)
    S, nodes_s = perceptual_similarity_matrix(features)

    if nodes_k != nodes_s:
        raise ValueError("Node ordering mismatch between K and S")

    # Log‑domain conversion (values are in (0,1], safe for log)
    logK = log_safe(K, log_eps)
    logS = log_safe(S, log_eps)

    # Tropical product in log space
    logC = t_matmul(logK, logS)

    # Back to similarity space
    C = exp_safe(logC)
    return C, nodes_k


def node_gain_vector(combined: np.ndarray) -> np.ndarray:
    """
    Gain for each node = maximum fused similarity to any other node.
    """
    return np.max(combined, axis=1)


def hybrid_split_decision(
    gains: np.ndarray,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> SplitDecision:
    """
    Apply Hoeffding bound on the ordered gains to decide whether the best
    candidate should be split. The range r is derived from the observed gains.
    """
    if gains.size < 2:
        return SplitDecision(True, 0.0, 0.0, "single_candidate")

    sorted_idx = np.argsort(gains)[::-1]  # descending order
    best = gains[sorted_idx[0]]
    second = gains[sorted_idx[1]]

    r = float(gains.max() - gains.min()) or 1e-12  # avoid zero range
    return should_split(best, second, r, delta, n, tie_threshold)


def modulated_probability(
    raw_p: float,
    node_idx: int,
    undecided_mask: np.ndarray,
    combined: np.ndarray,
) -> float:
    """
    Modulate a raw broadcast probability by the average fused similarity of the
    node to the currently undecided nodes. Result is clipped to [0,1].
    """
    if not (0.0 <= raw_p <= 1.0):
        raise ValueError("raw_p must be in [0,1]")

    # Row i contains similarity of node i to all others
    row = combined[node_idx]

    # Consider only undecided neighbours
    relevant = row[undecided_mask]
    if relevant.size == 0:
        modulation = 1.0
    else:
        modulation = np.mean(relevant)  # already in [0,1]

    return float(np.clip(raw_p * modulation, 0.0, 1.0))


# ----------------------------------------------------------------------
# High‑level class encapsulating the hybrid algorithm
# ----------------------------------------------------------------------
class HybridRBF_Tropical:
    """
    End‑to‑end implementation of the hybrid RBF–Tropical Hoeffding algorithm.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 0.05,
        tie_threshold: float = 0.05,
        log_eps: float = 1e-12,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.tie_threshold = tie_threshold
        self.log_eps = log_eps
        self._features: Dict[Node, FeatureVec] = {}
        self._combined: np.ndarray | None = None
        self._nodes: List[Node] = []

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------
    def add_node(self, node: Node, feature: FeatureVec) -> None:
        """Insert or update a node with its feature vector."""
        self._features[node] = list(feature)  # store as list for hashing safety

    def remove_node(self, node: Node) -> None:
        """Remove a node from the internal structures."""
        self._features.pop(node, None)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def _recompute_fusion(self) -> None:
        """Re‑compute the fused similarity matrix after any topology change."""
        if not self._features:
            self._combined = None
            self._nodes = []
            return
        self._combined, self._nodes = combined_similarity(
            self._features, epsilon=self.epsilon, log_eps=self.log_eps
        )

    def get_fused_matrix(self) -> Tuple[np.ndarray, List[Node]]:
        """Public accessor for the fused similarity matrix."""
        if self._combined is None:
            self._recompute_fusion()
        return self._combined, self._nodes

    def compute_gains(self) -> np.ndarray:
        """Return the gain vector for the current fused matrix."""
        C, _ = self.get_fused_matrix()
        return node_gain_vector(C)

    # ------------------------------------------------------------------
    # Decision making
    # ------------------------------------------------------------------
    def decide_split(self, sample_count: int) -> SplitDecision:
        """
        Decide whether the current best candidate should be split,
        given the number of observed samples (n).
        """
        gains = self.compute_gains()
        return hybrid_split_decision(
            gains,
            delta=self.delta,
            n=sample_count,
            tie_threshold=self.tie_threshold,
        )

    def broadcast_probability(
        self,
        raw_p: float,
        node: Node,
        undecided_nodes: Set[Node],
    ) -> float:
        """
        Compute a probability for node `node` to broadcast its state,
        modulated by its fused similarity to the set of undecided nodes.
        """
        if node not in self._features:
            raise KeyError("node not present in the current feature set")
        C, nodes = self.get_fused_matrix()
        idx = nodes.index(node)
        mask = np.array([n in undecided_nodes for n in nodes], dtype=bool)
        return modulated_probability(raw_p, idx, mask, C)

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------
    def top_k_nodes(self, k: int) -> List[Tuple[Node, float]]:
        """
        Return the k nodes with highest gain values, as (node, gain) pairs.
        """
        gains = self.compute_gains()
        if k <= 0:
            return []
        top_idx = np.argpartition(-gains, range(min(k, gains.size)))[:k]
        sorted_top = top_idx[np.argsort(-gains[top_idx])]
        return [(self._nodes[i], float(gains[i])) for i in sorted_top]

    def reset(self) -> None:
        """Clear all stored data."""
        self._features.clear()
        self._combined = None
        self._nodes = []