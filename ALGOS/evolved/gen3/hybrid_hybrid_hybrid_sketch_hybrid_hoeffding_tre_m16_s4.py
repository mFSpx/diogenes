# DARWIN HAMMER — match 16, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s1.py (gen2)
# parent_b: hybrid_hoeffding_tree_tropical_maxplus_m18_s0.py (gen1)
# born: 2026-05-29T23:26:26Z

import numpy as np
import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable


# ----------------------------------------------------------------------
# Tropical semiring utilities
# ----------------------------------------------------------------------
class Tropical:
    """Utility class implementing max‑plus (tropical) operations."""

    @staticmethod
    def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Tropical addition = maximum."""
        return np.maximum(x, y)

    @staticmethod
    def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Tropical multiplication = ordinary addition."""
        return np.add(x, y)

    @staticmethod
    def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Tropical matrix multiplication.
        (A ⊗ B)[i, j] = max_k (A[i, k] + B[k, j])
        """
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("Both A and B must be 2‑D matrices")
        if A.shape[1] != B.shape[0]:
            raise ValueError("Inner dimensions must agree")
        # Expand dimensions to broadcast addition over k
        # A: (i, k) -> (i, k, 1); B: (k, j) -> (1, k, j)
        A_exp = A[:, :, np.newaxis]
        B_exp = B[np.newaxis, :, :]
        # Compute all pairwise sums and then max over k
        return np.max(A_exp + B_exp, axis=1)

    @staticmethod
    def polyval(coeffs: Iterable[float], x: np.ndarray) -> np.ndarray:
        """
        Evaluate a tropical polynomial:
        p(x) = max_i (coeff_i + i * x)
        """
        coeffs = np.asarray(list(coeffs), dtype=float)
        x = np.asarray(x, dtype=float)
        exponents = np.arange(coeffs.size, dtype=float)
        # Broadcast coeffs and exponents over the shape of x
        terms = coeffs[:, np.newaxis] + exponents[:, np.newaxis] * x
        return np.max(terms, axis=0)


# ----------------------------------------------------------------------
# Sketching utilities
# ----------------------------------------------------------------------
def count_min_sketch(
    items: Iterable[bytes],
    width: int = 64,
    depth: int = 4,
) -> np.ndarray:
    """
    Classic Count‑Min sketch returning a depth×width integer matrix.
    """
    table = np.zeros((depth, width), dtype=np.int64)
    for item in items:
        for d in range(depth):
            h = int(
                hashlib.sha256(f"{d}:{item}".encode()).hexdigest(),
                16,
            )
            table[d, h % width] += 1
    return table


def minhash_lsh_index(docs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Simple MinHash LSH: each document is placed in a bucket keyed by the
    smallest 6‑hex‑digit SHA‑1 hash of its shingles.
    """
    buckets: Dict[str, List[str]] = defaultdict(list)
    for doc_id, shingles in docs.items():
        if not shingles:
            key = "empty"
        else:
            key = min(
                hashlib.sha1(s.encode()).hexdigest()[:6] for s in shingles
            )
        buckets[key].append(doc_id)
    return dict(buckets)


# ----------------------------------------------------------------------
# Hoeffding bound utilities
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """
    Hoeffding bound for the difference of two empirical means.
    r: range of the random variable (max - min)
    delta: confidence parameter (0 < delta < 1)
    n: number of observations
    """
    if r <= 0:
        raise ValueError("r must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must lie in (0,1)")
    if n <= 0:
        raise ValueError("n must be positive")
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
    Decide whether a Hoeffding tree node should split.
    """
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = (gap > eps) or (eps < tie_threshold)
    if gap > eps:
        reason = "gap_exceeds_bound"
    elif eps < tie_threshold:
        reason = "tight_bound"
    else:
        reason = "await_more_data"
    return SplitDecision(split, eps, gap, reason)


# ----------------------------------------------------------------------
# Sheaf‑cohomology inspired topological regularizer
# ----------------------------------------------------------------------
def build_document_complex(docs: Dict[str, List[str]]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Build a 0‑dimensional simplicial complex (graph) where vertices are documents
    and an edge connects two documents sharing at least one shingle.
    Returns adjacency matrix and edge list.
    """
    doc_ids = list(docs.keys())
    idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    size = len(doc_ids)
    adj = np.zeros((size, size), dtype=np.int8)

    # Inverse index: shingle -> set of docs containing it
    shingle_index: Dict[str, List[int]] = defaultdict(list)
    for doc_id, shingles in docs.items():
        for s in shingles:
            shingle_index[s].append(idx[doc_id])

    edges = set()
    for doc_list in shingle_index.values():
        for i in range(len(doc_list)):
            for j in range(i + 1, len(doc_list)):
                a, b = doc_list[i], doc_list[j]
                adj[a, b] = adj[b, a] = 1
                edges.add((min(a, b), max(a, b)))

    return adj, list(edges)


def betti_0(adj: np.ndarray) -> int:
    """
    Compute the 0‑th Betti number (number of connected components)
    via a simple BFS/DFS on the adjacency matrix.
    """
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = 0

    for v in range(n):
        if not visited[v]:
            components += 1
            stack = [v]
            visited[v] = True
            while stack:
                cur = stack.pop()
                neighbors = np.where(adj[cur] > 0)[0]
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
    return components


# ----------------------------------------------------------------------
# Core hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_tropical_hoeffding(
    count_min_table: np.ndarray,
    docs: Dict[str, List[str]],
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> SplitDecision:
    """
    Fuse Count‑Min sketch, MinHash LSH, tropical max‑plus algebra,
    and a sheaf‑cohomology inspired topological regularizer to decide
    whether a Hoeffding tree node should split.
    """
    # 1️⃣ Build LSH buckets
    buckets = minhash_lsh_index(docs)

    # 2️⃣ Compute tropical scores per bucket
    tropical_scores: List[float] = []
    width = count_min_table.shape[1]

    for bucket_key, doc_ids in buckets.items():
        # Initialise tropical accumulator as the tropical zero (−∞)
        # Using a very negative sentinel because we later take max.
        bucket_score = -np.inf
        for doc_id in doc_ids:
            shingles = docs.get(doc_id, [])
            for shingle in shingles:
                h = int(hashlib.sha256(shingle.encode()).hexdigest(), 16)
                col = h % width
                # Extract the column vector across all depths
                column_vec = count_min_table[:, col].astype(float)
                # Tropical addition across depths = max
                bucket_score = Tropical.add(bucket_score, column_vec.max())
        tropical_scores.append(float(bucket_score))

    if not tropical_scores:
        raise RuntimeError("No buckets produced – cannot evaluate split decision")

    # 3️⃣ Determine best and second‑best gains safely
    sorted_scores = sorted(tropical_scores, reverse=True)
    best_gain = sorted_scores[0]
    second_best_gain = sorted_scores[1] if len(sorted_scores) > 1 else -np.inf

    # 4️⃣ Topological regularizer: penalise many connected components
    adj, _ = build_document_complex(docs)
    components = betti_0(adj)
    # Simple heuristic: subtract a small factor per extra component
    topo_penalty = 0.1 * (components - 1)  # 0 penalty if graph is connected
    best_gain_adj = best_gain - topo_penalty
    second_best_gain_adj = second_best_gain - topo_penalty

    # 5️⃣ Hoeffding split decision
    decision = should_split(
        best_gain_adj,
        second_best_gain_adj,
        r,
        delta,
        n,
        tie_threshold,
    )
    return decision


# ----------------------------------------------------------------------
# Auxiliary estimator (unchanged but with clearer validation)
# ----------------------------------------------------------------------
def estimate_rlct_from_losses(train_losses_per_n: List[float], n_values: List[int]) -> float:
    """
    Estimate the asymptotic rate of loss convergence (RLCT) by regressing
    log(loss) against log(log(n)).
    """
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)

    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have the same length")
    if np.any(ns <= np.e):
        raise ValueError("All n_values must be > e for log(log(n)) to be defined")

    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    var_x = np.dot(x_centered, x_centered)

    if var_x < 1e-15:
        raise ValueError("Insufficient variance in log(log(n)) space")

    slope = np.dot(x_centered, y_centered) / var_x
    return slope


# ----------------------------------------------------------------------
# Demo / sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic example
    docs_example = {
        "doc1": ["shingleA", "shingleB"],
        "doc2": ["shingleB", "shingleC"],
        "doc3": ["shingleX"],
    }
    items_example = [b"item1", b"item2", b"item3", b"item4"]
    cm_table = count_min_sketch(items_example, width=128, depth=5)

    r_val = 5.0          # assumed range of gain values
    delta_val = 0.05
    n_val = 200

    split_dec = hybrid_tropical_hoeffding(cm_table, docs_example, r_val, delta_val, n_val)
    print(split_dec)