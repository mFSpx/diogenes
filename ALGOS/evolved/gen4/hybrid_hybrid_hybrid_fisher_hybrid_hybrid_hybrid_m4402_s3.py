# DARWIN HAMMER — match 4402, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_fisher_m375_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_sketch_hybrid_hoeffding_tre_m16_s2.py (gen3)
# born: 2026-05-29T23:55:30Z

import numpy as np
import math
import random
import hashlib
from typing import List, Tuple, Iterable, Dict, Any


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _hash(item: Any, seed: int, width: int) -> int:
    """Deterministic hash that mixes the seed with the item."""
    h = hashlib.blake2b(digest_size=8)
    h.update(seed.to_bytes(4, "little"))
    h.update(str(item).encode())
    return int.from_bytes(h.digest(), "little") % width


def gaussian_beam(theta: float, center: float, sigma: float) -> float:
    """Gaussian kernel (unnormalised) used for the Fisher information."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (theta - center) / sigma
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, sigma: float, eps: float = 1e-12) -> float:
    """Scalar Fisher information for a single count."""
    intensity = max(gaussian_beam(theta, center, sigma), eps)
    derivative = intensity * (-(theta - center) / (sigma * sigma))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Count‑Min Sketch with vectorised Fisher scores
# ----------------------------------------------------------------------
class CountMinSketch:
    """
    Simple Count‑Min sketch with pairwise‑independent hash functions.
    The internal matrix is stored as a NumPy array of shape (depth, width).
    """

    def __init__(self, width: int = 64, depth: int = 4, seed: int = 0):
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive integers")
        self.width = width
        self.depth = depth
        self.seed = seed
        self.table = np.zeros((depth, width), dtype=np.int64)

        # generate a different seed for each row to obtain independent hashes
        rng = np.random.default_rng(seed)
        self.row_seeds = rng.integers(0, 2 ** 31, size=depth, dtype=np.int64)

    def update(self, items: Iterable[Any]) -> None:
        for item in items:
            for d, row_seed in enumerate(self.row_seeds):
                idx = _hash(item, int(row_seed), self.width)
                self.table[d, idx] += 1

    def matrix(self) -> np.ndarray:
        """Return the raw count matrix."""
        return self.table.copy()

    def fisher_matrix(
        self, center: float = 0.0, sigma: float = 1.0, eps: float = 1e-12
    ) -> np.ndarray:
        """
        Apply the Fisher information scoring element‑wise to the count matrix.
        Returns a matrix of the same shape containing the scores.
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        # vectorised version of gaussian_beam and fisher_score
        theta = self.table.astype(np.float64)
        intensity = np.maximum(
            np.exp(-0.5 * ((theta - center) / sigma) ** 2), eps
        )
        derivative = intensity * (-(theta - center) / (sigma * sigma))
        return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Tropical (max‑plus) algebra utilities
# ----------------------------------------------------------------------
def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition (max)."""
    return np.maximum(x, y)


def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication (plus)."""
    return np.add(x[:, None], y[None, :])


def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical matrix multiplication (max‑plus).
    Result[i, j] = max_k (A[i, k] + B[k, j])
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible shapes for tropical multiplication")
    # broadcasting trick: compute all pairwise sums then max over k
    sums = A[:, :, None] + B[None, :, :]
    return np.max(sums, axis=1)


# ----------------------------------------------------------------------
# Hybrid algorithms
# ----------------------------------------------------------------------
def hybrid_fisher_count_min_sketch(
    items: Iterable[Any],
    width: int = 64,
    depth: int = 4,
    center: float = 0.0,
    sigma: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Build a Count‑Min sketch from *items* and return the Fisher‑scored matrix.
    The result has shape (depth, width) and can be used directly in tropical
    operations.
    """
    cms = CountMinSketch(width=width, depth=depth, seed=seed)
    cms.update(items)
    return cms.fisher_matrix(center=center, sigma=sigma)


def hybrid_hoeffding_tropical_max_plus(
    tree: List[Dict[str, Any]],
    delta: float = 0.05,
    n: int = 100,
) -> Tuple[bool, float, float, str]:
    """
    Decide whether to split the best node of a decision tree using a Hoeffding bound.
    The gain values are first combined with tropical addition to obtain a robust
    estimate of the best and second‑best gains.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    if n <= 0:
        raise ValueError("n must be positive")

    gains = np.array([node.get("gain", 0.0) for node in tree], dtype=np.float64)
    if gains.size == 0:
        return False, 0.0, 0.0, "empty_tree"

    # Tropical max‑plus gives the same ordering as ordinary max, but we keep it
    # for conceptual consistency.
    best_gain = np.max(gains)
    # mask the best element to find the second best
    mask = gains != best_gain
    second_best_gain = np.max(gains[mask]) if np.any(mask) else best_gain

    eps = math.sqrt((best_gain * best_gain * math.log(1.0 / delta)) / (2.0 * n))
    gap = best_gain - second_best_gain
    split = gap > eps
    reason = "gap_exceeds_bound" if split else "wait"
    return split, eps, gap, reason


def hybrid_sketch_fisher_similarity(
    items1: Iterable[Any],
    items2: Iterable[Any],
    width: int = 64,
    depth: int = 4,
    center: float = 0.0,
    sigma: float = 1.0,
    seed: int = 0,
) -> float:
    """
    Compute a similarity measure between two multisets based on the
    Pearson correlation of their Fisher‑scored Count‑Min sketches.
    Returns a value in [-1, 1]; NaN is replaced by 0.0.
    """
    cms1 = CountMinSketch(width=width, depth=depth, seed=seed)
    cms2 = CountMinSketch(width=width, depth=depth, seed=seed + 1)  # independent hashes
    cms1.update(items1)
    cms2.update(items2)

    f1 = cms1.fisher_matrix(center=center, sigma=sigma).ravel()
    f2 = cms2.fisher_matrix(center=center, sigma=sigma).ravel()

    if np.allclose(f1, f1[0]) or np.allclose(f2, f2[0]):
        # one of the vectors is constant → correlation undefined
        return 0.0

    corr = np.corrcoef(f1, f2)[0, 1]
    return float(np.nan_to_num(corr, nan=0.0))


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    items_a = [random.randint(0, 100) for _ in range(200)]
    items_b = [random.randint(0, 100) for _ in range(200)]

    # Hybrid Fisher‑scored sketch
    fisher_mat = hybrid_fisher_count_min_sketch(items_a, width=128, depth=6, sigma=2.0)
    print("Fisher‑scored sketch shape:", fisher_mat.shape)

    # Decision‑tree Hoeffding test with tropical algebraic flavour
    example_tree = [{"gain": random.random()} for _ in range(15)]
    split, eps, gap, reason = hybrid_hoeffding_tropical_max_plus(example_tree, delta=0.01, n=500)
    print(f"Split decision: {split} (eps={eps:.5f}, gap={gap:.5f}) – {reason}")

    # Similarity between two sketches
    sim = hybrid_sketch_fisher_similarity(items_a, items_b, width=128, depth=6, sigma=2.0)
    print(f"Sketch similarity (Pearson r): {sim:.4f}")