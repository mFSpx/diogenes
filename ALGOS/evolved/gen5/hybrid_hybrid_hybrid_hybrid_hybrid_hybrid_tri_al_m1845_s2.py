# DARWIN HAMMER — match 1845, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m498_s0.py (gen4)
# parent_b: hybrid_hybrid_tri_algo_cond_hybrid_hard_truth_ma_m755_s0.py (gen3)
# born: 2026-05-29T23:39:23Z

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
from pathlib import Path


# ----------------------------------------------------------------------
# Immutable decision container
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ConduitDecision:
    """Immutable container for a decision made by the hybrid system."""
    action: str
    confidence_gap: float
    epsilon: float
    signal_score: float
    noise_score: float
    dormancy_probability: float
    recovery_priority: float
    reason: str


# ----------------------------------------------------------------------
# Lead‑lag transform (unchanged, but with explicit validation)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Interleave lead‑lag channels.
    Input: (T, d) array.
    Output: (2T‑1, 2d) array.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dimension)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


# ----------------------------------------------------------------------
# B‑spline basis (robust implementation)
# ----------------------------------------------------------------------
def _augmented_knots(grid: np.ndarray, k: int) -> np.ndarray:
    """Create a clamped knot vector for a given grid and spline order."""
    grid = np.asarray(grid, dtype=float)
    if k < 1:
        raise ValueError("Spline order k must be >= 1")
    # Clamp ends with multiplicity k
    left = np.full(k, grid[0])
    right = np.full(k, grid[-1])
    return np.concatenate([left, grid, right])


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Evaluate B‑spline basis functions of order k at positions x.
    Returns B where B[i, j] = B_j(x_i).
    The number of basis functions equals len(grid) + k - 1.
    """
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError("grid must be one‑dimensional")
    if k < 1:
        raise ValueError("order k must be >= 1")

    t = _augmented_knots(grid, k)
    n_basis = len(t) - k - 1  # standard B‑spline count

    # Initialise order‑1 (piecewise constant) basis
    B = np.zeros((len(x), n_basis), dtype=float)
    for i in range(n_basis):
        left, right = t[i], t[i + 1]
        B[:, i] = np.where((x >= left) & (x < right), 1.0, 0.0)
    # Include the right‑most knot explicitly
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    # Cox‑de Boor recursion
    for order in range(2, k + 1):
        B_next = np.zeros((len(x), n_basis - order + 1), dtype=float)
        for i in range(n_basis - order + 1):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]

            term_l = ((x - t[i]) / denom_l) * B[:, i] if denom_l > 0 else 0.0
            term_r = ((t[i + order] - x) / denom_r) * B[:, i + 1] if denom_r > 0 else 0.0
            B_next[:, i] = term_l + term_r
        B = B_next
    return B


# ----------------------------------------------------------------------
# Entropy utilities (local, per‑segment weighting)
# ----------------------------------------------------------------------
def shannon_entropy(sequence: bytes) -> float:
    """Shannon entropy (bits) for a byte sequence."""
    if not sequence:
        return 0.0
    freq = np.bincount(np.frombuffer(sequence, dtype=np.uint8), minlength=256)
    prob = freq[freq > 0] / len(sequence)
    return -np.sum(prob * np.log2(prob))


def normalized_byte_entropy(data: bytes) -> float:
    """Normalized entropy in [0,1] (8 bits max)."""
    if not data:
        return 0.0
    return shannon_entropy(data) / 8.0


def local_entropy_vector(text: bytes, length: int) -> np.ndarray:
    """
    Split `text` into `length` (approximately) equal chunks and compute a
    normalized entropy for each chunk.  If the text is shorter than `length`,
    the last entropy value is repeated.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if not text:
        return np.zeros(length, dtype=float)

    chunk_size = max(1, len(text) // length)
    entropies = []
    for i in range(length):
        start = i * chunk_size
        end = start + chunk_size if i < length - 1 else len(text)
        entropies.append(normalized_byte_entropy(text[start:end]))
    return np.array(entropies, dtype=float)


def deterministic_jitter(idx: int, scale: float = 0.005) -> float:
    """
    Produce a small deterministic jitter based on an integer index.
    Guarantees reproducibility without a global RNG seed.
    """
    # Simple hash‑like mixing
    h = (idx * 0x9e3779b9) & 0xffffffff
    jitter = ((h >> 16) & 0xffff) / 0xffff  # in [0,1]
    jitter = (jitter - 0.5) * 2 * scale      # centred around 0
    return jitter


def entropy_weight_vector(text: bytes, dim: int) -> np.ndarray:
    """
    Produce a length‑dim weight vector from text entropy.
    Uses a local entropy profile combined with a tiny deterministic jitter
    to avoid exact singularities.
    """
    base = local_entropy_vector(text, dim)
    jitter = np.array([deterministic_jitter(i) for i in range(dim)], dtype=float)
    w = np.clip(base + jitter, 0.0, 1.0)
    return w


# ----------------------------------------------------------------------
# Count‑Min Sketch (fixed key handling and extraction)
# ----------------------------------------------------------------------
class CountMinSketch:
    """
    Lightweight Count‑Min Sketch.
    Keys are 64‑bit unsigned integers; hashing uses universal linear hashing.
    """
    def __init__(self, depth: int = 5, width: int = 2 ** 12):
        self.depth = depth
        self.width = width
        self.tables = np.zeros((depth, width), dtype=np.int64)
        rng = np.random.default_rng(seed=12345)
        self._hash_a = rng.integers(1, 2 ** 31 - 1, size=depth, dtype=np.int64)
        self._hash_b = rng.integers(0, 2 ** 31 - 1, size=depth, dtype=np.int64)
        self._prime = 2 ** 31 - 1

    def _hash(self, x: int, row: int) -> int:
        return ((self._hash_a[row] * x + self._hash_b[row]) % self._prime) % self.width

    def update(self, key: int, increment: int = 1) -> None:
        for r in range(self.depth):
            idx = self._hash(key, r)
            self.tables[r, idx] += increment

    def estimate(self, key: int) -> int:
        return min(self.tables[r, self._hash(key, r)] for r in range(self.depth))


def _pair_key(i: int, j: int) -> int:
    """
    Deterministic 64‑bit pairing of two non‑negative integers.
    Uses Cantor's pairing then masks to 64 bits.
    """
    return ((i + j) * (i + j + 1) // 2 + j) & 0xFFFFFFFFFFFFFFFF


def sketch_from_matrix(Z: np.ndarray, cms: CountMinSketch) -> Dict[Tuple[int, int], int]:
    """
    Insert integer‑rounded entries of Z into the sketch and return a dict
    mapping (i, j) → estimated count.
    """
    rows, cols = Z.shape
    for i in range(rows):
        for j in range(cols):
            val = int(round(Z[i, j]))
            if val != 0:
                key = _pair_key(i, j)
                cms.update(key, val)

    estimates: Dict[Tuple[int, int], int] = {}
    for i in range(rows):
        for j in range(cols):
            key = _pair_key(i, j)
            est = cms.estimate(key)
            if est != 0:
                estimates[(i, j)] = est
    return estimates


# ----------------------------------------------------------------------
# Hybrid core functions (deeper integration)
# ----------------------------------------------------------------------
def hybrid_feature_matrix(path: np.ndarray, grid_points: int = 20, spline_order: int = 3) -> np.ndarray:
    """
    Apply lead‑lag transform, then project each coordinate onto a B‑spline basis.
    The resulting feature matrix concatenates the bases of all coordinates,
    yielding Φ ∈ ℝ^{(2T‑1) × (d·(grid_points + k - 2))}.
    """
    lead_lag = lead_lag_transform(path)                # (2T‑1, 2d)
    T2, dim = lead_lag.shape
    # Normalise each coordinate to [0, 1] before spline evaluation
    normalized = (lead_lag - lead_lag.min(axis=0)) / (lead_lag.ptp(axis=0) + 1e-12)

    # Build a common grid in [0,1] for all dimensions
    grid = np.linspace(0.0, 1.0, grid_points)

    basis_list = []
    for d_idx in range(dim):
        col_series = normalized[:, d_idx]
        B = bspline_basis(col_series, grid, k=spline_order)   # (2T‑1, M)
        basis_list.append(B)

    Φ = np.concatenate(basis_list, axis=1)  # (2T‑1, dim·M)
    # Row‑wise L2 normalisation to improve numerical stability
    row_norms = np.linalg.norm(Φ, axis=1, keepdims=True) + 1e-12
    Φ = Φ / row_norms
    return Φ


def entropy_weighted_bilinear(Φ: np.ndarray, text: bytes) -> np.ndarray:
    """
    Compute Z = Φᵀ diag(w_t) Φ where w_t is a per‑observation weight derived
    from a local entropy profile of `text`.  This couples the temporal
    structure of the path with the information density of the accompanying
    text at a finer granularity than a single scalar weight.
    """
    N = Φ.shape[0]
    w = local_entropy_vector(text, N)                     # (N,)
    # Add deterministic jitter to avoid exact zero rows
    jitter = np.array([deterministic_jitter(i) for i in range(N)], dtype=float)
    w = np.clip(w + jitter, 0.0, 1.0)

    # Weighted Gram matrix
    Z = Φ.T @ (Φ * w[:, None])
    # Symmetrise and normalise (optional but improves downstream sketch stability)
    Z = (Z + Z.T) / 2.0
    norm = np.linalg.norm(Z, ord='fro') + 1e-12
    Z = Z / norm
    return Z


def hybrid_pipeline(path: np.ndarray, text: bytes,
                    grid_points: int = 20,
                    spline_order: int = 3,
                    cms_depth: int = 5,
                    cms_width: int = 2 ** 12) -> Tuple[Dict[Tuple[int, int], int], np.ndarray]:
    """
    End‑to‑end hybrid processing:
      1. Build feature matrix Φ.
      2. Compute entropy‑weighted bilinear form Z.
      3. Compress Z with a Count‑Min Sketch.
    Returns the sketch dictionary and the (normalised) matrix Z for optional
    downstream use.
    """
    Φ = hybrid_feature_matrix(path, grid_points=grid_points, spline_order=spline_order)
    Z = entropy_weighted_bilinear(Φ, text)
    cms = CountMinSketch(depth=cms_depth, width=cms_width)
    sketch = sketch_from_matrix(Z, cms)
    return sketch, Z


# ----------------------------------------------------------------------
# Example usage (can be removed or adapted in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic path: a 2‑D sinusoid
    t = np.linspace(0, 2 * np.pi, 50)
    path = np.column_stack([np.sin(t), np.cos(t)])

    # Synthetic text: repeating pattern to give moderate entropy
    text = (b"The quick brown fox jumps over the lazy dog. " * 10)

    sketch_dict, Z_norm = hybrid_pipeline(path, text)

    # Simple decision heuristic based on sketch sparsity
    nonzero = len(sketch_dict)
    total = Z_norm.size
    confidence = 1.0 - nonzero / total
    decision = ConduitDecision(
        action="accept" if confidence > 0.7 else "reject",
        confidence_gap=confidence,
        epsilon=1e-3,
        signal_score=nonzero,
        noise_score=total - nonzero,
        dormancy_probability=1.0 - confidence,
        recovery_priority=confidence,
        reason=f"Sketch sparsity={nonzero}/{total}"
    )
    print(decision)