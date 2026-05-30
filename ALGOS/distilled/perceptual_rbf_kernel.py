# DARWIN HAMMER — match 57, survivor 2
# gen: 2
# parent_a: hybrid_rbf_surrogate_tri_algo_conduit_m8_s0.py (gen1)
# parent_b: perceptual_dedupe.py (gen0)
# born: 2026-05-29T23:24:02Z
#
# DISTILLED USE: Near-duplicate corpus detection with gradient. Perceptual-
# hash Hamming distance folded into the RBF exponent (still PD via Hadamard
# product). Documents 90% identical but differing in headers → near-zero
# Hamming → kernel near 1 → flagged as dupes even when embeddings diverge.
# Drop into CHIMNEY ETL dedup stage as replacement for exact-hash dedup.

"""HybridRBFPerceptual – Fusion of `hybrid_rbf_surrogate_tri_algo_conduit_m8_s0.py` and `perceptual_dedupe.py`.

The original RBF surrogate (Parent A) builds a kernel matrix **K** from Euclidean distances
between feature vectors and solves **K·w = y** for the weights *w*.
The perceptual‑hash utilities (Parent B) map a numeric vector to a binary hash and
provide a Hamming distance *d_H* between two hashes.

The mathematical bridge is a **combined kernel** that mixes the Euclidean metric
used by the RBF with the normalized Hamming distance derived from the perceptual
hashes:


K_ij = exp( -ε_e * ‖x_i - x_j‖²  - ε_h * (d_H(h_i, h_j) / B)² )


where *x_i* are the original feature vectors, *h_i* are their perceptual hashes,
*B* is the hash length in bits, and ε_e, ε_h control the contribution of each
metric.  This kernel retains the linear‑system formulation of the RBF surrogate
while embedding the similarity notion of the perceptual hashing stage.

The module implements:
* `compute_phash` – a lightweight perceptual hash (Parent B).
* `combined_kernel` – builds the hybrid kernel matrix.
* `fit_hybrid` – solves the linear system for weights.
* `HybridRBFSurrogate` – predicts using the hybrid kernel.
* `hybrid_decide` – end‑to‑end decision function that uses both signal/noise
  scoring (from Parent A) and the hybrid surrogate.

All code is pure Python 3 with only the allowed standard‑library modules and
NumPy."""

import math
import random
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Parent A – signal/noise scoring (kept unchanged)
# ----------------------------------------------------------------------
def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> Tuple[float, float]:
    size = len(data)
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(x in (mime or "").lower() for x in ("html", "json", "text", "xml")) else 0.02
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.01)
    signal = max(0.0, min(1.0, 0.20 + status_bonus + mime_bonus + size_bonus + keyword_bonus + structure_bonus))
    noise = max(0.0, min(1.0, 0.58 - keyword_bonus - structure_bonus + (0.12 if size < 64 else 0.0)))
    return signal, noise


# ----------------------------------------------------------------------
# Parent B – perceptual hash utilities (kept unchanged)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash of the numeric vector."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Hybrid kernel utilities
# ----------------------------------------------------------------------
def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def combined_kernel(
    xs: List[Vector],
    hs: List[int],
    epsilon_e: float = 1.0,
    epsilon_h: float = 1.0,
    bits: int = 64,
) -> np.ndarray:
    """
    Build the hybrid kernel matrix K_ij = exp(-ε_e * ||x_i - x_j||² - ε_h * (d_H/bits)²).

    Parameters
    ----------
    xs : list of feature vectors (e.g. [(signal, noise), ...])
    hs : list of integer perceptual hashes corresponding to xs
    epsilon_e : weight for Euclidean component
    epsilon_h : weight for Hamming component
    bits : length of the hash in bits (used for normalization)

    Returns
    -------
    K : np.ndarray of shape (n, n)
    """
    n = len(xs)
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            eu = euclidean(xs[i], xs[j])
            ham = hamming_distance(hs[i], hs[j]) / bits
            K[i, j] = math.exp(-epsilon_e * eu * eu - epsilon_h * ham * ham)
    return K


def solve_linear(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A·x = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1, 1)])
    for col in range(n):
        # pivot
        pivot = max(range(col, n), key=lambda r: abs(M[r, col]))
        if abs(M[pivot, col]) < 1e-12:
            raise ValueError("singular matrix in hybrid surrogate")
        if pivot != col:
            M[[col, pivot]] = M[[pivot, col]]
        # normalize pivot row
        M[col] = M[col] / M[col, col]
        # eliminate other rows
        for row in range(n):
            if row == col:
                continue
            factor = M[row, col]
            M[row] -= factor * M[col]
    return M[:, -1]


# ----------------------------------------------------------------------
# Hybrid surrogate class
# ----------------------------------------------------------------------
class HybridRBFSurrogate:
    """
    RBF surrogate that incorporates perceptual‑hash similarity via the combined kernel.
    """

    def __init__(
        self,
        centers: List[Vector],
        hashes: List[int],
        weights: np.ndarray,
        epsilon_e: float = 1.0,
        epsilon_h: float = 1.0,
        bits: int = 64,
    ):
        self.centers = centers
        self.hashes = hashes
        self.weights = weights
        self.epsilon_e = epsilon_e
        self.epsilon_h = epsilon_h
        self.bits = bits

    def predict(self, x: Vector, h: int) -> float:
        """Predict a scalar using the hybrid kernel."""
        contributions = [
            w
            * math.exp(
                -self.epsilon_e * euclidean(x, c) ** 2
                - self.epsilon_h * (hamming_distance(h, hc) / self.bits) ** 2
            )
            for w, c, hc in zip(self.weights, self.centers, self.hashes)
        ]
        return sum(contributions)


def fit_hybrid(
    points: Iterable[Vector],
    values: Iterable[float],
    epsilon_e: float = 1.0,
    epsilon_h: float = 1.0,
    ridge: float = 1e-9,
) -> HybridRBFSurrogate:
    """
    Fit a HybridRBFSurrogate to training data.

    Each point is first hashed with `compute_phash`. The kernel matrix is built
    with the combined Euclidean/Hamming distance and ridge regularization is
    added on the diagonal.
    """
    xs = [tuple(map(float, p)) for p in points]
    ys = np.array([float(v) for v in values], dtype=float)

    if not xs or len(xs) != len(ys):
        raise ValueError("points and values must be non‑empty and of equal length")

    # Perceptual hash for each point (using the raw numeric values)
    hs = [compute_phash(list(x)) for x in xs]

    K = combined_kernel(xs, hs, epsilon_e, epsilon_h, bits=64)
    # Ridge regularization
    K += ridge * np.eye(len(xs))

    w = solve_linear(K, ys)
    return HybridRBFSurrogate(xs, hs, w, epsilon_e, epsilon_h, bits=64)


# ----------------------------------------------------------------------
# Hybrid decision pipeline
# ----------------------------------------------------------------------
def hybrid_decide(
    data: bytes,
    observations: int,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
    epsilon_e: float = 1.0,
    epsilon_h: float = 1.0,
    ridge: float = 1e-9,
) -> str:
    """
    End‑to‑end decision function.

    1. Compute signal/noise scores (Parent A).
    2. Build a tiny training set consisting of the current (signal, noise) pair
       and a synthetic opposite pair; fit the hybrid surrogate.
    3. Predict using the hybrid model; thresholds map the prediction to a decision.
    """
    signal, noise = signal_scores(data, status_code, mime, keyword_hits, structural_links)

    # Training data: we use the current point and a mirrored point.
    # The mirrored point is designed to have low signal (≈0) and high noise (≈1).
    train_points = [
        (signal, noise),
        (0.0, 1.0),  # opposite extreme
    ]
    train_values = [1.0, 0.0]  # we want the model to output high value for the real point

    surrogate = fit_hybrid(train_points, train_values, epsilon_e, epsilon_h, ridge)

    # Predict for the real observation
    pred = surrogate.predict((signal, noise), compute_phash([signal, noise]))

    # Map prediction to decision space
    if pred > 0.6:
        return "burst"
    elif pred < 0.4:
        return "standby"
    else:
        return "recover"


# ----------------------------------------------------------------------
# Additional helper demonstrating clustering of hash signatures
# ----------------------------------------------------------------------
def cluster_hashes(
    data_items: List[bytes],
    max_distance: int = 4,
) -> List[List[int]]:
    """
    Compute perceptual hashes for a list of byte strings and cluster them
    using a simple Hamming‑distance threshold (Parent B logic).
    Returns clusters as lists of indices into `data_items`.
    """
    hashes = {idx: compute_phash([float(b) for b in item[:64]]) for idx, item in enumerate(data_items)}
    clusters: List[List[int]] = []
    for idx, h in hashes.items():
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            if hamming_distance(h, hashes[rep]) <= max_distance:
                cluster.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
    return clusters


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic payloads
    payloads = [
        b"Hello, world!",
        b"<html><body>Test</body></html>",
        b"{\"key\": \"value\", \"list\": [1,2,3]}",
        b"\x00\x01\x02\x03\x04\x05",
    ]

    # Run hybrid decisions
    for i, p in enumerate(payloads):
        decision = hybrid_decide(
            data=p,
            observations=5,
            status_code=200 if i % 2 == 0 else 404,
            mime="text/plain",
            keyword_hits=i,
            structural_links=i // 2,
        )
        print(f"Payload {i}: decision = {decision}")

    # Demonstrate clustering of the same payloads
    clusters = cluster_hashes(payloads, max_distance=2)
    print("\nHash clusters (indices):")
    for c in clusters:
        print(c)