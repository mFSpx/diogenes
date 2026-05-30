# DARWIN HAMMER — match 3597, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_possum_filter_m323_s0.py (gen4)
# parent_b: hybrid_hybrid_rsa_cipher_hy_hybrid_hybrid_hybrid_m827_s0.py (gen5)
# born: 2026-05-29T23:50:52Z

"""Hybrid Entropy‑RSA‑RBF Fusion
This module fuses the core ideas of two parent algorithms:

* **Parent A** – a span‑based entropy filter that uses deterministic label
  matching and a distance‑threshold to prune models.  The key mathematical
  objects are the Shannon entropy of a score distribution and Euclidean
  distances between span coordinate vectors.

* **Parent B** – an RSA‑protected Radial‑Basis‑Function (RBF) surrogate model.
  The RSA primitive supplies a scalar encryption/decryption channel, while
  the RBF surrogate supplies a kernel‑based linear system solved with a
  matrix inversion.

**Mathematical bridge**

The scalar entropy value `H` computed from the filtered spans is interpreted
as a message `m` for RSA (`c = m^e mod n`).  Simultaneously the span coordinate
vectors are fed to an RBF surrogate whose Gaussian kernel depends on the same
Euclidean distance metric used in the parent‑A filter.  Thus the distance
threshold, the entropy scalar, and the RBF kernel share a single geometric
basis, enabling a seamless hybrid pipeline.

The public‑key RSA encryption secures the surrogate’s prediction, and the
private‑key RSA decryption recovers it, demonstrating the fused workflow.
"""

import math
import random
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared Types and Utilities
# ----------------------------------------------------------------------
Vector = Sequence[float]


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy of a probability vector."""
    probs = probs[probs > 0]  # avoid log(0)
    return -np.sum(probs * np.log2(probs))


# ----------------------------------------------------------------------
# Parent A – Span & Pheromone (simplified)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """A deterministic label span."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    """Lightweight pheromone entry used as a scalar carrier."""
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


# ----------------------------------------------------------------------
# Parent B – RSA & RBF Surrogate
# ----------------------------------------------------------------------
def rsa_encrypt(message: int, e: int, n: int) -> int:
    """RSA encryption: c = m^e mod n."""
    if not 0 <= message < n:
        raise ValueError("message must be in [0, n)")
    return pow(message, e, n)


def rsa_decrypt(ciphertext: int, d: int, n: int) -> int:
    """RSA decryption: m = c^d mod n."""
    if not 0 <= ciphertext < n:
        raise ValueError("ciphertext must be in [0, n)")
    return pow(ciphertext, d, n)


@dataclass
class RBFSurrogate:
    """Gaussian RBF surrogate with analytically solved weights."""
    centers: np.ndarray        # shape (m, d)
    epsilon: float
    weights: np.ndarray        # shape (m,)

    def _kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute Gaussian kernel K_ij = exp(-epsilon^2 * ||x_i - c_j||^2)."""
        dists = np.linalg.norm(X[:, None, :] - self.centers[None, :, :], axis=2)
        return np.exp(- (self.epsilon * dists) ** 2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        K = self._kernel_matrix(X)
        return K @ self.weights


def fit_rbf_surrogate(X: np.ndarray, y: np.ndarray, epsilon: float = 1.0) -> RBFSurrogate:
    """
    Fit an RBF surrogate using Gaussian kernels.
    The centers are taken as the training points X themselves.
    Solves (K + λI) w = y with λ = 1e-8 for numerical stability.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2‑D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1‑D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of rows in X must match length of y")
    # Kernel matrix
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    K = np.exp(- (epsilon * dists) ** 2)
    # Regularization
    lam = 1e-8
    A = K + lam * np.eye(K.shape[0])
    w = np.linalg.solve(A, y)
    return RBFSurrogate(centers=X, epsilon=epsilon, weights=w)


# ----------------------------------------------------------------------
# Hybrid Functions (core of the fusion)
# ----------------------------------------------------------------------
def filter_spans_by_distance(spans: List[Span], threshold: float) -> List[Span]:
    """
    Remove spans whose (start, end) vectors are closer than `threshold`
    to a previously kept span.  Implements the distance‑threshold logic
    from Parent A.
    """
    kept: List[Span] = []
    for sp in spans:
        vec = np.array([sp.start, sp.end], dtype=float)
        if all(euclidean(vec, np.array([k.start, k.end], dtype=float)) >= threshold
               for k in kept):
            kept.append(sp)
    return kept


def compute_entropy_pheromone(spans: List[Span]) -> PheromoneEntry:
    """
    Compute Shannon entropy of the normalized scores of `spans`.
    The resulting entropy (a float in [0, log2 N]) is stored in a
    PheromoneEntry, providing the scalar bridge to RSA.
    """
    if not spans:
        raise ValueError("span list is empty")
    scores = np.array([sp.score for sp in spans], dtype=float)
    probs = scores / scores.sum()
    H = shannon_entropy(probs)
    # Use a half‑life of 300 s arbitrarily
    return PheromoneEntry(
        surface_key="entropy",
        signal_kind="entropy",
        signal_value=H,
        half_life_seconds=300
    )


def hybrid_predict_secure(
    spans: List[Span],
    distance_thr: float,
    rsa_pub: Tuple[int, int],
    rsa_priv: Tuple[int, int],
    epsilon: float = 1.0
) -> Tuple[float, int]:
    """
    Full hybrid pipeline:

    1. Filter spans by Euclidean distance (Parent A).
    2. Compute entropy and wrap it in a pheromone entry.
    3. Fit an RBF surrogate on the filtered spans (Parent B).
    4. Predict the surrogate output for the centroid of the filtered spans.
    5. Encrypt the *integer‑scaled* prediction with RSA.
    6. Return the decrypted float (for verification) and the ciphertext.

    The integer scaling factor is 10⁶ to preserve six decimal places.
    """
    # 1. Distance‑threshold filtering
    filtered = filter_spans_by_distance(spans, distance_thr)
    if not filtered:
        raise RuntimeError("All spans filtered out")

    # 2. Entropy pheromone (provides a scalar but is not directly used later)
    pheromone = compute_entropy_pheromone(filtered)
    # Decay once to simulate time‑passage (no‑op for this demo)
    pheromone.apply_decay()

    # 3. Prepare data for RBF
    X = np.array([[sp.start, sp.end] for sp in filtered], dtype=float)
    y = np.array([sp.score for sp in filtered], dtype=float)

    # 4. Fit surrogate
    surrogate = fit_rbf_surrogate(X, y, epsilon=epsilon)

    # 5. Predict at the centroid of the kept spans
    centroid = X.mean(axis=0, keepdims=True)          # shape (1, 2)
    pred = surrogate.predict(centroid)[0]             # scalar

    # 6. RSA encryption/decryption
    scale = 1_000_000
    msg_int = int(round(pred * scale)) % rsa_pub[1]   # ensure < n
    ciphertext = rsa_encrypt(msg_int, rsa_pub[0], rsa_pub[1])
    decrypted_int = rsa_decrypt(ciphertext, rsa_priv[0], rsa_priv[1])
    decrypted_float = decrypted_int / scale

    return decrypted_float, ciphertext


# ----------------------------------------------------------------------
# Demo / Smoke Test
# ----------------------------------------------------------------------
def _generate_dummy_spans(n: int) -> List[Span]:
    """Create `n` random spans with scores in (0,1]."""
    spans: List[Span] = []
    for _ in range(n):
        start = random.randint(0, 900)
        end = start + random.randint(1, 100)
        text = f"token_{start}_{end}"
        label = random.choice(["A", "B", "C"])
        score = random.random() + 0.01  # avoid zero
        spans.append(Span(start, end, text, label, score))
    return spans


def _simple_rsa_keypair() -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Generate a tiny RSA keypair for demonstration.
    WARNING: NOT SECURE – only for unit‑test purposes.
    """
    # Two small primes
    p = 1019
    q = 1237
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    # Compute modular inverse of e mod phi
    def egcd(a: int, b: int) -> Tuple[int, int, int]:
        if b == 0:
            return (a, 1, 0)
        g, y, x = egcd(b, a % b)
        return (g, x, y - (a // b) * x)

    g, x, _ = egcd(e, phi)
    if g != 1:
        raise RuntimeError("e and phi are not coprime")
    d = x % phi
    return (e, n), (d, n)


if __name__ == "__main__":
    # 1. Create synthetic spans
    random.seed(42)
    spans = _generate_dummy_spans(20)

    # 2. RSA keys
    pub_key, priv_key = _simple_rsa_keypair()

    # 3. Run hybrid pipeline
    try:
        decrypted, cipher = hybrid_predict_secure(
            spans=spans,
            distance_thr=50.0,
            rsa_pub=pub_key,
            rsa_priv=priv_key,
            epsilon=0.5
        )
        print(f"Decrypted prediction: {decrypted:.6f}")
        print(f"Ciphertext (int): {cipher}")
    except Exception as exc:
        print(f"Hybrid pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)