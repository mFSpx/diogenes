# DARWIN HAMMER — match 940, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_voronoi_parti_m53_s0.py (gen4)
# parent_b: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s2.py (gen3)
# born: 2026-05-29T23:31:43Z

"""Hybrid Bayesian‑RBF‑Perceptual Model
Integrates:
- Parent A (hybrid_hybrid_hybrid_bayes__...): SSIM similarity, prototype‑based hybrid_score.
- Parent B (hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_...): perceptual hashing and radial‑basis‑function surrogate.

Mathematical bridge:
The perceptual hash of a payload is turned into a binary feature vector. This vector is fed to an
RBF surrogate (Gaussian kernel). The raw RBF output is modulated by the SSIM between the payload
and a fixed prototype vector and by the prototype‑distance similarity from Parent A. The final
fused score is

    fused = SSIM(x, p) · RBF(hash_vec) · (1 / (1+‖x‑p‖₂))

where x is the payload, p is PROTOTYPE_VECTOR and hash_vec is the binary hash representation.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants and utilities (from Parent A)
# ----------------------------------------------------------------------
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)


def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index (SSIM) for 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)


def hybrid_score(packet: dict) -> float:
    """Prototype‑distance similarity used in Parent A."""
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        return 0.0
    payload_vec = np.asarray(payload, dtype=np.float64)
    # Pad or truncate to prototype size for a fair distance
    if payload_vec.size < PROTOTYPE_VECTOR.size:
        payload_vec = np.pad(payload_vec, (0, PROTOTYPE_VECTOR.size - payload_vec.size))
    elif payload_vec.size > PROTOTYPE_VECTOR.size:
        payload_vec = payload_vec[: PROTOTYPE_VECTOR.size]
    dist = np.linalg.norm(payload_vec - PROTOTYPE_VECTOR)
    # Convert distance to similarity in (0,1]
    return 1.0 / (1.0 + dist)


# ----------------------------------------------------------------------
# Perceptual hashing and RBF surrogate (from Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: compare each value to the mean."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hash_to_vector(h: int, length: int = 64) -> List[float]:
    """Convert integer hash to a binary float vector."""
    return [(h >> i) & 1 for i in reversed(range(length))]


@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: List[float]) -> float:
        """Gaussian‑RBF prediction."""
        return sum(
            w * gaussian(euclidean(x, list(c)), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


def fit_rbf_surrogate(points: List[List[float]], values: List[float], epsilon: float = 1.0) -> RBFSurrogate:
    """Fit an RBF surrogate by solving (K·w = y) with least‑squares."""
    n = len(points)
    if n == 0:
        raise ValueError("no training points")
    # Build kernel matrix K[i,j] = gaussian(||p_i - p_j||)
    K = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            K[i, j] = gaussian(euclidean(points[i], points[j]), epsilon)
    y = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    # Least‑squares solution
    w, *_ = np.linalg.lstsq(K, y, rcond=None)
    return RBFSurrogate(centers=[tuple(p) for p in points], weights=w.flatten().tolist(), epsilon=epsilon)


# ----------------------------------------------------------------------
# Hybrid model that fuses both parent topologies
# ----------------------------------------------------------------------
class HybridModel:
    """Combines SSIM, prototype similarity and an RBF surrogate built on perceptual hashes."""

    def __init__(self, rbf: RBFSurrogate):
        self.rbf = rbf

    def predict(self, packet: dict) -> float:
        """
        Compute the fused score:
            fused = SSIM(payload, prototype) * RBF(hash_vec) * hybrid_score(payload)
        """
        payload = packet.get("payload")
        if not isinstance(payload, (list, tuple)):
            raise ValueError("packet must contain a 'payload' list")
        payload_list = list(map(float, payload))

        # 1. SSIM with prototype (truncate/pad to same length)
        min_len = min(len(payload_list), len(PROTOTYPE_VECTOR))
        ssim_val = compute_ssim(payload_list[:min_len], PROTOTYPE_VECTOR.tolist()[:min_len])

        # 2. Perceptual hash → binary vector → RBF prediction
        ph = compute_phash(payload_list)
        hash_vec = hash_to_vector(ph, length=len(self.rbf.centers[0]))
        rbf_val = self.rbf.predict(hash_vec)

        # 3. Prototype‑distance similarity from Parent A
        hyb_val = hybrid_score(packet)

        return ssim_val * rbf_val * hyb_val


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def demo_hybrid_score():
    pkt = {"payload": [0.2, 0.5, 0.31, 0.68, 0.09]}
    return hybrid_score(pkt)


def demo_rbf_training():
    # Generate synthetic hash vectors and target values
    points = []
    values = []
    for _ in range(20):
        vec = [random.random() for _ in range(64)]
        points.append(vec)
        # Arbitrary target: sum of bits + small noise
        values.append(sum(vec) + random.uniform(-0.1, 0.1))
    return fit_rbf_surrogate(points, values, epsilon=0.5)


def demo_fused_prediction(model: HybridModel):
    test_packet = {"payload": [0.15, 0.55, 0.28, 0.72, 0.12]}
    return model.predict(test_packet)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Verify hybrid_score works
    print("Hybrid score:", demo_hybrid_score())

    # 2. Train an RBF surrogate
    surrogate = demo_rbf_training()
    print("RBF surrogate trained with", len(surrogate.centers), "centers.")

    # 3. Build the fused model and evaluate a packet
    model = HybridModel(surrogate)
    fused = demo_fused_prediction(model)
    print("Fused prediction:", fused)