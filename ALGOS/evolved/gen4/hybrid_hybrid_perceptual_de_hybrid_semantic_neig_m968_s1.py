# DARWIN HAMMER — match 968, survivor 1
# gen: 4
# parent_a: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s0.py (gen3)
# parent_b: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s4.py (gen2)
# born: 2026-05-29T23:32:08Z

"""
Hybrid Perceptual‑RBF‑Morphology Circuit

This module fuses the two parent algorithms:

* **Parent A** – a radial‑basis‑function (RBF) surrogate model together with
  perceptual hashing utilities (d‑hash / p‑hash) for clustering similar data.
* **Parent B** – morphological descriptors (sphericity, flatness, righting‑time)
  and a cosine‑similarity based “semantic neighbour” notion that drives an
  endpoint circuit‑breaker whose failure threshold is modulated by a recovery
  priority.

**Mathematical bridge**

The bridge is the *similarity score* that both sides compute:

* In Parent A similarity is expressed by the Gaussian kernel  
  `K(x, c) = exp(-(ε·‖x‑c‖)²)`.
* In Parent B similarity is expressed by the cosine similarity  
  `cos(a, b) = (a·b) / (‖a‖‖b‖)` and by morphological similarity via the
  recovery priority `ρ = righting_time / max_index`.

The hybrid algorithm uses the Gaussian kernel to obtain a surrogate prediction
`ŷ(x)`.  This prediction is then weighted by the recovery priority `ρ` derived
from the morphology of the system.  The weighted prediction modulates the
circuit‑breaker’s dynamic failure threshold:


dynamic_threshold = base_threshold * (1 - ρ)


Thus the RBF surrogate provides a data‑driven signal, while the morphology‑driven
priority adapts the robustness of the endpoint circuit.  Perceptual hashing is
employed to cluster input points, allowing separate surrogate models per
cluster, which further refines the similarity assessment.

The module supplies three high‑level functions that demonstrate the hybrid
behaviour:
`fit_hybrid`, `predict_hybrid`, and `update_circuit`.
"""

import math
import random
import sys
import pathlib
import numpy as np
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Parent A utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()

def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: median‑based 64‑bit signature."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

class RBFSurrogate:
    """Radial‑basis‑function surrogate model."""
    def __init__(self, centers: List[Tuple[float, ...]], weights: np.ndarray, epsilon: float = 1.0):
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def predict(self, x: List[float]) -> float:
        """Predict a scalar value for input vector x."""
        return float(
            sum(
                w * gaussian(euclidean(x, c), self.epsilon)
                for w, c in zip(self.weights, self.centers)
            )
        )

def fit_rbf(points: List[List[float]],
            values: List[float],
            epsilon: float = 1.0,
            ridge: float = 1e-9) -> RBFSurrogate:
    """
    Fit an RBF surrogate by solving (Φ + λI)w = y,
    where Φ_{ij}=gaussian(||p_i-p_j||, ε).
    """
    n = len(points)
    if n == 0:
        raise ValueError("No points to fit.")
    # Build kernel matrix Φ
    phi = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            phi[i, j] = gaussian(euclidean(points[i], points[j]), epsilon)
    # Regularisation
    phi += ridge * np.eye(n)
    # Solve for weights
    y = np.asarray(values, dtype=float)
    weights, *_ = np.linalg.lstsq(phi, y, rcond=None)
    centers = [tuple(map(float, p)) for p in points]
    return RBFSurrogate(centers, weights, epsilon)

# ----------------------------------------------------------------------
# Parent B utilities
# ----------------------------------------------------------------------
class Morphology:
    """Geometric description of an object."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        if min(length, width, height, mass) <= 0:
            raise ValueError("All morphology parameters must be positive.")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean to the longest dimension."""
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness as (length+width)/(2·height)."""
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology,
                        b: float = 1.0 / 3.0,
                        k: float = 0.35,
                        neck_lever: float = 1.0) -> float:
    """Estimated time for a self‑righting object to recover."""
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """
    Normalised priority in [0,1] that reflects how quickly the object can
    recover; higher priority means more tolerant circuit behaviour.
    """
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if den == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / den

# ----------------------------------------------------------------------
# Hybrid structures
# ----------------------------------------------------------------------
class HybridCircuitRBF:
    """
    Combines an RBF surrogate with morphology‑driven circuit‑breaker logic.
    """
    def __init__(self,
                 surrogate: RBFSurrogate,
                 morphology: Morphology,
                 base_failure_threshold: int = 3):
        self.surrogate = surrogate
        self.morphology = morphology
        self.base_failure_threshold = base_failure_threshold
        self.failure_counter = 0

    def predict(self, x: List[float]) -> float:
        """Raw surrogate prediction."""
        return self.surrogate.predict(x)

    def _dynamic_threshold(self) -> float:
        """
        Threshold that shrinks when recovery priority is high,
        i.e. the system tolerates more failures.
        """
        rho = recovery_priority(self.morphology)
        return self.base_failure_threshold * (1.0 - rho)

    def check_and_update(self, x: List[float]) -> Tuple[float, bool, int]:
        """
        Evaluate input x, update the failure counter and report circuit status.

        Returns
        -------
        pred : float
            Surrogate prediction.
        tripped : bool
            True if the failure counter exceeds the dynamic threshold.
        counter : int
            Current failure counter value.
        """
        pred = self.predict(x)
        # Simple heuristic: negative predictions indicate a “failure”.
        if pred < 0.0:
            self.failure_counter += 1
        else:
            self.failure_counter = max(0, self.failure_counter - 1)

        tripped = self.failure_counter >= self._dynamic_threshold()
        return pred, tripped, self.failure_counter

# ----------------------------------------------------------------------
# Public hybrid API
# ----------------------------------------------------------------------
def fit_hybrid(points: List[List[float]],
               values: List[float],
               morphology: Morphology,
               epsilon: float = 1.0,
               ridge: float = 1e-9,
               base_failure_threshold: int = 3) -> HybridCircuitRBF:
    """
    Fit a HybridCircuitRBF model to training data.
    """
    surrogate = fit_rbf(points, values, epsilon, ridge)
    return HybridCircuitRBF(surrogate, morphology, base_failure_threshold)

def predict_hybrid(hybrid: HybridCircuitRBF, x: List[float]) -> float:
    """
    Shortcut to obtain the raw surrogate prediction from a hybrid model.
    """
    return hybrid.predict(x)

def update_circuit(hybrid: HybridCircuitRBF, x: List[float]) -> Dict[str, object]:
    """
    Evaluate x with the hybrid model, update the circuit‑breaker state and
    return a dictionary summarising the outcome.
    """
    pred, tripped, count = hybrid.check_and_update(x)
    return {
        "prediction": pred,
        "circuit_tripped": tripped,
        "failure_counter": count,
        "recovery_priority": recovery_priority(hybrid.morphology)
    }

def cluster_surrogates(points: List[List[float]],
                      values: List[float],
                      epsilon: float = 1.0) -> Dict[int, RBFSurrogate]:
    """
    Group points by perceptual hash and fit an independent RBF surrogate for each
    cluster.  Returns a mapping hash → surrogate.
    """
    clusters: Dict[int, List[Tuple[List[float], float]]] = {}
    for p, v in zip(points, values):
        h = compute_phash(p)
        clusters.setdefault(h, []).append((p, v))

    surrogates: Dict[int, RBFSurrogate] = {}
    for h, pv in clusters.items():
        pts = [pt for pt, _ in pv]
        vals = [val for _, val in pv]
        surrogates[h] = fit_rbf(pts, vals, epsilon)
    return surrogates

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic data
    dim = 5
    n_samples = 30
    random.seed(42)
    np.random.seed(42)

    points = [list(np.random.rand(dim) * 10) for _ in range(n_samples)]
    # Target function: sum of sines of coordinates
    values = [float(np.sin(p).sum()) for p in points]

    # Random morphology
    morph = Morphology(
        length=random.uniform(0.5, 2.0),
        width=random.uniform(0.5, 2.0),
        height=random.uniform(0.5, 2.0),
        mass=random.uniform(1.0, 5.0)
    )

    # Fit hybrid model
    hybrid = fit_hybrid(points, values, morph, epsilon=0.8, base_failure_threshold=4)

    # New query point
    query = list(np.random.rand(dim) * 10)

    # Perform prediction
    pred = predict_hybrid(hybrid, query)
    print(f"Raw surrogate prediction: {pred:.4f}")

    # Update circuit breaker
    result = update_circuit(hybrid, query)
    print("Circuit update result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Demonstrate clustering surrogates
    cluster_map = cluster_surrogates(points, values, epsilon=0.8)
    print(f"Number of perceptual‑hash clusters: {len(cluster_map)}")
    # Predict with a cluster surrogate (choose first hash)
    first_hash = next(iter(cluster_map))
    cluster_pred = cluster_map[first_hash].predict(query)
    print(f"Prediction from first cluster surrogate (hash {first_hash}): {cluster_pred:.4f}")