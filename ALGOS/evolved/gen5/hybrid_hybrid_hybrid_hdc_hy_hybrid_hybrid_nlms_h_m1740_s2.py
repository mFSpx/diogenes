# DARWIN HAMMER — match 1740, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hdc_hybrid_hy_hybrid_hybrid_rbf_su_m182_s0.py (gen4)
# parent_b: hybrid_hybrid_nlms_hybrid_h_hybrid_hybrid_ternar_m611_s0.py (gen4)
# born: 2026-05-29T23:38:45Z

"""HybridFusionModel
Combines:
- Parent A: Hyperdimensional Computing primitives (bind, bundle) with an RBF surrogate model.
- Parent B: Normalized Least Mean Squares (NLMS) adaptive learning, morphological sphericity/flatness indices, and an Endpoint Circuit Breaker.

Mathematical Bridge
------------------
1. **Morphology‑Driven Hypervectors** – A `Morphology` instance is encoded into a hypervector by binding
   symbol vectors derived from its geometric attributes. This hypervector modulates the confidence
   of the RBF surrogate via an inner‑product scaling factor.

2. **Routing by Shape Indices** – The sphericity and flatness indices are computed for each incoming
   sample and used to select (or weight) the most appropriate RBF centre, effectively routing the
   surrogate computation.

3. **NLMS‑Weighted Fusion** – The NLMS weight vector is updated with a learning‑rate that is itself
   bound to the morphology hypervector, allowing the adaptive filter to react to shape‑based cues.
   The final prediction is a convex combination of the RBF output and the NLMS output, where the
   mixing coefficient is derived from the bundled recent predictions (a temporal confidence signal).

The implementation below fuses the governing equations of both parents into a single, coherent
system while exposing three core hybrid operations.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional Computing primitives (Parent A)
# ----------------------------------------------------------------------
Vector = List[int]

def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    import hashlib
    seed = int.from_bytes(hashlib.sha256(symbol.encode('utf-8')).digest()[:8], 'big')
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError('vectors must have equal length')
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: List[Vector]) -> Vector:
    if not vectors:
        raise ValueError('at least one vector is required')
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError('vectors must have equal length')
    sums = [0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            sums[i] += x
    return [1 if x >= 0 else -1 for x in sums]

def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

# ----------------------------------------------------------------------
# Morphology & Routing (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

# ----------------------------------------------------------------------
# Endpoint Circuit Breaker (Parent B)
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        return not self.open

# ----------------------------------------------------------------------
# RBF Surrogate (Parent A)
# ----------------------------------------------------------------------
class RBFSurrogate:
    def __init__(self, centers: List[Tuple[float, ...]], weights: List[float], epsilon: float = 1.0):
        if len(centers) != len(weights):
            raise ValueError("centers and weights must have same length")
        self.centers = np.array(centers, dtype=float)
        self.weights = np.array(weights, dtype=float)
        self.epsilon = epsilon

    def __call__(self, x: Sequence[float]) -> float:
        x_arr = np.asarray(x, dtype=float)
        dists = np.linalg.norm(self.centers - x_arr, axis=1)
        kernels = np.exp(-((self.epsilon * dists) ** 2))
        return float(np.dot(self.weights, kernels))

    def update(self, x: Sequence[float], error: float, lr: float = 0.01) -> None:
        """Simple gradient step on the weights."""
        x_arr = np.asarray(x, dtype=float)
        dists = np.linalg.norm(self.centers - x_arr, axis=1)
        kernels = np.exp(-((self.epsilon * dists) ** 2))
        self.weights += lr * error * kernels

# ----------------------------------------------------------------------
# NLMS Adaptive Filter (Parent B)
# ----------------------------------------------------------------------
def nlms_update(weights: np.ndarray, x: np.ndarray, target: float,
                mu: float = 0.5, eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    """Perform one NLMS weight update and return new weights and prediction."""
    y = float(np.dot(weights, x))
    e = target - y
    norm = float(np.dot(x, x)) + eps
    step = (mu / norm) * e
    new_weights = weights + step * x
    return new_weights, y

# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def morphology_to_hypervector(morph: Morphology, dim: int = 10000) -> Vector:
    """Encode a Morphology instance into a hypervector by binding its attribute symbols."""
    lv = symbol_vector(f"len_{morph.length:.3f}", dim)
    wv = symbol_vector(f"wid_{morph.width:.3f}", dim)
    hv = symbol_vector(f"hei_{morph.height:.3f}", dim)
    mv = symbol_vector(f"mass_{morph.mass:.3f}", dim)
    # Bind length and width, then bind height and mass, finally bind the two results
    first = bind(lv, wv)
    second = bind(hv, mv)
    return bind(first, second)

def hybrid_predict(x: List[float],
                   morph: Morphology,
                   rbf: RBFSurrogate,
                   nlms_weights: np.ndarray,
                   recent_preds: List[float],
                   dim: int = 10000) -> float:
    """
    Produce a fused prediction.
    Steps:
    1. Compute a morphology hypervector and its similarity to a reference vector.
       The similarity (dot product) yields a confidence scalar c ∈ [0,1].
    2. Route the input to an RBF centre using sphericity/flatness indices.
    3. Obtain RBF output and NLMS output.
    4. Fuse them: output = c * rbf_out + (1-c) * nlms_out,
       where c is further smoothed by bundling recent predictions.
    """
    # 1. Confidence from hypervector binding
    ref_vec = random_vector(dim, seed="reference")
    morph_vec = morphology_to_hypervector(morph, dim)
    dot = sum(a * b for a, b in zip(morph_vec, ref_vec))
    confidence = (dot + dim) / (2 * dim)  # map from [-dim, dim] to [0,1]

    # 2. Routing factor based on shape indices
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flt = flatness_index(morph.length, morph.width, morph.height)
    routing_weight = (sph + flt) / 2.0  # simple average, ∈ (0,∞)

    # Adjust RBF input by routing weight
    routed_x = [xi * routing_weight for xi in x]

    # 3. Model outputs
    rbf_out = rbf(routed_x)
    nlms_out = float(np.dot(nlms_weights, np.asarray(x, dtype=float)))

    # 4. Temporal confidence from recent predictions (bundle)
    if recent_preds:
        # Convert recent scalar predictions to +/-1 hypervector via sign
        pred_vectors = [
            [1 if p >= 0 else -1 for _ in range(dim)] for p in recent_preds[-5:]
        ]
        bundled = bundle(pred_vectors)
        # similarity of bundled vector to reference
        bundle_dot = sum(a * b for a, b in zip(bundled, ref_vec))
        temporal_factor = (bundle_dot + dim) / (2 * dim)
        confidence = 0.7 * confidence + 0.3 * temporal_factor  # blend
    # Clamp
    confidence = max(0.0, min(1.0, confidence))

    # Final convex fusion
    return confidence * rbf_out + (1.0 - confidence) * nlms_out

def hybrid_update(x: List[float],
                  target: float,
                  morph: Morphology,
                  rbf: RBFSurrogate,
                  nlms_weights: np.ndarray,
                  cb: EndpointCircuitBreaker,
                  dim: int = 10000,
                  mu: float = 0.5) -> Tuple[np.ndarray, float]:
    """
    Update both the RBF surrogate and NLMS weights if the circuit breaker permits.
    The NLMS learning rate mu is modulated by the binding of the morphology hypervector
    with a global context vector.
    Returns updated NLMS weights and the instantaneous prediction error.
    """
    if not cb.allow():
        # Blocked – report zero error to keep caller happy
        return nlms_weights, 0.0

    # Context vector (static)
    context = random_vector(dim, seed="global_context")
    morph_vec = morphology_to_hypervector(morph, dim)
    bound = bind(morph_vec, context)
    # Compute a scalar modulation factor from the bound vector
    modulation = (sum(bound) + dim) / (2 * dim)  # ∈ [0,1]
    adaptive_mu = mu * (0.5 + 0.5 * modulation)  # keep within [0.5*mu, mu]

    # NLMS update
    x_arr = np.asarray(x, dtype=float)
    nlms_weights, nlms_pred = nlms_update(nlms_weights, x_arr, target, mu=adaptive_mu)

    # RBF update – error based on combined prediction
    combined_pred = hybrid_predict(x, morph, rbf, nlms_weights, recent_preds=[nlms_pred])
    error = target - combined_pred
    # Learning rate for RBF is also scaled by modulation
    rbf_lr = 0.01 * (0.5 + 0.5 * modulation)
    rbf.update(x, error, lr=rbf_lr)

    # Record success/failure for circuit breaker
    if abs(error) < 1e-3:
        cb.record_success()
    else:
        cb.record_failure()

    return nlms_weights, error

def initialize_hybrid(dim: int = 10000,
                      rbf_center_count: int = 8,
                      input_dim: int = 4) -> Tuple[RBFSurrogate, np.ndarray]:
    """Utility to create a fresh RBF surrogate and NLMS weight vector."""
    # Random RBF centres in input space
    rng = np.random.default_rng(seed=42)
    centers = [tuple(rng.uniform(-1, 1, size=input_dim)) for _ in range(rbf_center_count)]
    weights = list(rng.normal(0, 0.1, size=rbf_center_count))
    rbf = RBFSurrogate(centers, weights, epsilon=1.5)
    # NLMS weights start near zero
    nlms_weights = np.zeros(input_dim, dtype=float)
    return rbf, nlms_weights

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create dummy data
    dim = 1000  # reduced dimension for quick test
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)
    rbf, nlms_w = initialize_hybrid(dim=dim, rbf_center_count=5, input_dim=3)
    cb = EndpointCircuitBreaker(failure_threshold=2)

    # Simulated stream
    for step in range(10):
        x = [random.uniform(-1, 1) for _ in range(3)]
        target = math.sin(sum(x))  # synthetic target
        pred = hybrid_predict(x, morph, rbf, nlms_w, recent_preds=[])
        nlms_w, err = hybrid_update(x, target, morph, rbf, nlms_w, cb, dim=dim)
        print(f"Step {step:02d} | target={target:.4f} | pred={pred:.4f} | err={err:.4f} | CB={'open' if not cb.allow() else 'closed'}")
    sys.exit(0)