# DARWIN HAMMER — match 2558, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m1168_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_caputo_m1013_s1.py (gen4)
# born: 2026-05-29T23:42:52Z

"""Hybrid Morphology‑RBF‑Caputo State‑Space Model

This module fuses the core topologies of two parent algorithms:

* **Parent A** – a tropical‑semiring state‑space model that evaluates engine‑endpoint
  health via morphology‑derived indices (sphericity, flatness, righting‑time) and
  uses those health scores to weight state transitions.

* **Parent B** – an RBF surrogate that predicts stylometric‑style features from
  input vectors and supplies them to a Caputo fractional‑derivative weighting
  scheme; the resulting weights are combined with matrix‑multiplication bilinear
  forms.

**Mathematical bridge** – both parents rely on weighted bilinear forms:
the health score (a scalar) multiplies the tropical state‑space transition,
while the RBF‑predicted feature (a scalar) multiplies the Caputo fractional
weight (also a scalar).  By forming the product  


γ = health * rbf_feature * caputo_weight(α)


and inserting `γ` as a scalar multiplier of the tropical matrix‑vector
operation, we obtain a single unified update rule that respects both the
max‑plus algebra of the tropical model and the fractional‑derivative influence
of the RBF‑Caputo model.

The three demonstration functions below illustrate:
1. a single hybrid step,
2. a sequential hybrid evolution,
3. a parallel batch evaluation.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Mapping, Hashable, Set, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – morphology and tropical state‑space utilities
# ----------------------------------------------------------------------
Vector = Sequence[float]

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

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Health score in [0,1] derived from righting‑time index."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

# Tropical (max‑plus) algebra helpers -------------------------------------------------
def tropical_mul(a: float, b: float) -> float:
    """Tropical multiplication is ordinary addition."""
    return a + b

def tropical_add(a: float, b: float) -> float:
    """Tropical addition is the maximum."""
    return max(a, b)

def tropical_matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute y_i = max_j (A_ij + x_j) using max‑plus algebra."""
    rows, cols = A.shape
    if x.shape[0] != cols:
        raise ValueError("dimension mismatch")
    y = np.full(rows, -np.inf)
    for i in range(rows):
        for j in range(cols):
            y[i] = tropical_add(y[i], tropical_mul(A[i, j], x[j]))
    return y

# ----------------------------------------------------------------------
# Parent B – RBF surrogate and Caputo fractional derivative utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """RBF surrogate prediction for a single input vector."""
        if len(self.centers) != len(self.weights):
            raise ValueError("centers and weights length mismatch")
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

def caputo_weight(alpha: float, dt: float = 1.0) -> float:
    """
    Simple Caputo fractional‑derivative scaling factor.
    For order 0 < α < 1 the weight ~ dt^{1‑α} / Γ(2‑α).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    return (dt ** (1.0 - alpha)) / math.gamma(2.0 - alpha)

# ----------------------------------------------------------------------
# Hybrid core – combine health, RBF feature, and Caputo weight into tropical SSM
# ----------------------------------------------------------------------
def hybrid_compute_gamma(
    morph: Morphology,
    rbf: RBFSurrogate,
    x_input: Vector,
    alpha: float,
) -> float:
    """
    Compute the scalar coupling γ = health * rbf_feature * caputo_weight.
    """
    health = recovery_priority(morph)               # scalar in [0,1]
    rbf_feat = rbf.predict(x_input)                # scalar (may be >1)
    caputo = caputo_weight(alpha)                  # scalar >0
    return health * rbf_feat * caputo

def hybrid_state_step(
    state: np.ndarray,
    input_vec: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    One hybrid update:
        y = γ * tropical_matvec(A, state) ⊕ γ * tropical_matvec(B, input_vec)
    where ⊕ is tropical addition (max) and γ scales the result in the ordinary
    (real) domain before re‑entering tropical space via log‑exp trick.
    """
    # Tropical products
    a_term = tropical_matvec(A, state)
    b_term = tropical_matvec(B, input_vec)

    # Scale by γ in ordinary arithmetic, then convert back to tropical by
    # taking logarithm of the positive values.  To stay within max‑plus algebra
    # we add log(γ) to the tropical values (since tropical multiplication ↔ +).
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    log_gamma = math.log(gamma)

    a_scaled = a_term + log_gamma
    b_scaled = b_term + log_gamma

    # Tropical addition (max) of the two contributions
    next_state = np.maximum(a_scaled, b_scaled)
    return next_state

def hybrid_ssm_sequential(
    init_state: np.ndarray,
    inputs: List[np.ndarray],
    A: np.ndarray,
    B: np.ndarray,
    morph: Morphology,
    rbf: RBFSurrogate,
    alpha: float,
) -> List[np.ndarray]:
    """
    Run a sequence of hybrid steps, returning the list of states.
    """
    states = [init_state]
    state = init_state.copy()
    for inp in inputs:
        gamma = hybrid_compute_gamma(morph, rbf, inp.tolist(), alpha)
        state = hybrid_state_step(state, inp, A, B, gamma)
        states.append(state.copy())
    return states

def hybrid_ssm_parallel(
    init_states: np.ndarray,
    inputs: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    morphs: List[Morphology],
    rbf: RBFSurrogate,
    alphas: np.ndarray,
) -> np.ndarray:
    """
    Parallel batch update.
    * init_states : (batch, n) matrix of tropical states
    * inputs      : (batch, m) matrix of inputs
    * morphs      : list of Morphology objects, length = batch
    * alphas      : (batch,) array of fractional orders
    Returns updated states of shape (batch, n).
    """
    batch, n = init_states.shape
    _, m = inputs.shape
    if len(morphs) != batch or alphas.shape[0] != batch:
        raise ValueError("batch size mismatch")
    updated = np.empty_like(init_states)
    for i in range(batch):
        gamma = hybrid_compute_gamma(
            morphs[i],
            rbf,
            inputs[i].tolist(),
            float(alphas[i]),
        )
        updated[i] = hybrid_state_step(init_states[i], inputs[i], A, B, gamma)
    return updated

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic data
    np.random.seed(0)
    random.seed(0)

    # Morphology instance
    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=3.0)

    # RBF surrogate with random centers and weights
    dim = 3
    centers = [tuple(np.random.rand(dim)) for _ in range(5)]
    weights = [random.uniform(-1.0, 1.0) for _ in range(5)]
    rbf = RBFSurrogate(centers=centers, weights=weights, epsilon=1.2)

    # Tropical state‑space matrices (small sizes for demo)
    n_state = 4
    n_input = dim
    A = np.random.randn(n_state, n_state)   # real numbers; tropical uses them as “weights”
    B = np.random.randn(n_state, n_input)

    # Initial tropical state (log‑domain values)
    init_state = np.full(n_state, 0.0)  # log(1)=0

    # Input sequence
    inputs = [np.random.rand(n_input) for _ in range(3)]

    # Fractional order
    alpha = 0.4

    # Run sequential hybrid SSM
    states = hybrid_ssm_sequential(init_state, inputs, A, B, morph, rbf, alpha)
    print("Sequential states:")
    for idx, s in enumerate(states):
        print(f" step {idx}: {s}")

    # Parallel batch test
    batch = 2
    init_states = np.stack([init_state, init_state])
    input_batch = np.stack([inputs[0], inputs[1]])
    morphs = [morph, morph]
    alphas = np.array([0.3, 0.6])
    updated_batch = hybrid_ssm_parallel(init_states, input_batch, A, B, morphs, rbf, alphas)
    print("\nParallel batch updated states:")
    print(updated_batch)