# DARWIN HAMMER — match 1632, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m824_s1.py (gen5)
# parent_b: hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s2.py (gen3)
# born: 2026-05-29T23:38:00Z

"""Hybrid Algorithm combining:
- Parent A (store dynamics, sigmoid, variational free energy)
- Parent B (tropical max‑plus algebra)

Mathematical bridge:
The tropical polynomial p(x)=max_i (c_i + i·x) yields a piecewise‑linear
“energy landscape”.  By feeding this landscape through a sigmoid we obtain
a bounded probability‑like vector q(x).  The StoreState dynamics treat q(x)
as an inflow (or outflow) and update a scalar “level”.  The updated level
is then used as a temperature parameter for a second sigmoid that defines
a reference distribution p_ref.  The variational free energy
VFE(q‖p_ref)=∑ q·log(q/p_ref) measures the mismatch between the tropical‑
induced belief and the StoreState‑driven belief.  This coupling fuses the
max‑plus algebraic structure with the Bayesian‑style free‑energy update of
Parent A.

The module provides three high‑level hybrid operations:
1. `tropical_sigmoid_transform` – evaluates a tropical polynomial and
   maps the result to (0,1) via a sigmoid.
2. `hybrid_state_update` – updates a `StoreState` using inflow/outflow
   derived from tropical‑sigmoid vectors.
3. `hybrid_variational_energy` – computes the variational free energy between
   the tropical‑sigmoid distribution and a reference distribution generated
   from the StoreState level.

All functions are pure NumPy and standard‑library code, satisfying the
fusion requirements.
"""

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Standard leaky‑integrator update."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """A bounded gain‑modulated signal derived from the last delta."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    # Clip to avoid overflow in exp
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def extract_master_vector(text: str, dim: int = 12) -> np.ndarray:
    """Deterministic pseudo‑random vector derived from a string."""
    seed = _hash(123, text)
    random.seed(seed)
    return np.array([random.random() for _ in range(dim)], dtype=float)

def variational_free_energy(q: np.ndarray, p: np.ndarray) -> float:
    """
    Variational (Kullback‑Leibler) free energy:  ∑ q·log(q/p)
    Both inputs must be non‑negative and sum to 1 (probability vectors).
    """
    epsilon = 1e-15
    q = np.clip(q, epsilon, 1.0)
    p = np.clip(p, epsilon, 1.0)
    q = q / q.sum()
    p = p / p.sum()
    return float(np.sum(q * np.log(q / p)))

# ----------------------------------------------------------------------
# Parent B building blocks (tropical max‑plus algebra)
# ----------------------------------------------------------------------
def t_add(x, y):
    """Tropical addition: max(x, y) – element‑wise."""
    return np.maximum(x, y)

def t_mul(x, y):
    """Tropical multiplication: x + y – element‑wise."""
    return np.add(x, y)

def t_matmul(A, B):
    """
    Tropical matrix multiplication.
    C[i, j] = max_k (A[i, k] + B[k, j])
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # Expand dimensions for broadcasting: (m, p, 1) + (1, p, n)
    tmp = A[:, :, np.newaxis] + B[np.newaxis, :, :]
    return np.max(tmp, axis=1)

def t_polyval(coeffs: np.ndarray, x):
    """
    Evaluate a tropical polynomial:
        p(x) = max_i (coeffs[i] + i * x)
    coeffs – 1‑D array of length d+1 (coeff for x^i).
    x – scalar or array broadcastable with coeffs.
    Returns same shape as x.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)

    # exponents 0,1,…,d
    exponents = np.arange(coeffs.shape[0], dtype=float)

    # Reshape for broadcasting: (d+1, 1,…,1) + (1,…,1) * exponents
    term_shape = (coeffs.shape[0],) + (1,) * x.ndim
    terms = coeffs.reshape(term_shape) + exponents[:, np.newaxis] * x
    return np.max(terms, axis=0)

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def tropical_sigmoid_transform(coeffs: np.ndarray, x) -> np.ndarray:
    """
    1. Evaluate tropical polynomial p(x).
    2. Apply a sigmoid to map the result into (0,1).
    3. Normalise to obtain a probability vector (sums to 1).

    Parameters
    ----------
    coeffs : np.ndarray
        Tropical coefficients.
    x : scalar or array‑like
        Evaluation points.

    Returns
    -------
    np.ndarray
        Normalised probability vector of the same shape as x.
    """
    raw = t_polyval(coeffs, x)
    prob = sigmoid(raw)
    prob_sum = prob.sum()
    if prob_sum == 0:
        # fallback to uniform distribution
        return np.full_like(prob, 1.0 / prob.size, dtype=float)
    return prob / prob_sum

def hybrid_state_update(state: StoreState,
                        inflow_coeffs: np.ndarray,
                        outflow_coeffs: np.ndarray,
                        x) -> Tuple[float, float]:
    """
    Update a StoreState where:
    - inflow is the tropical‑sigmoid distribution derived from `inflow_coeffs`.
    - outflow is the same derived from `outflow_coeffs`.
    The level after the update is returned together with the raw delta.

    Parameters
    ----------
    state : StoreState
        The mutable state object.
    inflow_coeffs, outflow_coeffs : np.ndarray
        Coefficient vectors for the two tropical polynomials.
    x : scalar or array‑like
        Input to the tropical polynomials.

    Returns
    -------
    Tuple[float, float]
        (new_level, delta) after the update.
    """
    inflow_vec = tropical_sigmoid_transform(inflow_coeffs, x)
    outflow_vec = tropical_sigmoid_transform(outflow_coeffs, x)
    level, delta = state.update(inflow_vec.tolist(), outflow_vec.tolist())
    return level, delta

def hybrid_variational_energy(state: StoreState,
                              tropical_coeffs: np.ndarray,
                              x) -> float:
    """
    Compute the variational free energy between:
    - q(x): tropical‑sigmoid distribution from `tropical_coeffs`.
    - p_ref: a reference distribution generated from the current StoreState level
      via a temperature‑scaled sigmoid.

    The temperature τ = 1 + state.dance modulates the sharpness of p_ref.

    Parameters
    ----------
    state : StoreState
        Current state providing the level and dance.
    tropical_coeffs : np.ndarray
        Coefficients for the tropical polynomial.
    x : scalar or array‑like
        Evaluation points.

    Returns
    -------
    float
        Variational free energy VFE(q‖p_ref).
    """
    q = tropical_sigmoid_transform(tropical_coeffs, x)

    # Reference distribution: sigmoid of (level * x) with temperature τ
    tau = 1.0 + state.dance
    raw_ref = sigmoid(state.level * np.asarray(x, dtype=float) / tau)
    p_ref = raw_ref / raw_ref.sum()
    return variational_free_energy(q, p_ref)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)

    # Create a StoreState instance
    store = StoreState(level=0.5, alpha=0.8, beta=0.3, dt=0.1, base=0.2, gain=0.5, limit=5.0)

    # Tropical coefficients (random but reproducible)
    inflow_coeffs = np.array([0.0, -1.2, 0.5, -0.3])   # degree 3
    outflow_coeffs = np.array([-0.5, 0.8, -0.2, 0.1])  # degree 3
    eval_points = np.linspace(-2, 2, 9)               # 9 points

    # Perform a hybrid update
    lvl, dlt = hybrid_state_update(store, inflow_coeffs, outflow_coeffs, eval_points)
    print(f"Updated level: {lvl:.4f}, delta: {dlt:.4f}")

    # Compute variational free energy with a new set of coefficients
    tropical_coeffs = np.array([0.2, -0.4, 0.6, -0.1])
    vfe = hybrid_variational_energy(store, tropical_coeffs, eval_points)
    print(f"Variational free energy: {vfe:.6f}")

    # Verify that the StoreState dance stays within limits
    assert 0.0 <= store.dance <= store.limit, "Dance out of bounds"

    # Simple sanity check: probabilities sum to 1
    prob = tropical_sigmoid_transform(tropical_coeffs, eval_points)
    assert np.isclose(prob.sum(), 1.0), "Probability vector not normalised"

    print("Smoke test completed successfully.")