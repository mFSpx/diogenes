# DARWIN HAMMER — match 3781, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_caputo_m1313_s0.py (gen6)
# born: 2026-05-29T23:51:33Z

"""Hybrid NLMS‑Rectified‑Flow with Caputo‑Fractional Store and MinHash Similarity.

Parent A:
- NLMS adaptive filter predicting a scalar from the straight‑line interpolant
  zₜ = t·x₁ + (1‑t)·x₀.
- Weight update: w←w+μ·e·x / (‖x‖²+ε).

Parent B:
- StoreState that integrates inflow/outflow to produce a bounded “dance”
  signal.
- Caputo fractional‑order memory of past errors (weighted sum with exponent α).
- MinHash signatures to obtain a similarity metric between feature vectors.

Mathematical bridge:
The NLMS error e(t) is fed as inflow to the StoreState, whose level
modulates the effective learning‑rate μ̂(t)=μ·(1+gain·sim(zₜ,Π)), where
sim(·) is a MinHash‑based similarity between the current interpolant
zₜ and a set of stored prototypes Π.  The Caputo fractional memory
produces a smoothed error ė(t) that replaces the raw error in the NLMS
update, thus coupling the adaptive filter with the fractional store
dynamics in a single unified system.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Callable

# ----------------------------------------------------------------------
# MinHash utilities (deterministic simple implementation)
# ----------------------------------------------------------------------
_NUM_PERMUTATIONS = 64
_random_seeds = [random.randint(1, 2**31 - 1) for _ in range(_NUM_PERMUTATIONS)]

def _hash_component(value: float, seed: int) -> int:
    """Hash a single float component with a given seed."""
    # Convert float to its binary representation and mix with seed
    bits = np.float64(value).view(np.int64)
    return hash((bits, seed))

def minhash_signature(vec: np.ndarray) -> np.ndarray:
    """Return a MinHash signature (uint64 array) for a real‑valued vector."""
    sig = np.empty(_NUM_PERMUTATIONS, dtype=np.uint64)
    for i, seed in enumerate(_random_seeds):
        # The minimum hash over all components defines the i‑th signature entry
        hashes = [_hash_component(v, seed) for v in vec]
        sig[i] = min(hashes)
    return sig

def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Jaccard‑like similarity between two MinHash signatures."""
    if sig1.shape != sig2.shape:
        raise ValueError("Signature shapes must match")
    return float(np.mean(sig1 == sig2))

# ----------------------------------------------------------------------
# Caputo fractional memory (simple discrete approximation)
# ----------------------------------------------------------------------
def caputo_weighted_sum(errors: List[float], alpha: float) -> float:
    """
    Approximate the Caputo fractional derivative of order α (0<α<1)
    using a weighted sum of past errors.

    D^α e_n ≈ (1/Γ(1‑α)) * Σ_{k=0}^{n-1} (e_{k+1}‑e_k) / (Δt)^{α}
    For uniform Δt=1 this reduces to a power‑law weighting.
    """
    if not errors:
        return 0.0
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    n = len(errors)
    coeff = 1.0 / math.gamma(1.0 - alpha)
    weighted = 0.0
    for k in range(1, n):
        weight = (k) ** (-alpha)  # power‑law decay
        weighted += (errors[k] - errors[k - 1]) * weight
    return coeff * weighted

# ----------------------------------------------------------------------
# StoreState (honeybee‑style store)
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0          # inflow coefficient
    beta: float = 1.0           # outflow coefficient
    dt: float = 1.0
    gain: float = 1.0           # amplifies similarity effect
    limit: float = 10.0         # max level (for bounding)
    prototypes: List[np.ndarray] = field(default_factory=list)

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Integrate inflow/outflow and clamp the level."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, min(self.limit, self.level + self.dt * delta))
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ."""
        delta = getattr(self, "_last_delta", 0.0)
        # Simple sigmoid to keep the signal in [0, limit]
        return self.limit / (1.0 + math.exp(-delta))

    def add_prototype(self, vec: np.ndarray) -> None:
        """Store a new prototype (if not already present)."""
        self.prototypes.append(vec.copy())

    def max_similarity(self, vec: np.ndarray) -> float:
        """Maximum MinHash similarity between vec and stored prototypes."""
        if not self.prototypes:
            return 0.0
        sig = minhash_signature(vec)
        sims = [minhash_similarity(sig, minhash_signature(p)) for p in self.prototypes]
        return max(sims)

# ----------------------------------------------------------------------
# NLMS + Rectified Flow core (parent A)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot‑product prediction w·x."""
    return float(weights @ x)

def interpolant(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    """Straight‑line interpolant Zₜ = t·x₁ + (1‑t)·x₀."""
    return t * x1 + (1.0 - t) * x0

def flow_target(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Target vector field v = x₁ − x₀."""
    return x1 - x0

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """Standard NLMS weight update."""
    error = target - nlms_predict(weights, x)
    update = mu * error * x / (np.linalg.norm(x) ** 2 + eps)
    return weights + update, error

# ----------------------------------------------------------------------
# Hybrid operations (fusion of A & B)
# ----------------------------------------------------------------------
def hybrid_predict(
    weights: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    t: float,
    store: StoreState,
) -> float:
    """
    Predict using NLMS on the interpolant, then blend with the store's
    dance signal weighted by similarity to stored prototypes.
    """
    z_t = interpolant(x0, x1, t)
    base_pred = nlms_predict(weights, z_t)

    # similarity between current interpolant and prototypes
    sim = store.max_similarity(z_t)

    # blend: when similarity is high, trust the store's dance signal more
    blended = (1.0 - sim) * base_pred + sim * store.dance
    return blended

def hybrid_update(
    weights: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    t: float,
    store: StoreState,
    error_history: List[float],
    mu_base: float = 0.5,
    eps: float = 1e-9,
    alpha_frac: float = 0.7,
) -> Tuple[np.ndarray, float, StoreState]:
    """
    Perform a fused update:
    1. Compute target flow and NLMS prediction.
    2. Compute fractional‑memory error (Caputo weighted sum).
    3. Modulate learning rate with similarity·gain.
    4. NLMS weight update using the modulated μ and fractional error.
    5. Update the StoreState with |error| as inflow and a small outflow.
    6. Store the current interpolant as a prototype for future similarity.
    """
    # 1. target and raw prediction
    target_vec = flow_target(x0, x1)
    target = float(np.linalg.norm(target_vec))  # scalar magnitude as target
    z_t = interpolant(x0, x1, t)

    # 2. fractional‑memory error
    raw_error = target - nlms_predict(weights, z_t)
    error_history.append(raw_error)
    frac_error = caputo_weighted_sum(error_history, alpha_frac)

    # 3. similarity‑driven learning‑rate modulation
    sim = store.max_similarity(z_t)
    mu_eff = mu_base * (1.0 + store.gain * sim)

    # 4. NLMS update with fractional error as target correction
    #    We treat frac_error as an additional bias term.
    corrected_target = target + frac_error
    weights, nlms_err = nlms_update(weights, z_t, corrected_target, mu=mu_eff, eps=eps)

    # 5. Store dynamics
    inflow = [abs(nlms_err)]
    outflow = [store.level * 0.05]  # gentle leakage proportional to current level
    store.update(inflow, outflow)

    # 6. Add current interpolant as a prototype (optional throttling)
    if len(store.prototypes) < 20:  # keep a modest memory
        store.add_prototype(z_t)

    return weights, nlms_err, store

def run_hybrid_demo(steps: int = 10) -> None:
    """Simple demonstration of the fused algorithm."""
    dim = 5
    rng = np.random.default_rng(42)
    # Initialise NLMS weights randomly
    weights = rng.standard_normal(dim)
    # Initialise StoreState
    store = StoreState(gain=0.8, limit=5.0)
    # History of fractional errors
    error_hist: List[float] = []

    for i in range(steps):
        # Random start/end points and interpolation factor
        x0 = rng.standard_normal(dim)
        x1 = rng.standard_normal(dim)
        t = rng.random()
        # Hybrid prediction (just for logging)
        pred = hybrid_predict(weights, x0, x1, t, store)
        # Hybrid update
        weights, err, store = hybrid_update(
            weights, x0, x1, t, store, error_hist, mu_base=0.4, alpha_frac=0.6
        )
        print(
            f"Step {i+1:02d}: t={t:.3f}, pred={pred:.4f}, err={err:.4f}, "
            f"store_level={store.level:.3f}, mu_eff≈{store.gain * store.max_similarity(interpolant(x0, x1, t)) + 0.4:.3f}"
        )

if __name__ == "__main__":
    run_hybrid_demo(steps=15)