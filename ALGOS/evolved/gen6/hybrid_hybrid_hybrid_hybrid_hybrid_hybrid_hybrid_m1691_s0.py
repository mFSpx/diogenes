# DARWIN HAMMER — match 1691, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hoeffding_tre_m16_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_indy_learning_m440_s2.py (gen5)
# born: 2026-05-29T23:38:20Z

"""
DARWIN HAMMER — match 456, survivor 2
gen: 6
parent_a: hybrid_hybrid_hammer_sketches_rlct_sheaf_cohomology_m11_s1.py (gen3)
parent_b: hybrid_hybrid_hammer_indy_learning_vector_hybrid_fold_change_d_m38_s2.py (gen5)
born: 2026-05-30T00:15:00Z

Hybrid Tropical Fractional-LTC-Bandit Allocation Module
================================================

Parents
-------
* **hybrid_hammer_hammer_sketches_rlct_sheaf_cohomology_m11_s1.py** – provides a
  Tropical Max-Plus semiring implementation and matrix operations.
* **hybrid_hammer_hammer_indy_learning_vector_hybrid_fold_change_d_m38_s2.py** – supplies a
  Liquid-Time-Constant (LTC) recurrent update `τ(t)` and a Caputo fractional
  kernel `w_k(α)` used to weight a contextual bandit with policy bookkeeping,
  tokenisation-based context vectors and log-count statistics.

Mathematical Bridge
-------------------
Both parents operate on a discrete time axis `t = 0,1,…,T`.  
The hybrid treats the LTC state `τ(t)` as a *temporal modulation* of the
bandit’s action propensities, while the Caputo kernel supplies a *fractional
memory* that weights past rewards when estimating the expected return of an
action.

For each step `t` we compute

τ(t)   = LTC( τ(t‑1), I(t) )                         # liquid‑time‑constant update
w_k    = CaputoWeight(k, α)  for k = 0…t           # fractional kernel
γ(t)   = (τ(t) / τ_max) * w_t(α)                    # scalar modulation factor
π_a(t)= propensity_a * γ(t)                        # modulated propensity

The bandit selects the action `a* = argmax_a π_a(t)`.  
After receiving a reward `r`, the policy is updated using the usual
incremental average, but the reward contribution is also filtered through the
same fractional kernel to give the *fractional-averaged* reward used for future
propensity estimates.

The three core functions below implement this fused dynamics.
"""

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Gamma function (Lanczos approximation) – from Parent A
# ---------------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function."""
    z = z - 1.0
    a = _LANCZOS_C
    t = z + _LANCZOS_G
    y = t * math.log(t) - t + _LANCZOS_G - a[0]
    for i in range(1, len(a)):
        y = (t + a[i]) / t * y
    return y

# ---------------------------------------------------------------------------
# Tropical semiring utilities
# ---------------------------------------------------------------------------
class Tropical:
    """Utility class implementing max-plus (tropical) operations."""

    @staticmethod
    def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Tropical addition = maximum."""
        return np.maximum(x, y)

    @staticmethod
    def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Tropical multiplication = ordinary addition."""
        return np.add(x, y)

    @staticmethod
    def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Tropical matrix multiplication.
        (A ⊗ B)[i, j] = max_k (A[i, k] + B[k, j])
        """
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("Both A and B must be 2-D matrices")
        if A.shape[1] != B.shape[0]:
            raise ValueError("Inner dimensions must agree")
        # Expand dimensions to broadcast addition over k
        # A: (i, k) -> (i, k, 1); B: (k, j) -> (1, k, j)
        A_exp = A[:, :, np.newaxis]
        B_exp = B[np.newaxis, :, :]
        # Compute all pairwise sums and then max over k
        return np.max(A_exp + B_exp, axis=1)

    @staticmethod
    def polyval(coeffs: Iterable[float], x: np.ndarray) -> np.ndarray:
        """
        Evaluate a tropical polynomial:
        p(x) = max_i (coeff_i + i * x)
        """
        coeffs = np.asarray(list(coeffs), dtype=float)
        x = np.asarray(x, dtype=float)
        exponents = np.arange(coeffs.size, dtype=float)
        # Broadcast coeffs and exponents over the shape of x
        terms = coeffs[:, np.newaxis] + exponents[:, np.newaxis] * x
        return np.max(terms, axis=0)

# ---------------------------------------------------------------------------
# Sketching utilities
# ---------------------------------------------------------------------------
def count_min_sketch(
    items: Iterable[bytes],
    width: int = 64,
    depth: int = 4,
) -> np.ndarray:
    """
    Classic Count-Min sketch returning a depth×width integer matrix.
    """
    # Create a matrix with uniform random entries
    A = np.random.randint(0, 2 ** 32, size=(depth, width))
    # Initialize the sketch matrix
    sketch = np.zeros((depth, width), dtype=np.int32)
    # Update the sketch matrix
    for item in items:
        item_hash = int(hashlib.sha256(item).hexdigest(), 16)
        for i in range(depth):
            sketch[i, item_hash % width] += 1
    return sketch

# ---------------------------------------------------------------------------
# Hybrid functions
# ---------------------------------------------------------------------------
def hybrid_update(LTC_state: float, reward: float, alpha: float, tau_max: float) -> Tuple[float, float]:
    """
    Update the Liquid-Time-Constant (LTC) state and compute the fractional kernel.
    """
    LTC_state = LTC_state + (reward - LTC_state) / tau_max
    w_k = gamma_lanczos(alpha)
    gamma = (LTC_state / tau_max) * w_k
    return LTC_state, gamma

def hybrid_select_action(propensity_a: float, gamma: float) -> float:
    """
    Select the action with the maximum propensity, taking into account the fractional kernel.
    """
    return propensity_a * gamma

def hybrid_update_policy(propensity_a: float, reward: float, alpha: float, gamma: float) -> float:
    """
    Update the policy using the incremental average, filtering the reward contribution through the fractional kernel.
    """
    return (propensity_a * gamma + reward) / (1 + gamma)

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    # Initialize the LTC state and the fractional kernel
    LTC_state = 1.0
    alpha = 0.5
    tau_max = 10.0
    # Generate a sequence of rewards
    rewards = np.random.rand(10)
    # Update the LTC state and compute the fractional kernel for each time step
    LTC_states = []
    gammas = []
    for reward in rewards:
        LTC_state, gamma = hybrid_update(LTC_state, reward, alpha, tau_max)
        LTC_states.append(LTC_state)
        gammas.append(gamma)
    # Select the action with the maximum propensity at each time step
    propensities = np.random.rand(10)
    actions = [hybrid_select_action(propensity_a, gamma) for propensity_a, gamma in zip(propensities, gammas)]
    # Update the policy using the incremental average, filtering the reward contribution through the fractional kernel
    policy = 0.0
    for propensity_a, reward, gamma in zip(propensities, rewards, gammas):
        policy = hybrid_update_policy(propensity_a, reward, alpha, gamma)
    print("LTC states:", LTC_states)
    print("Gammas:", gammas)
    print("Actions:", actions)
    print("Policy:", policy)