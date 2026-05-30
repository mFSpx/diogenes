# DARWIN HAMMER — match 444, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_caputo_fracti_hybrid_geometric_pro_m291_s0.py (gen3)
# parent_b: serpentina_self_righting.py (gen0)
# born: 2026-05-29T23:29:01Z

"""Hybrid Caputo‑Geometric Morphology (HCGM) algorithm.

This module fuses:

* **Hybrid Caputo Fractional derivative** (parent A) – provides a power‑law memory
  kernel via the Caputo fractional derivative.
* **Geometric Product rotor** (parent A) – rotates vectors in a Clifford‑algebra
  sense.
* **Morphology right‑ing metrics** (parent B) – flatness, sphericity and
  right‑ing time of a turtle‑like body.

**Mathematical bridge**  
The fractional‑derivative kernel ϕ(t;α) is interpreted as a scalar weight that
parameterises a GA‑rotor.  The rotor R(θ) = exp(θ B) (B a bivector) rotates the
four‑dimensional morphology vector **v** = (length, width, height, mass).  The
angle θ is obtained by aggregating the Caputo weights, thus embedding long‑range
temporal memory into the geometric transformation that feeds the right‑ing
time formula.  The resulting rotated morphology is then evaluated with the
original biological equations, yielding a hybrid recovery priority that accounts
for both history‑dependent dynamics and shape‑dependent physics.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np
from math import gamma

# ----------------------------------------------------------------------
# Parent‑A utilities (Caputo derivative & rotor)
# ----------------------------------------------------------------------
def lanczos_gamma(z: float) -> float:
    """Lanczos approximation of Γ(z) for z > 0."""
    if z < 0.5:
        return gamma(1 - z) * gamma(z) / math.sin(math.pi * z)
    g = 7
    z += g + 0.5
    term = 1.0
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    for c in p:
        term *= (z + c) / (z - c)
    return math.sqrt(2 * math.pi) * (z ** (z - 0.5)) * math.exp(-z) * term


def caputo_derivative(f: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    """
    Discrete Caputo fractional derivative of order `alpha` (0<α<1)
    for a sampled signal `f(t)`.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    if len(f) != len(t):
        raise ValueError("f and t must have the same length")
    dt = np.diff(t, prepend=t[0])
    df = np.diff(f, prepend=f[0])
    # Kernel dt^{-α}
    kernel = dt ** (-alpha)
    integral = np.dot(df, kernel) / lanczos_gamma(1 - alpha)
    # Return an array matching the input length (first element zero by definition)
    out = np.empty_like(f, dtype=float)
    out[0] = 0.0
    out[1:] = integral
    return out


def apply_rotor(R: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rotate Euclidean vector `x` with orthogonal matrix `R`."""
    return R @ x


def rotation_matrix_4d(theta: float) -> np.ndarray:
    """
    Simple 4‑D rotation matrix that rotates in the (0,1) plane by angle `theta`
    and leaves the remaining dimensions unchanged.
    """
    c, s = math.cos(theta), math.sin(theta)
    R = np.eye(4)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


def compute_rotor_from_weights(weights: np.ndarray, eta_r: float = 1.0) -> np.ndarray:
    """
    Convert a vector of Caputo‑derived weights into a GA rotor.
    The scalar angle is taken as the weighted sum of the memory kernel.
    """
    if weights.ndim != 1:
        raise ValueError("weights must be a 1‑D array")
    theta = eta_r * np.sum(weights)
    return rotation_matrix_4d(theta)


# ----------------------------------------------------------------------
# Parent‑B utilities (Morphology metrics)
# ----------------------------------------------------------------------
from dataclasses import dataclass


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
    """0..1 priority for scheduler rescue/rollback assistance."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_morphology_vector(m: Morphology) -> np.ndarray:
    """Pack a Morphology into a 4‑D Euclidean vector."""
    return np.array([m.length, m.width, m.height, m.mass], dtype=float)


def hybrid_righting_series(
    morphologies: list[Morphology],
    times: np.ndarray,
    alpha: float = 0.5,
    eta_r: float = 0.1,
) -> list[float]:
    """
    Compute a time‑series of right‑ing times where each step is influenced by
    the fractional‑memory kernel of the whole history.

    Steps:
    1. Build a scalar signal `f(t)` = mass(t) (any monotonic observable works).
    2. Obtain Caputo weights w = D^α f(t).
    3. Convert w → rotor R using `compute_rotor_from_weights`.
    4. Rotate each morphology vector with R and evaluate the biological right‑ing
       formula on the rotated geometry.
    """
    if len(morphologies) != len(times):
        raise ValueError("morphologies and times must have equal length")
    # 1. Signal for memory – use mass as a simple proxy
    mass_signal = np.array([m.mass for m in morphologies], dtype=float)
    # 2. Fractional derivative weights
    weights = caputo_derivative(mass_signal, times, alpha)
    # 3. Rotor from aggregated memory
    R = compute_rotor_from_weights(weights, eta_r)

    # 4. Apply rotor and compute right‑ing times
    righting_series = []
    for m in morphologies:
        v = hybrid_morphology_vector(m)
        v_rot = apply_rotor(R, v)
        # Map rotated vector back to a Morphology (keep physical sense)
        m_rot = Morphology(
            length=max(v_rot[0], 1e-6),
            width=max(v_rot[1], 1e-6),
            height=max(v_rot[2], 1e-6),
            mass=max(v_rot[3], 1e-6),
        )
        rt = righting_time_index(m_rot)
        righting_series.append(rt)
    return righting_series


def hybrid_recovery_priority_series(
    morphologies: list[Morphology],
    times: np.ndarray,
    alpha: float = 0.5,
    eta_r: float = 0.1,
    max_index: float = 10.0,
) -> list[float]:
    """
    Same pipeline as `hybrid_righting_series` but maps the result to a
    normalized priority in [0,1] using the original `recovery_priority`
    formulation.
    """
    righting = hybrid_righting_series(morphologies, times, alpha, eta_r)
    # Convert each right‑ing time to a priority
    priorities = [max(0.0, min(1.0, rt / max_index)) for rt in righting]
    return priorities


def hybrid_caputo_geometric_step(
    W: np.ndarray,
    R: np.ndarray,
    x: np.ndarray,
    eta_w: float = 0.01,
    eta_r: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One learning‑style update that mirrors the unfinished `ttt_ga_forward` from
    parent A, now enriched with the hybrid morphology context.

    - Update weight matrix `W` with a gradient proportional to the bivector
      `x ∧ (W x - x)`.
    - Update rotor `R` with an infinitesimal rotation generated by the same bivector.
    """
    # Linear transform
    y = W @ x
    # Bivector generator (outer product antisymmetrized)
    biv = np.outer(x, y) - np.outer(y, x)
    # Gradient step on W
    W_new = W - eta_w * biv
    # Exponential map for rotor update (first‑order approximation)
    R_new = R @ (np.eye(R.shape[0]) + eta_r * biv)
    # Re‑orthogonalize rotor (Gram‑Schmidt on columns)
    q, _ = np.linalg.qr(R_new)
    return W_new, q


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a simple time grid
    times = np.linspace(0, 5, 6)  # 0,1,2,3,4,5

    # Generate a list of morphologies with gradually increasing mass
    morphologies = [
        Morphology(length=5.0 - 0.2 * i,
                   width=3.0 - 0.1 * i,
                   height=2.0 + 0.05 * i,
                   mass=1.0 + 0.5 * i)
        for i in range(len(times))
    ]

    # Run hybrid right‑ing series
    rt_series = hybrid_righting_series(morphologies, times, alpha=0.6, eta_r=0.05)
    print("Hybrid right‑ing time series:")
    for t, rt in zip(times, rt_series):
        print(f" t={t: .1f} → RT={rt:.4f}")

    # Run hybrid priority series
    pr_series = hybrid_recovery_priority_series(morphologies, times, alpha=0.6, eta_r=0.05)
    print("\nHybrid recovery priority series:")
    for t, pr in zip(times, pr_series):
        print(f" t={t: .1f} → Priority={pr:.3f}")

    # Demonstrate a single Caputo‑Geometric update
    x0 = np.array([1.0, 0.5, 0.2, 2.0])
    W0 = np.eye(4) * 0.9
    R0 = np.eye(4)
    W1, R1 = hybrid_caputo_geometric_step(W0, R0, x0)
    print("\nUpdated weight matrix W1:")
    print(W1)
    print("\nUpdated rotor R1 (orthogonal):")
    print(R1)