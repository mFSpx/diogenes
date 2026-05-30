# DARWIN HAMMER — match 60, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_state_space_duality_m143_s0.py (gen2)
# parent_b: state_space_duality.py (gen0)
# born: 2026-05-29T23:26:38Z

"""
Hybrid Poikilotherm‑State‑Space Duality (HPSSD)

This module fuses two parent algorithms:

* **Parent A** – the Schoolfield‑Rollinson poikilotherm developmental‑rate primitive.
* **Parent B** – the State‑Space Duality (SSD) sequential and semiseparable parallel forms.

The mathematical bridge is the temperature‑dependent scalar
`r(t) = developmental_rate(T(t))` which modulates the state‑transition matrix `A`
in the SSD.  By scaling `A` with `r(t)` at each time step we obtain a
time‑varying state‑space that respects the underlying thermodynamic kinetics.
Consequently the semiseparable parallel matrix `M` is built from the same
temperature‑scaled `A_t`, preserving the exact duality between the sequential
scan and the dense matrix‑multiply formulation while embedding the poikilotherm
physics.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import numpy as np

__all__ = [
    "SchoolfieldParams",
    "developmental_rate",
    "temperature_dependent_state_transition",
    "hybrid_ssm_step",
    "hybrid_ssm_sequential",
    "hybrid_semiseparable_matrix",
    "verify_hybrid_duality",
]


# ----------------------------------------------------------------------
# Parent‑A: Poikilotherm developmental rate
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C (in arbitrary units)
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15            # K  (10 °C)
    t_high: float = 307.15           # K  (34 °C)
    delta_h_low: float = -45_000.0   # J mol⁻¹
    delta_h_high: float = 65_000.0   # J mol⁻¹
    r_cal: float = 1.987             # cal mol⁻¹ K⁻¹ (≈8.314 J mol⁻¹ K⁻¹)


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield‑Rollinson temperature‑dependent developmental rate.

    Parameters
    ----------
    temp_k : float
        Absolute temperature in Kelvin (must be > 0).
    params : SchoolfieldParams, optional
        Model parameters.

    Returns
    -------
    rate : float
        Dimensionless rate factor; 1.0 at 25 °C when `rho_25` = 1.
    """
    if temp_k <= 0:
        raise ValueError("temperature must be Kelvin‑positive")
    if params.rho_25 < 0:
        raise ValueError("rho_25 must be non‑negative")

    # Core Schoolfield equation (calorie‑based constants converted to J)
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp(
        (params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k))
    )
    high = math.exp(
        (params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k))
    )
    return numerator / (1.0 + low + high)


def temperature_dependent_state_transition(
    base_A: np.ndarray,
    temp_k: float,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> np.ndarray:
    """
    Scale a base state‑transition matrix `A` by the poikilotherm rate.

    The scaling is a simple scalar multiplication because the developmental
    rate is dimensionless and applies uniformly to all state‑space dynamics.

    Parameters
    ----------
    base_A : np.ndarray, shape (n, n)
        Baseline (temperature‑independent) transition matrix.
    temp_k : float
        Current temperature in Kelvin.
    params : SchoolfieldParams, optional
        Parameters for the developmental‑rate model.

    Returns
    -------
    A_t : np.ndarray, shape (n, n)
        Temperature‑adjusted transition matrix for the current time step.
    """
    rate = developmental_rate(temp_k, params)
    return base_A * rate


# ----------------------------------------------------------------------
# Parent‑B: State‑Space Duality (sequential & semiseparable parallel)
# ----------------------------------------------------------------------
def ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single sequential SSM step (identical to Parent B).

    Returns the new hidden state and the output.
    """
    h_new = A @ h + B @ x
    y = C @ h_new
    return h_new, y


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    base_A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_k: float,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sequential SSM step where the transition matrix is temperature‑scaled.

    This is the core hybrid operation: the poikilotherm rate modulates the
    dynamics before the usual linear update.

    Parameters
    ----------
    h, x, base_A, B, C : as in `ssm_step`
    temp_k : float
        Current temperature (K) used to compute the scaling factor.
    params : SchoolfieldParams, optional
        Rate‑model parameters.

    Returns
    -------
    h_new, y : tuple of ndarrays
    """
    A_t = temperature_dependent_state_transition(base_A, temp_k, params)
    return ssm_step(h, x, A_t, B, C)


def hybrid_ssm_sequential(
    x_seq: np.ndarray,
    base_A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temps_k: np.ndarray,
    params: SchoolfieldParams = SchoolfieldParams(),
    h0: np.ndarray | None = None,
) -> np.ndarray:
    """
    Run a temperature‑aware SSM sequentially over a sequence.

    Parameters
    ----------
    x_seq : (T, input_dim)
        Input sequence.
    base_A, B, C : as in `ssm_step`
    temps_k : (T,)
        Temperature (K) at each time step.
    params : SchoolfieldParams, optional
    h0 : (state_dim,) or None
        Initial hidden state (zeros if None).

    Returns
    -------
    Y : (T, output_dim)
        Output sequence.
    """
    T, _ = x_seq.shape
    state_dim = base_A.shape[0]
    if h0 is None:
        h = np.zeros(state_dim)
    else:
        h = h0.copy()

    Y = np.empty((T, C.shape[0]))
    for t in range(T):
        h, y = hybrid_ssm_step(h, x_seq[t], base_A, B, C, temps_k[t], params)
        Y[t] = y
    return Y


def hybrid_semiseparable_matrix(
    base_A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temps_k: np.ndarray,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> np.ndarray:
    """
    Build the semiseparable matrix `M` for the temperature‑modulated SSM.

    The matrix satisfies `Y = M @ X` where `X` is the input sequence
    (flattened to shape (T, input_dim)) and `Y` the output sequence.

    Parameters
    ----------
    base_A, B, C : as in `ssm_step`
    temps_k : (T,)
        Temperature per time step.
    params : SchoolfieldParams, optional

    Returns
    -------
    M : (T, T, output_dim, input_dim)
        Lower‑triangular block matrix.  For i ≥ j,
        `M[i, j] = C_i @ (∏_{k=j+1}^{i} A_k) @ B_j`,
        otherwise the block is zero.
    """
    T = len(temps_k)
    state_dim = base_A.shape[0]
    input_dim = B.shape[1]
    output_dim = C.shape[0]

    # Pre‑compute temperature‑scaled A matrices
    A_seq = [
        temperature_dependent_state_transition(base_A, temps_k[t], params)
        for t in range(T)
    ]

    M = np.zeros((T, T, output_dim, input_dim))

    # Compute each causal block
    for i in range(T):
        for j in range(i + 1):
            # Product of A_{j+1} .. A_i (identity if j == i)
            prod = np.eye(state_dim)
            for k in range(j + 1, i + 1):
                prod = A_seq[k] @ prod
            M[i, j] = C @ prod @ B
    return M


def verify_hybrid_duality(
    x_seq: np.ndarray,
    base_A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temps_k: np.ndarray,
    params: SchoolfieldParams = SchoolfieldParams(),
    atol: float = 1e-6,
) -> bool:
    """
    Verify that the sequential hybrid scan matches the parallel semiseparable
    computation within a tolerance.

    Returns True if the two outputs are close enough.
    """
    Y_seq = hybrid_ssm_sequential(x_seq, base_A, B, C, temps_k, params)

    M = hybrid_semiseparable_matrix(base_A, B, C, temps_k, params)

    # Parallel multiplication Y = M @ X
    T, input_dim = x_seq.shape
    output_dim = C.shape[0]
    Y_par = np.empty((T, output_dim))
    for i in range(T):
        y_i = np.zeros(output_dim)
        for j in range(i + 1):
            y_i += M[i, j] @ x_seq[j]
        Y_par[i] = y_i

    return np.allclose(Y_seq, Y_par, atol=atol)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Dimensions
    T = 7
    state_dim = 4
    input_dim = 3
    output_dim = 2

    # Random model parameters
    base_A = np.random.randn(state_dim, state_dim)
    B = np.random.randn(state_dim, input_dim)
    C = np.random.randn(output_dim, state_dim)

    # Random input sequence
    x_seq = np.random.randn(T, input_dim)

    # Random temperatures between 10 °C and 35 °C (in Kelvin)
    temps_c = np.random.uniform(10, 35, size=T)
    temps_k = temps_c + 273.15

    # Run hybrid sequential scan
    Y_seq = hybrid_ssm_sequential(x_seq, base_A, B, C, temps_k)

    # Build and apply parallel semiseparable matrix
    M = hybrid_semiseparable_matrix(base_A, B, C, temps_k)
    Y_par = np.empty_like(Y_seq)
    for i in range(T):
        y = np.zeros(output_dim)
        for j in range(i + 1):
            y += M[i, j] @ x_seq[j]
        Y_par[i] = y

    # Verify duality
    assert np.allclose(Y_seq, Y_par, atol=1e-6), "Duality check failed"

    # Simple sanity print
    print("Hybrid sequential output (first row):", Y_seq[0])
    print("Hybrid parallel output   (first row):", Y_par[0])
    print("Verification passed.")