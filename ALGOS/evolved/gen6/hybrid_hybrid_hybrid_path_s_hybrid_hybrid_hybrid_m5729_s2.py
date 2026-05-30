# DARWIN HAMMER — match 5729, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_nlms_omni_cha_m142_s0.py (gen3)
# born: 2026-05-30T00:04:29Z

import numpy as np
from dataclasses import dataclass
from datetime import date
from typing import Tuple


@dataclass(frozen=True)
class MathAction:
    """Immutable description of a decision action."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Immutable description of a counterfactual outcome."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Perform the classic lead‑lag (or time‑delay) embedding of a path.

    Parameters
    ----------
    path : np.ndarray, shape (T, d)
        Original trajectory.

    Returns
    -------
    np.ndarray, shape (2*T‑1, 2*d)
        Lead‑lag transformed path.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dimension)")

    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)

    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])          # lead
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])  # lag
    out[2 * (T - 1)] = np.concatenate([path[-1], path[-1]])       # final point

    return out


def signature_level1(path: np.ndarray) -> np.ndarray:
    """
    First level (increment) signature: ΔX = X_T – X_0.
    """
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path: np.ndarray) -> np.ndarray:
    """
    Second level (area) signature: Σ (X_{t‑1} – X_0) ⊗ ΔX_t.
    Returns a (d, d) matrix.
    """
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running = path[:-1] - path[0]               # (T‑1, d)
    return running.T @ increments               # (d, d)


def combined_signature(path: np.ndarray) -> float:
    """
    Produce a scalar measure of the path's signature by blending
    level‑1 norm and level‑2 Frobenius norm.

    The blending factor α can be tuned; here we use α = 0.5.
    """
    lead_lag = lead_lag_transform(path)
    lvl1 = signature_level1(lead_lag)
    lvl2 = signature_level2(lead_lag)

    norm1 = np.linalg.norm(lvl1)                # ℓ2 norm of level‑1 vector
    norm2 = np.linalg.norm(lvl2, ord="fro")    # Frobenius norm of level‑2 matrix

    α = 0.5
    return α * norm1 + (1 - α) * norm2


def nlms_update(
    weights: np.ndarray,
    input_signal: np.ndarray,
    desired_signal: float,
    step_size: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Normalized Least‑Mean‑Squares weight update.

    Parameters
    ----------
    weights : np.ndarray, shape (d,)
        Current filter coefficients.
    input_signal : np.ndarray, shape (d,)
        Current input vector.
    desired_signal : float
        Desired (target) scalar output.
    step_size : float
        Base learning rate (must be positive).
    epsilon : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray, shape (d,)
        Updated weights.
    """
    input_signal = np.asarray(input_signal, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if input_signal.shape != weights.shape:
        raise ValueError("input_signal and weights must have the same shape")

    # Prediction error
    error = desired_signal - np.dot(weights, input_signal)

    # Normalized step (μ / (‖x‖² + ε))
    norm_sq = np.dot(input_signal, input_signal)
    mu_norm = step_size / (norm_sq + epsilon)

    # Weight update
    return weights + mu_norm * error * input_signal


def adaptive_step_size(
    base_step: float,
    signature_scalar: float,
    input_signal: np.ndarray,
    α: float = 0.7,
    β: float = 0.3,
) -> float:
    """
    Compute a deeper‑integrated adaptive step size.

    The formula blends the influence of the path signature (via its magnitude)
    and the instantaneous energy of the input signal.

    μ_adapt = base_step * (1 + α·‖sig‖) / (1 + β·‖x‖²)

    Parameters
    ----------
    base_step : float
        Baseline learning rate.
    signature_scalar : float
        Positive scalar summarising the path signature.
    input_signal : np.ndarray
        Current input vector.
    α, β : float
        Tuning coefficients (default α=0.7, β=0.3).

    Returns
    -------
    float
        Adapted step size (always positive).
    """
    sig_norm = max(signature_scalar, 0.0)
    x_norm_sq = np.dot(input_signal, input_signal)
    return base_step * (1.0 + α * sig_norm) / (1.0 + β * x_norm_sq)


def hybrid_operation(
    path: np.ndarray,
    weights: np.ndarray,
    input_signal: np.ndarray,
    desired_signal: float,
    base_step: float,
) -> np.ndarray:
    """
    Fuse path‑signature information with NLMS adaptation.

    1. Compute a scalar signature from the lead‑lag transformed path.
    2. Derive an adaptive step size that respects both signature magnitude
       and input energy.
    3. Perform a normalized LMS update with the adaptive step.

    Returns the updated weight vector.
    """
    sig_scalar = combined_signature(path)
    step = adaptive_step_size(base_step, sig_scalar, input_signal)
    return nlms_update(weights, input_signal, desired_signal, step)


def sphericity_index(length: float, width: float, height: float) -> float:
    """
    Ratio of the geometric mean of the three dimensions to the largest dimension.
    """
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be strictly positive")
    geo_mean = (length * width * height) ** (1.0 / 3.0)
    return geo_mean / max(length, width, height)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return the day‑of‑week index where Monday=0, …, Sunday=6.
    """
    return date(year, month, day).weekday()


if __name__ == "__main__":
    # Simple sanity check
    rng = np.random.default_rng(seed=42)

    path = rng.random((12, 4))                     # 12 time steps, 4‑D state
    weights = rng.random(4)                        # initial NLMS coefficients
    input_signal = rng.random(4)                   # current observation
    desired_signal = rng.random()                  # target scalar
    base_step = 0.05                               # modest learning rate

    updated_weights = hybrid_operation(
        path, weights, input_signal, desired_signal, base_step
    )
    print("Updated weights:", updated_weights)

    print("Sphericity index (1,2,3):", sphericity_index(1.0, 2.0, 3.0))
    print("Doomsday for 2024‑01‑01 (Mon=0):", doomsday(2024, 1, 1))