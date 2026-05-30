# DARWIN HAMMER — match 2125, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_korpus_text_h_m802_s0.py (gen5)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s4.py (gen4)
# born: 2026-05-29T23:41:01Z

import datetime as _dt
import math
import random
import sys
import pathlib
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    Uses Python's ``datetime`` module.
    """
    return (int(_dt.date(year, month, day).weekday()) + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    A sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("`groups` must contain at least one element.")
    base_angles = np.arange(n, dtype=np.float64) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


# ----------------------------------------------------------------------
# GPU selection utilities
# ----------------------------------------------------------------------
def vram_aware_gpu_selection(
    gpus: List[Dict[str, Any]],
    budget_mb: int,
    reserve_mb: int,
) -> List[Dict[str, Any]]:
    """
    Return GPUs whose *free* VRAM can satisfy ``budget_mb`` plus ``reserve_mb``.
    """
    required = budget_mb + reserve_mb
    return [gpu for gpu in gpus if gpu.get("memory.free", 0) >= required]


def score_gpus_by_memory(
    gpus: List[Dict[str, Any]],
    weight_vec: np.ndarray,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Compute a score for each GPU based on its free memory and a weight vector.
    The weight vector is applied to a normalised memory vector of length ``len(weight_vec)``.
    If the number of GPUs differs from ``len(weight_vec)`` the weight vector is
    interpolated linearly to match the GPU count.
    """
    if not gpus:
        return []

    mem_free = np.array([gpu.get("memory.free", 0) for gpu in gpus], dtype=np.float64)
    # Normalise memory to [0, 1] to avoid scale issues.
    if mem_free.max() == 0:
        norm_mem = mem_free
    else:
        norm_mem = mem_free / mem_free.max()

    # Align weight vector length with number of GPUs.
    if len(weight_vec) != len(gpus):
        # Linear interpolation of the weight vector.
        x_src = np.linspace(0, 1, len(weight_vec))
        x_dst = np.linspace(0, 1, len(gpus))
        weight_vec = np.interp(x_dst, x_src, weight_vec)

    scores = norm_mem * weight_vec
    return [(gpu, float(score)) for gpu, score in zip(gpus, scores)]


def hybrid_selection(
    gpus: List[Dict[str, Any]],
    budget_mb: int,
    reserve_mb: int,
    weight_vec: np.ndarray,
    top_k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Select GPUs that satisfy VRAM constraints and are ranked by a
    memory‑aware weight score. ``top_k`` limits the number of returned GPUs.
    """
    eligible = vram_aware_gpu_selection(gpus, budget_mb, reserve_mb)
    scored = score_gpus_by_memory(eligible, weight_vec)
    # Sort descending by score.
    scored.sort(key=lambda pair: pair[1], reverse=True)
    selected = [gpu for gpu, _ in scored]
    if top_k is not None:
        selected = selected[:top_k]
    return selected


# ----------------------------------------------------------------------
# NLMS (Normalized Least‑Mean‑Squares) core
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction ``w·x``."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS weight update.
    Returns the new weight vector and the raw prediction error.
    """
    pred = nlms_predict(weights, x)
    error = target - pred
    norm = float(np.dot(x, x) + eps)
    new_weights = weights + (mu * error / norm) * x
    return new_weights, error


# ----------------------------------------------------------------------
# Spline / KAN (Kernel‑Based Activation) utilities
# ----------------------------------------------------------------------
def _spline_piecewise_linear(x: np.ndarray, knots: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Vectorised linear spline evaluation."""
    return np.interp(x, knots, coeffs)


def kan_transform(
    x: np.ndarray,
    params: Dict[str, List[np.ndarray]],
) -> np.ndarray:
    """
    Apply a simple piece‑wise‑linear KAN transformation.
    ``params`` must contain ``"knots"`` and ``"coeffs"``, each a list whose
    length matches ``x.shape[0]``.
    The function returns the transformed vector with an appended bias term.
    """
    knots_list = params["knots"]
    coeffs_list = params["coeffs"]
    if not (len(knots_list) == len(coeffs_list) == x.shape[0]):
        raise ValueError("Parameter lists must match the dimensionality of ``x``.")

    transformed = np.empty_like(x, dtype=np.float64)
    for idx, (kn, cf) in enumerate(zip(knots_list, coeffs_list)):
        transformed[idx] = _spline_piecewise_linear(
            np.array([x[idx]]), kn, cf
        )[0]
    # Append a constant bias term for downstream linear layers.
    return np.concatenate([transformed, np.array([1.0], dtype=np.float64)])


# ----------------------------------------------------------------------
# Integrated hybrid prediction / update
# ----------------------------------------------------------------------
def hybrid_prediction(
    weights: np.ndarray,
    x: np.ndarray,
    kan_params: Dict[str, List[np.ndarray]] | None = None,
) -> float:
    """
    Predict using NLMS. If ``kan_params`` is supplied the input ``x`` is first
    passed through a KAN transform, effectively deepening the model.
    """
    if kan_params is not None:
        x = kan_transform(x, kan_params)
    return nlms_predict(weights, x)


def hybrid_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
    kan_params: Dict[str, List[np.ndarray]] | None = None,
) -> Tuple[np.ndarray, float]:
    """
    Update NLMS weights. ``x`` can be optionally transformed by a KAN layer
    before the update, providing a deeper, non‑linear representation.
    """
    if kan_params is not None:
        x = kan_transform(x, kan_params)
    return nlms_update(weights, x, target, mu, eps)


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample GPU inventory.
    gpus = [
        {"memory.free": 4096},
        {"memory.free": 8192},
        {"memory.free": 12288},
    ]

    # Weekday‑dependent weight vector.
    dow = doomsday(2026, 5, 29)  # today
    weight_vec = weekday_weight_vector(GROUPS, dow)

    # Select the best GPU(s) respecting VRAM constraints.
    selected = hybrid_selection(
        gpus,
        budget_mb=DEFAULT_BUDGET_MB,
        reserve_mb=DEFAULT_RESERVE_MB,
        weight_vec=weight_vec,
        top_k=2,
    )
    print("Selected GPUs:", selected)

    # NLMS model initialisation.
    input_dim = 3
    # If we plan to use a KAN transform we need an extra bias term.
    use_kan = True
    if use_kan:
        input_dim += 1  # bias appended by ``kan_transform``

    rng = np.random.default_rng(seed=42)
    weights = rng.normal(scale=0.1, size=input_dim)

    # Dummy KAN parameters (simple identity splines).
    if use_kan:
        kan_params = {
            "knots": [np.array([0.0, 1.0]) for _ in range(3)],
            "coeffs": [np.array([0.0, 1.0]) for _ in range(3)],
        }
    else:
        kan_params = None

    # Example input and target.
    x_raw = np.array([1.0, 2.0, 3.0])
    target = 5.0

    # Prediction & update cycle.
    pred = hybrid_prediction(weights, x_raw, kan_params)
    print("Prediction before update:", pred)

    weights, err = hybrid_update(weights, x_raw, target, kan_params=kan_params)
    print("Updated weights:", weights)
    print("Prediction error:", err)