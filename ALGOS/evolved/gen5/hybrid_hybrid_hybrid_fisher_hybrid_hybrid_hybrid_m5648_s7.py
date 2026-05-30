# DARWIN HAMMER — match 5648, survivor 7
# gen: 5
# parent_a: hybrid_hybrid_fisher_locali_jepa_energy_m52_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1380_s3.py (gen4)
# born: 2026-05-30T00:04:01Z

import math
import random

import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places for percentage‑style reporting."""
    return round(float(value), 6)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ----------------------------------------------------------------------
# Parent A – Fisher information utilities (vectorised)
# ----------------------------------------------------------------------
def fisher_scores(thetas: np.ndarray, center: float, width: float, eps: float = 1e-12) -> np.ndarray:
    """
    Vectorised Fisher information for an array of angular measurements.
    Returns a vector of the same shape as ``thetas``.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    z = (thetas - center) / width
    intensity = np.exp(-0.5 * z * z)
    intensity = np.maximum(intensity, eps)
    derivative = intensity * (-(thetas - center) / (width * width))
    return (derivative ** 2) / intensity


# ----------------------------------------------------------------------
# Parent B – LTC utilities with deeper coupling
# ----------------------------------------------------------------------
def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    Wx: np.ndarray,
    Wi: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Liquid‑Time‑Constant (LTC) transformation with separate weight matrices
    for the external input ``x`` and the intensity vector ``I``.
    """
    return sigmoid(Wx @ x + Wi @ I + b)


# ----------------------------------------------------------------------
# Hybrid functionality – deeper mathematical integration
# ----------------------------------------------------------------------
def hybrid_fisher_ltc(
    thetas: np.ndarray,
    center: float,
    width: float,
    alpha: float = 0.1,
    beta: float = 0.9,
) -> np.ndarray:
    """
    Compute a Fisher‑regularized LTC gating signal.

    * ``alpha`` controls the contribution of the neutral external input ``x``.
    * ``beta`` controls the contribution of the Fisher intensity ``I``.
    The two contributions are linearly combined before the sigmoid,
    providing a genuine fusion rather than a trivial pass‑through.
    """
    I = fisher_scores(thetas, center, width)
    x = np.full_like(I, 0.5)

    n = len(I)
    Wx = alpha * np.eye(n)
    Wi = beta * np.eye(n)
    b = np.zeros(n)

    return ltc_f(x, I, Wx, Wi, b)


def allocate_workshare(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: tuple[str, ...] = GROUPS,
) -> dict[str, float]:
    """Allocate deterministic vs. LLM‑driven work units across model groups."""
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups)

    lanes = [
        {
            "group": group,
            "llm_units": _pct(per_group),
            "llm_share_pct": _pct(100.0 / len(groups)),
            "proof_required": True,
        }
        for group in groups
    ]

    return {
        "total_units": _pct(total_units),
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "lanes": lanes,
    }


def hybrid_allocate_workshare(
    total_units: float,
    center: float,
    width: float,
    thetas: np.ndarray,
    deterministic_target_pct: float = 90.0,
    groups: tuple[str, ...] = GROUPS,
) -> dict:
    """
    Allocate work units and modulate each lane's LLM portion with a
    *per‑lane* Fisher‑LTC gating factor.

    If ``len(thetas)`` differs from the number of groups, ``thetas`` is tiled
    to match the group count.
    """
    if len(groups) == 0:
        raise ValueError("At least one group required")

    # Align thetas with groups
    if thetas.size != len(groups):
        thetas = np.resize(thetas, len(groups))

    base = allocate_workshare(
        total_units=total_units,
        deterministic_target_pct=deterministic_target_pct,
        groups=groups,
    )

    gating = hybrid_fisher_ltc(thetas, center, width)  # shape == len(groups)

    # Normalise gating so its mean equals 1 (preserves total LLM units)
    gating = np.maximum(gating, 1e-6)  # avoid zero scaling
    scaling_factors = gating / gating.mean()

    # Apply per‑lane scaling
    for lane, factor in zip(base["lanes"], scaling_factors):
        lane["llm_units"] = _pct(lane["llm_units"] * factor)
        lane["llm_share_pct"] = _pct(lane["llm_share_pct"] * factor)

    # Re‑compute totals while keeping the overall deterministic portion unchanged
    total_llm = sum(l["llm_units"] for l in base["lanes"])
    base["llm_units"] = _pct(total_llm)
    base["deterministic_units"] = _pct(base["total_units"] - base["llm_units"])

    # Renormalise lane percentages to sum to exactly 100 %
    total_pct = sum(l["llm_share_pct"] for l in base["lanes"])
    if total_pct == 0:
        uniform = 100.0 / len(base["lanes"])
        for lane in base["lanes"]:
            lane["llm_share_pct"] = _pct(uniform)
    else:
        for lane in base["lanes"]:
            lane["llm_share_pct"] = _pct(lane["llm_share_pct"] * 100.0 / total_pct)

    return base


def hybrid_select_action(
    thetas: np.ndarray,
    center: float,
    width: float,
    action_space: list[int],
    temperature: float = 1.0,
) -> int:
    """
    Sample an action from ``action_space`` where probabilities are shaped by a
    Fisher‑LTC gating factor.  Higher Fisher information yields a sharper
    distribution.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    gating = hybrid_fisher_ltc(thetas, center, width)
    store_factor = float(np.mean(gating))

    # Inverse‑rank weighting (higher rank → lower weight)
    ranks = np.arange(len(action_space))[::-1]
    raw = store_factor * (1.0 / (1.0 + ranks))

    logits = raw / temperature
    # Stable softmax
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / exp_logits.sum()

    return random.choices(action_space, weights=probs, k=1)[0]