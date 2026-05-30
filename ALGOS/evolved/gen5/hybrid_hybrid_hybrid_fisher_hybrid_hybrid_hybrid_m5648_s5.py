# DARWIN HAMMER — match 5648, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_fisher_locali_jepa_energy_m52_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1380_s3.py (gen4)
# born: 2026-05-30T00:04:01Z

"""Hybrid Fisher-LTC Algorithm
This module fuses two parent algorithms:

* **Parent A** – `hybrid_hybrid_fisher_locali_jepa_energy_m52_s0.py`  
  Provides Gaussian beam modeling and Fisher information scoring for angular
  measurements.

* **Parent B** – `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1380_s3.py`  
  Supplies a Liquid‑Time‑Constant (LTC) neural‑style transformation
  (`ltc_f`) together with a work‑share allocation routine.

**Mathematical Bridge**  
Both parents operate on *information density* vectors.  
Parent A computes a scalar Fisher score `I(θ)` for each angle `θ`.  
Parent B concatenates an external input vector `x` with an “intensity” vector
`I` and applies a linear map followed by a sigmoid (`ltc_f`).  

The hybrid therefore treats the Fisher scores as the intensity vector `I` in
the LTC computation, letting the Fisher information directly modulate the
LTC gating of downstream quantities (e.g. work‑share fractions).  The result
is a unified system where statistical information (Fisher) regularizes the
learned allocation dynamics (LTC)."""

import math
import random
import sys
from datetime import datetime, timezone, date
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Fisher information utilities
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity of a beam at angle ``theta``."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Scalar Fisher information for a single angular measurement."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def parse_loose_datetime(raw: str) -> datetime | None:
    """Best‑effort ISO‑8601 parser that always returns a UTC datetime or ``None``."""
    text = raw.strip().strip("'\"`[]()")
    if not text:
        return None
    try:
        val = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    except ValueError:
        return None


# ----------------------------------------------------------------------
# Parent B – LTC and work‑share utilities
# ----------------------------------------------------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Liquid‑Time‑Constant (LTC) transformation.

    The function concatenates the external input ``x`` with the intensity
    vector ``I`` and applies a linear map ``W`` plus bias ``b`` before a
    sigmoid non‑linearity.
    """
    concat = np.concatenate([x, I], axis=0)
    return sigmoid(W @ concat + b)


GROUPS = ("codex", "groq", "cohere", "local_models")


def _pct(value: float) -> float:
    return round(float(value), 6)


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


# ----------------------------------------------------------------------
# Hybrid functionality – three core functions
# ----------------------------------------------------------------------


def hybrid_fisher_ltc(
    thetas: np.ndarray,
    center: float,
    width: float,
) -> np.ndarray:
    """
    Compute Fisher scores for a set of angles and feed them as the intensity
    vector ``I`` to the LTC block.

    The external input ``x`` is a constant‑value vector (0.5) of the same
    length, representing a neutral prior.  The weight matrix ``W`` passes the
    Fisher component unchanged (identity) while ignoring ``x``; the bias is
    zero.  The returned vector has the same length as ``thetas`` and can be
    interpreted as a Fisher‑regularized gating signal.
    """
    # Fisher intensity vector
    I = np.vectorize(lambda th: fisher_score(th, center, width))(thetas)

    # Neutral external input
    x = np.full_like(I, 0.5)

    # Build W: zeros for the x‑part, identity for the I‑part
    n = len(I)
    W = np.concatenate([np.zeros((n, n)), np.eye(n)], axis=1)
    b = np.zeros(n)

    return ltc_f(x, I, W, b)


def hybrid_allocate_workshare(
    total_units: float,
    center: float,
    width: float,
    thetas: np.ndarray,
) -> dict:
    """
    Allocate work units and then modulate the LLM portion of each lane by the
    Fisher‑LTC gating signal computed for ``thetas``.

    The function first calls :func:`allocate_workshare` to obtain a baseline
    allocation, then rescales each lane's ``llm_units`` by the average of the
    gating signal (produced by :func:`hybrid_fisher_ltc`).  This couples the
    statistical information about the angular domain directly to the resource
    distribution.
    """
    base = allocate_workshare(total_units=total_units, deterministic_target_pct=90.0)

    gating = hybrid_fisher_ltc(thetas, center, width)
    scale = float(np.mean(gating))  # scalar in (0,1)

    for lane in base["lanes"]:
        lane["llm_units"] = _pct(lane["llm_units"] * scale)
        lane["llm_share_pct"] = _pct(lane["llm_share_pct"] * scale)

    # Re‑compute totals after scaling
    base["llm_units"] = _pct(sum(l["llm_units"] for l in base["lanes"]))
    base["deterministic_units"] = _pct(base["total_units"] - base["llm_units"])
    return base


def hybrid_select_action(
    thetas: np.ndarray,
    center: float,
    width: float,
    action_space: list[int],
    temperature: float = 1.0,
) -> int:
    """
    Choose an action from ``action_space`` where the probability of each action
    is weighted by a Fisher‑LTC factor.

    The factor ``store_factor`` is the mean gating signal; higher Fisher
    information yields a more peaked distribution.  A softmax with temperature
    is applied to obtain a proper probability distribution.
    """
    gating = hybrid_fisher_ltc(thetas, center, width)
    store_factor = float(np.mean(gating))

    # Raw scores: inverse‑rank weighted by store_factor
    raw = np.array([store_factor * (1.0 / (1 + i)) for i in action_space], dtype=float)

    # Temperature‑scaled softmax
    logits = raw / max(temperature, 1e-8)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    choice = random.choices(action_space, weights=probs, k=1)[0]
    return choice


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example angular domain
    thetas = np.linspace(-math.pi, math.pi, 16)

    # Hybrid Fisher‑LTC gating demonstration
    gating = hybrid_fisher_ltc(thetas, center=0.0, width=1.0)
    print("Gating signal (first 5):", gating[:5])

    # Hybrid allocation
    allocation = hybrid_allocate_workshare(
        total_units=1000.0,
        center=0.0,
        width=1.0,
        thetas=thetas,
    )
    print("\nHybrid allocation:")
    for lane in allocation["lanes"]:
        print(lane)

    # Hybrid action selection
    action = hybrid_select_action(
        thetas,
        center=0.0,
        width=1.0,
        action_space=list(range(1, 6)),
        temperature=0.5,
    )
    print("\nSelected action:", action)