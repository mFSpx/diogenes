# DARWIN HAMMER — match 2772, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_ternar_m1199_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1399_s2.py (gen6)
# born: 2026-05-29T23:45:45Z

"""
Hybrid Allocation Module
=======================

This module fuses the mathematical cores of two parent algorithms:

* **Parent A** – Provides a Structural Similarity Index (SSIM) based work‑share
  allocation across a set of LLM groups.
* **Parent B** – Supplies a Fisher information computation for Gaussian beams,
  a Lanczos Gamma approximation, and a pheromone‑signal model.

**Mathematical bridge**

Both parents operate on normalized weight distributions:

* SSIM yields a scalar similarity `s ∈ [0, 1]` that can be interpreted as a
  confidence weight.
* Fisher information `I(θ)` is a non‑negative scalar that, after normalisation,
  forms a probability‑like distribution over a set of angles.

The hybrid algorithm treats each LLM group as an “angle” `θ_i` on the unit
circle.  For each group we compute a Fisher score `I_i`.  After normalising the
scores we obtain a probability vector `p_i`.  The SSIM scalar `s` modulates the
LLM portion of the total work units, scaling every `p_i`.  Thus the final
allocation per group is:


deterministic_units = total_units * deterministic_target_pct / 100
llm_units          = total_units - deterministic_units
allocation_i       = llm_units * s * p_i


The same `s` can also modulate auxiliary signals such as the pheromone
signal, providing a unified weighting across the whole system.

The module offers three high‑level hybrid functions:
`hybrid_allocate_workshare`, `hybrid_pheromone_signal`, and
`hybrid_count_min_sketch`.
"""

import math
import random
import sys
import pathlib
import hashlib
from datetime import date
from typing import Any, Iterable, Tuple, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A core (SSIM)
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Structural Similarity Index (SSIM) between two 1‑D vectors."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    )
    return float(ssim)


# ----------------------------------------------------------------------
# Parent B core (Lanczos Gamma, Gaussian beam, Fisher score, sketch)
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array(
    [
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
)


def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of Γ(z) for z > 0."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * (t ** (z + 0.5)) * math.exp(-t) * x


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) centred at *center* with standard deviation *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ of a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def count_min_sketch(items: Iterable[Any], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Very simple count‑min sketch using SHA‑256 as hash."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = hashlib.sha256(f"{d}:{item}".encode()).hexdigest()
            idx = int(h, 16) % width
            table[d][idx] += 1
    return table


def calculate_pheromone_signal(
    surface_key: Tuple[float, float],
    signal_kind: str,
    signal_value: float,
    half_life_seconds: float,
    alpha: float,
) -> np.ndarray:
    """
    Compute a pheromone signal using a Caputo fractional derivative
    (approximated analytically via the Gamma function).
    """
    # time axis discretised at 10 ms resolution
    current_time = np.arange(0, half_life_seconds, 0.01)
    # fractional power law decay
    signal = signal_value * (current_time ** (alpha - 1) / gamma_lanczos(alpha))
    return signal


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def _angular_fisher_distribution(
    groups: Tuple[str, ...], width: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map each group to an angle θ_i uniformly on [0, 2π) and compute its Fisher score.
    Returns the raw scores and the normalised probability vector.
    """
    n = len(groups)
    thetas = np.linspace(0, 2 * math.pi, n, endpoint=False)
    centers = thetas  # centre = angle itself
    raw_scores = np.array(
        [fisher_score(theta, center, width) for theta, center in zip(thetas, centers)]
    )
    total = raw_scores.sum()
    probs = raw_scores / total if total > 0 else np.full_like(raw_scores, 1.0 / n)
    return raw_scores, probs


def hybrid_allocate_workshare(
    x: np.ndarray,
    y: np.ndarray,
    *,
    groups: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models"),
    total_units: float,
    deterministic_target_pct: float = 90.0,
    fisher_width: float = 0.3,
) -> Dict[str, Any]:
    """
    Allocate work units across groups using a blend of SSIM and Fisher information.

    Steps
    -----
    1. Compute similarity `s = compute_ssim(x, y)`.
    2. Derive a normalized Fisher probability vector `p_i` over the groups.
    3. Deterministic portion = `total_units * deterministic_target_pct / 100`.
    4. LLM portion = remaining units, scaled by `s`.
    5. Allocate to each group: `allocation_i = llm_units * s * p_i`.

    Returns a dictionary mirroring the structure of Parent A's allocation output,
    but enriched with the underlying Fisher scores.
    """
    s = compute_ssim(x, y)
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units

    raw_fisher, prob_fisher = _angular_fisher_distribution(groups, width=fisher_width)

    lanes = []
    for grp, raw, prob in zip(groups, raw_fisher, prob_fisher):
        llm_alloc = llm_units * s * prob
        lane = {
            "group": grp,
            "llm_units": _pct(llm_alloc),
            "llm_share_pct": _pct(100.0 * prob),
            "fisher_raw_score": _pct(raw),
            "proof_required": True,
        }
        lanes.append(lane)

    return {
        "total_units": _pct(total_units),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "ssim": _pct(s),
        "lanes": lanes,
    }


def hybrid_pheromone_signal(
    x: np.ndarray,
    y: np.ndarray,
    *,
    surface_key: Tuple[float, float] = (0.0, 0.0),
    signal_kind: str = "generic",
    signal_value: float = 1.0,
    half_life_seconds: float = 10.0,
    alpha: float = 0.9,
) -> np.ndarray:
    """
    Produce a pheromone signal whose amplitude is modulated by the SSIM between
    `x` and `y`.  The underlying fractional decay uses the Lanczos Gamma
    approximation from Parent B.
    """
    s = compute_ssim(x, y)
    # Modulate the base signal value by similarity (higher similarity → stronger signal)
    modulated_value = signal_value * s
    return calculate_pheromone_signal(
        surface_key, signal_kind, modulated_value, half_life_seconds, alpha
    )


def hybrid_count_min_sketch(
    x: np.ndarray,
    y: np.ndarray,
    *,
    groups: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models"),
    width: int = 64,
    depth: int = 4,
) -> Dict[str, List[List[int]]]:
    """
    Build a count‑min sketch for a combined stream consisting of:
    * the raw values of `x`,
    * the raw values of `y`,
    * the string identifiers of the groups.

    The sketch provides a compact probabilistic summary that can be used by
    downstream components (e.g., for quick frequency queries).
    """
    combined = list(x) + list(y) + list(groups)
    sketch = count_min_sketch(combined, width=width, depth=depth)
    return {"sketch": sketch, "width": width, "depth": depth}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic vectors for demonstration
    rng = np.random.default_rng(42)
    vec_a = rng.random(100)
    vec_b = rng.random(100)

    groups = ("codex", "groq", "cohere", "local_models")
    allocation = hybrid_allocate_workshare(
        vec_a,
        vec_b,
        groups=groups,
        total_units=1000.0,
        deterministic_target_pct=85.0,
        fisher_width=0.25,
    )
    print("Hybrid Allocation:")
    for lane in allocation["lanes"]:
        print(lane)

    pheromone = hybrid_pheromone_signal(
        vec_a,
        vec_b,
        surface_key=(1.2, 3.4),
        signal_kind="test",
        signal_value=2.5,
        half_life_seconds=5.0,
        alpha=0.8,
    )
    print("\nPheromone signal shape:", pheromone.shape)

    sketch_info = hybrid_count_min_sketch(vec_a, vec_b, groups=groups, width=32, depth=3)
    print("\nCount‑Min Sketch (depth × width):", len(sketch_info["sketch"]), "×", len(sketch_info["sketch"][0]))
    # Verify that the script runs without raising exceptions.