# DARWIN HAMMER — match 2772, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_ternar_m1199_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1399_s2.py (gen6)
# born: 2026-05-29T23:45:45Z

"""Hybrid Allocation Engine

This module fuses two parent algorithms:

* **Parent A** – SSIM‑based work‑share allocation across LLM groups.
* **Parent B** – Lanczos Gamma, Gaussian beam, Fisher information, Count‑Min sketch and fractional‑derivative pheromone signal.

**Mathematical bridge**

The bridge is the scalar similarity `SSIM(x, y)` computed by Parent A.  
We use this scalar to modulate the Fisher information `I(θ)` from Parent B, turning the
angle‑dependent Fisher score into a *group‑specific similarity weight*:


w_g = SSIM(x, y) * I(θ_g)


where `θ_g` is a group‑specific angle derived from the group index.
The weight `w_g` is then combined with a frequency weight obtained from a
Count‑Min sketch of the task items.  The product of the two weights determines the
fraction of the LLM‑derived work units assigned to each group.

Thus the hybrid algorithm jointly exploits:
* a *global* similarity measure (SSIM),
* a *local* information‑theoretic measure (Fisher score),
* and a *frequency* estimate (Count‑Min sketch).

The resulting allocation respects the deterministic‑vs‑LLM split of Parent A while
injecting the richer physics‑inspired weighting of Parent B.
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
# Constants and helpers (shared by both parents)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
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

def _pct(value: float) -> float:
    """Round to six decimal places – utility from Parent A."""
    return round(float(value), 6)

# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Structural Similarity Index (SSIM) between two vectors."""
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
# Parent B components
# ----------------------------------------------------------------------
def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function for z > 0."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def count_min_sketch(items: Iterable[Any], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Simple count‑min sketch using SHA‑256 as hash."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = hashlib.sha256(f"{d}:{item}".encode()).hexdigest()
            idx = int(h, 16) % width
            table[d][idx] += 1
    return table

def estimate_frequency(sketch: List[List[int]], item: Any, width: int = 64) -> int:
    """Estimate frequency of *item* from a count‑min sketch."""
    mins = []
    for d, row in enumerate(sketch):
        h = hashlib.sha256(f"{d}:{item}".encode()).hexdigest()
        idx = int(h, 16) % width
        mins.append(row[idx])
    return min(mins)

def calculate_pheromone_signal(
    surface_key: Tuple[float, float],
    signal_value: float,
    half_life_seconds: float,
    alpha: float,
) -> np.ndarray:
    """
    Fractional‑order pheromone signal using a Caputo derivative approximation.
    """
    current_time = np.arange(0, half_life_seconds, 0.01)
    pheromone_signal = signal_value * (current_time ** (alpha - 1) / gamma_lanczos(alpha))
    return pheromone_signal

# ----------------------------------------------------------------------
# Hybrid functions (the new core)
# ----------------------------------------------------------------------
def hybrid_fisher_ssim_weight(
    x: np.ndarray,
    y: np.ndarray,
    group_index: int,
    total_groups: int,
    beam_center: float = 0.0,
    beam_width: float = 1.0,
) -> float:
    """
    Combine SSIM and Fisher information into a single weight for a group.

    The group index is mapped to an angle θ ∈ [−π, π] and the Fisher score
    I(θ) is multiplied by the global SSIM(x, y).  The result is normalised later.
    """
    ssim = compute_ssim(x, y)
    theta = -math.pi + 2 * math.pi * (group_index / total_groups)
    fisher = fisher_score(theta, beam_center, beam_width)
    return ssim * fisher

def hybrid_allocate_workshare(
    x: np.ndarray,
    y: np.ndarray,
    items: Iterable[Any],
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
) -> Dict[str, Any]:
    """
    Allocate work units across LLM groups using a hybrid of SSIM, Fisher
    information and a Count‑Min sketch of the task items.

    Returns a dictionary mirroring the structure of Parent A's allocation
    but with per‑group units scaled by the hybrid weight.
    """
    # 1️⃣ Deterministic vs LLM split (Parent A)
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units

    # 2️⃣ Build a sketch of the incoming items (Parent B)
    sketch = count_min_sketch(items)

    # 3️⃣ Compute raw hybrid weights for each group
    raw_weights = []
    for idx, group in enumerate(groups):
        # SSIM‑Fisher component
        hybrid_weight = hybrid_fisher_ssim_weight(x, y, idx, len(groups))
        # Frequency component – estimate using a representative token per group
        freq_est = estimate_frequency(sketch, group)
        raw_weights.append(hybrid_weight * (freq_est + 1))  # +1 to avoid zero

    # 4️⃣ Normalise weights to obtain a share of the LLM pool
    weight_sum = sum(raw_weights) or 1.0
    per_group_units = [(llm_units * w) / weight_sum for w in raw_weights]

    lanes = []
    for group, units in zip(groups, per_group_units):
        lanes.append(
            {
                "group": group,
                "llm_units": _pct(units),
                "llm_share_pct": _pct(100.0 * units / llm_units if llm_units else 0.0),
                "proof_required": True,
            }
        )

    return {
        "total_units": _pct(total_units),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "lanes": lanes,
    }

def hybrid_pheromone_profile(
    groups: Tuple[str, ...] = GROUPS,
    base_signal: float = 1.0,
    half_life_seconds: float = 10.0,
    alpha: float = 0.75,
) -> Dict[str, np.ndarray]:
    """
    Produce a pheromone signal time‑series for each group.  The surface key is
    derived from the hash of the group name to give each group a distinct
    fractional decay pattern.
    """
    profiles = {}
    for group in groups:
        h = hashlib.sha256(group.encode()).hexdigest()
        # Use first 8 hex digits to create a reproducible (x, y) pair
        x = int(h[:8], 16) / 0xffffffff
        y = int(h[8:16], 16) / 0xffffffff
        signal = calculate_pheromone_signal(
            surface_key=(x, y),
            signal_value=base_signal,
            half_life_seconds=half_life_seconds,
            alpha=alpha,
        )
        profiles[group] = signal
    return profiles

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic vectors for SSIM
    rng = np.random.default_rng(42)
    vec_a = rng.random(100)
    vec_b = rng.random(100)

    # Synthetic task items (could be prompts, IDs, etc.)
    task_items = [f"task_{i%7}" for i in range(250)]

    allocation = hybrid_allocate_workshare(
        vec_a,
        vec_b,
        task_items,
        total_units=1_000.0,
        deterministic_target_pct=85.0,
    )

    print("Hybrid Allocation Result:")
    for lane in allocation["lanes"]:
        print(
            f"Group {lane['group']}: {lane['llm_units']} units "
            f"({lane['llm_share_pct']}% of LLM pool)"
        )

    # Generate pheromone profiles (just to verify they run)
    profiles = hybrid_pheromone_profile()
    sample_group = GROUPS[0]
    print(f"\nPheromone profile for {sample_group} (first 5 samples):")
    print(profiles[sample_group][:5])