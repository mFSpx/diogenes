# DARWIN HAMMER — match 2761, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_hybrid_m1138_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1491_s1.py (gen6)
# born: 2026-05-29T23:45:46Z

"""Hybrid Fisher‑Entropy‑Bandit Algorithm
Integrates:
- Parent A: Shannon entropy of pheromone signals + temperature‑dependent developmental rate (Schoolfield model).
- Parent B: Fisher information of a Gaussian beam + count‑min sketch for feature aggregation.

Mathematical bridge:
The pheromone signal value is treated as the “angle” θ of a Gaussian beam.
Fisher information I(θ) (parent B) quantifies the local sensitivity of the signal,
while the developmental rate r(T) (parent A) provides a temperature‑scaled activity weight.
The hybrid activity for each pheromone entry is defined as

    A_i = r(T_i) · I(θ_i) / (H + ε)

where H is the Shannon entropy of the whole pheromone distribution and ε avoids division by zero.
The sketch aggregates surface keys, enabling a fast approximation of the empirical distribution
used in the entropy term. This fuses the core topologies of both parents into a single unified system.
"""

import math
import random
import sys
import pathlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0          # rate at 25 °C
    delta_h_activation: float = 12_000.0   # J·mol⁻¹
    t_low: float = 283.15        # K
    t_high: float = 307.15       # K
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987         # universal gas constant (cal·K⁻¹·mol⁻¹)


@dataclass(frozen=True)
class PheromoneEntry:
    surface_key: str
    signal_kind: str
    signal_value: float          # interpreted as θ for Fisher info
    half_life_seconds: float
    temperature_c: float         # ambient temperature for developmental rate


# ----------------------------------------------------------------------
# Core functions from Parent A
# ----------------------------------------------------------------------
def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield model for temperature‑dependent developmental rate.
    Returns a dimensionless activity factor.
    """
    # Pre‑compute exponentials for low / high temperature inhibition
    exp_low_T = math.exp(params.delta_h_low / (params.r_cal * temp_k))
    exp_low_ref = math.exp(params.delta_h_low / (params.r_cal * params.t_low))
    exp_high_T = math.exp(params.delta_h_high / (params.r_cal * temp_k))
    exp_high_ref = math.exp(params.delta_h_high / (params.r_cal * params.t_high))

    # Arrhenius term
    arrhenius = math.exp(-params.delta_h_activation / (params.r_cal * temp_k))

    # Full Schoolfield equation
    numerator = params.rho_25 * arrhenius * (1.0 + exp_low_ref + exp_high_ref)
    denominator = 1.0 + exp_low_T + exp_high_T
    return numerator / denominator


def shannon_entropy(values: Iterable[float]) -> float:
    """Standard Shannon entropy H = -∑ p·log(p) for a discrete distribution."""
    counter = Counter(values)
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for cnt in counter.values():
        p = cnt / total
        entropy -= p * math.log(p + 1e-12)  # natural log
    return entropy


# ----------------------------------------------------------------------
# Core functions from Parent B
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_information(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam.
    I(θ) = (∂ln f/∂θ)² where f is the beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """
    Simple count‑min sketch: hash each item depth times and increment counters.
    """
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            index = (hash(item) + d) % width
            table[d][index] += 1
    return table


def sketch_entropy(sketch: List[List[int]]) -> float:
    """
    Approximate entropy from a count‑min sketch by flattening the counters.
    """
    flat = [cnt for row in sketch for cnt in row if cnt > 0]
    return shannon_entropy(flat)


# ----------------------------------------------------------------------
# Hybrid operations (the new fused algorithm)
# ----------------------------------------------------------------------
def hybrid_activity(
    entries: List[PheromoneEntry],
    params: SchoolfieldParams = SchoolfieldParams(),
    width: int = 64,
    depth: int = 4,
) -> Dict[str, float]:
    """
    Compute a hybrid activity score for each pheromone entry.

    Steps:
    1. Build a count‑min sketch of the surface keys → H_s (entropy of sketch).
    2. Compute global Shannon entropy H_g of all signal values.
    3. For each entry:
        a) r_i = developmental_rate(T_i)
        b) I_i = fisher_information(signal_value_i)
        c) A_i = r_i * I_i / (H_g + H_s + ε)

    Returns a mapping surface_key → activity score.
    """
    # 1. Sketch and its entropy
    keys = (e.surface_key for e in entries)
    sketch = count_min_sketch(keys, width=width, depth=depth)
    H_s = sketch_entropy(sketch)

    # 2. Global entropy of signal values
    signal_vals = [e.signal_value for e in entries]
    H_g = shannon_entropy(signal_vals)

    denom = H_g + H_s + 1e-12

    activity: Dict[str, float] = {}
    for e in entries:
        temp_k = c_to_k(e.temperature_c)
        r_i = developmental_rate(temp_k, params)
        I_i = fisher_information(e.signal_value)
        activity[e.surface_key] = r_i * I_i / denom
    return activity


def aggregate_features_via_sketch(
    entries: List[PheromoneEntry],
    width: int = 64,
    depth: int = 4,
) -> Dict[str, float]:
    """
    Produce a lightweight feature vector by:
    - Sketching surface keys.
    - Counting occurrences per sketch cell.
    - Normalising by total count.

    The resulting dict maps (depth, index) → normalized frequency.
    """
    sketch = count_min_sketch((e.surface_key for e in entries), width, depth)
    total = sum(sum(row) for row in sketch) + 1e-12
    features: Dict[str, float] = {}
    for d, row in enumerate(sketch):
        for idx, cnt in enumerate(row):
            if cnt > 0:
                key = f"cell_{d}_{idx}"
                features[key] = cnt / total
    return features


def curvature_estimate(activity: Dict[str, float]) -> float:
    """
    Very coarse Ollivier‑Ricci curvature analogue:
    curvature = 1 - (std_dev(activity) / mean(activity))
    Higher uniformity → curvature close to 1.
    """
    vals = np.fromiter(activity.values(), dtype=float)
    if vals.size == 0:
        return 0.0
    mean = vals.mean()
    std = vals.std()
    if mean == 0:
        return 0.0
    return 1.0 - (std / mean)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic pheromone entries
    random.seed(42)
    entries = []
    for i in range(30):
        entry = PheromoneEntry(
            surface_key=f"node_{random.randint(0, 5)}",
            signal_kind="odor",
            signal_value=random.gauss(0.0, 1.0),      # θ for Fisher info
            half_life_seconds=random.uniform(10, 100),
            temperature_c=random.uniform(15, 35)     # realistic ambient temp
        )
        entries.append(entry)

    # Hybrid activity computation
    act = hybrid_activity(entries)
    print("Hybrid activity (first 5):")
    for k, v in list(act.items())[:5]:
        print(f"  {k}: {v:.6f}")

    # Feature aggregation via sketch
    feats = aggregate_features_via_sketch(entries)
    print("\nSketch‑derived feature sample:")
    for k, v in list(feats.items())[:5]:
        print(f"  {k}: {v:.4f}")

    # Curvature estimate
    curv = curvature_estimate(act)
    print(f"\nCurvature estimate: {curv:.4f}")

    sys.exit(0)