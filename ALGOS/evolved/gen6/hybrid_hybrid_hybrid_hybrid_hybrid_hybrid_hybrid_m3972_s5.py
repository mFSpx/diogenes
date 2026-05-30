# DARWIN HAMMER — match 3972, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_model__hybrid_hybrid_hybrid_m1206_s8.py (gen4)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m892_s1.py (gen5)
# born: 2026-05-29T23:53:03Z

"""
Hybrid Module: fusion of hybrid_hybrid_hybrid_model__hybrid_hybrid_hybrid_m1206_s8.py (Parent A)
and hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m892_s1.py (Parent B).

Mathematical Bridge
-------------------
Parent A generates 1‑D time‑series signals (GPU‑memory and periodic) and provides an
SSIM component that compares two signals.  Parent B partitions a set of 2‑D points
into Voronoi cells, then evaluates a biological developmental‑rate function on the
cell‑center values.

The fusion treats each sampled index *i* together with its signal amplitude *s_i*
as a 2‑D point **p_i = (i, s_i)**.  These points are fed to the Voronoi routine,
producing regions whose centroids become surrogate “temperature” inputs for the
developmental‑rate equation.  Within each region we compute an SSIM score between
the original sub‑signal and a reconstructed version (here a simple moving‑average
approximation).  The SSIM scores and the developmental‑rate values are then
combined into a Shapley‑like contribution vector that quantifies each region’s
overall importance.

The resulting hybrid system therefore intertwines:
* signal generation & SSIM (linear algebra / statistics)   – from Parent A
* Voronoi partitioning, centroid extraction & biological rate – from Parent B
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – signal generation and SSIM utilities
# ----------------------------------------------------------------------
def generate_gpu_memory_signal(
    length: int = 256,
    base_mb: int = 4096,
    variance_mb: int = 512,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    sinusoid = np.sin(2 * math.pi * t / length)
    noise = rng.normal(0, variance_mb * 0.05, size=length)
    signal = base_mb + variance_mb * sinusoid + noise
    return signal.astype(np.float64)


def generate_periodic_signal(
    length: int = 256,
    amplitude: float = 500.0,
    frequency: float = 1.0,
    phase: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    signal = amplitude * np.sin(2 * math.pi * frequency * t / length + phase)
    jitter = rng.normal(0, amplitude * 0.01, size=length)
    return (signal + jitter).astype(np.float64)


def _ssim_component(
    x: np.ndarray, y: np.ndarray, C1: float = 6.5025, C2: float = 58.5225
) -> Tuple[float, float, float]:
    """Return luminance (l), contrast (c) and structure (s) components of SSIM."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    l = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    c = (2 * math.sqrt(sigma_x) * math.sqrt(sigma_y) + C2) / (sigma_x + sigma_y + C2)
    s = (sigma_xy + C2 / 2) / (math.sqrt(sigma_x) * math.sqrt(sigma_y) + C2 / 2)
    return l, c, s


def ssim_index(x: np.ndarray, y: np.ndarray) -> float:
    """Full SSIM index derived from its three components."""
    l, c, s = _ssim_component(x, y)
    return l * c * s


# ----------------------------------------------------------------------
# Parent B – Voronoi utilities and developmental rate model
# ----------------------------------------------------------------------
Vector = List[float]


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(
    points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]
) -> Dict[int, List[Tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


def developmental_rate(
    temp_k: float, params: SchoolfieldParams = SchoolfieldParams()
) -> float:
    """Schoolfield model for temperature‑dependent developmental rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError(
            "temperature must be Kelvin‑positive and rho_25 non‑negative"
        )
    numerator = params.rho_25 * (temp_k / 298.15) * np.exp(
        (params.delta_h_activation / params.r_cal)
        * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = np.exp(
        (params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k))
    )
    high = np.exp(
        (params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k))
    )
    return numerator / (1.0 + low + high)


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def build_voronoi_from_signal(signal: np.ndarray, seed_fraction: float = 0.1) -> Tuple[
    List[Tuple[float, float]], List[Tuple[float, float]], Dict[int, List[Tuple[float, float]]]
]:
    """
    Convert a 1‑D signal into 2‑D points (index, amplitude),
    pick a subset as Voronoi seeds, and assign points to regions.
    Returns (points, seeds, assignment_dict).
    """
    length = signal.shape[0]
    points = [(float(i), float(v)) for i, v in enumerate(signal)]

    # Choose seeds uniformly spaced across the index axis
    num_seeds = max(2, int(length * seed_fraction))
    step = length // num_seeds
    seeds = [points[i] for i in range(0, length, step)][:num_seeds]

    regions = assign(points, seeds)
    return points, seeds, regions


def region_metrics(
    regions: Dict[int, List[Tuple[float, float]]],
    original_signal: np.ndarray,
    window: int = 5,
) -> Dict[int, Dict[str, float]]:
    """
    For each Voronoi region compute:
    * mean amplitude (used as a surrogate temperature)
    * developmental rate at that temperature
    * SSIM between the original sub‑signal and a simple moving‑average reconstruction
    Returns a nested dict keyed by region index.
    """
    metrics: Dict[int, Dict[str, float]] = {}
    for idx, pts in regions.items():
        if not pts:
            continue
        # Extract amplitudes
        amps = np.array([p[1] for p in pts])
        mean_amp = float(np.mean(amps))

        # Map mean amplitude linearly to a temperature in Kelvin
        # Assume amplitude range roughly [0, 6000] -> Kelvin range [280, 320]
        temp_k = 280.0 + (mean_amp / 6000.0) * 40.0

        rate = developmental_rate(temp_k)

        # Build sub‑signal slice corresponding to indices in this region
        idxs = [int(p[0]) for p in pts]
        start, end = min(idxs), max(idxs) + 1
        sub_signal = original_signal[start:end]

        # Simple moving‑average reconstruction
        kernel = np.ones(window) / window
        recon = np.convolve(sub_signal, kernel, mode="same")

        # SSIM between original slice and reconstruction
        ssim_val = ssim_index(sub_signal, recon)

        metrics[idx] = {
            "mean_amplitude": mean_amp,
            "temperature_k": temp_k,
            "development_rate": rate,
            "ssim": ssim_val,
        }
    return metrics


def shapley_like_contributions(metrics: Dict[int, Dict[str, float]]) -> Dict[int, float]:
    """
    Produce a Shapley‑style contribution for each region based on a weighted
    sum of its developmental rate and SSIM score.  The weights are normalized
    so that contributions sum to 1.
    """
    raw_vals = {}
    for idx, m in metrics.items():
        # Weight: higher developmental rate and higher SSIM are both beneficial
        raw = m["development_rate"] * 0.6 + m["ssim"] * 0.4
        raw_vals[idx] = raw

    total = sum(raw_vals.values())
    if total == 0:
        # Avoid division by zero – assign equal share
        n = len(raw_vals)
        return {idx: 1.0 / n for idx in raw_vals}
    return {idx: val / total for idx, val in raw_vals.items()}


def hybrid_process(
    length: int = 256,
    seed: int | None = 42,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float]]:
    """
    End‑to‑end hybrid pipeline:
    1. Generate two correlated signals (GPU memory & periodic).
    2. Combine them (simple average) to obtain a composite signal.
    3. Build Voronoi partition from the composite.
    4. Compute region‑wise metrics.
    5. Derive Shapley‑like contributions.
    Returns (region_metrics, contributions).
    """
    gpu_sig = generate_gpu_memory_signal(length=length, seed=seed)
    per_sig = generate_periodic_signal(length=length, seed=seed + 1 if seed else None)

    # Composite signal – normalized to similar scale
    composite = (gpu_sig / np.max(gpu_sig) + per_sig / np.max(per_sig)) / 2.0

    points, seeds, regions = build_voronoi_from_signal(composite)

    metrics = region_metrics(regions, composite)
    contributions = shapley_like_contributions(metrics)

    return metrics, contributions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    m, contrib = hybrid_process()
    print("Region metrics (sample):")
    for idx in sorted(m)[:5]:
        print(f"  Region {idx}: {m[idx]}")
    print("\nShapley‑like contributions (sample):")
    for idx in sorted(contrib)[:5]:
        print(f"  Region {idx}: {contrib[idx]:.4f}")
    print("\nTotal contribution sum:", sum(contrib.values()))