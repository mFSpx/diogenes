# DARWIN HAMMER — match 3972, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_model__hybrid_hybrid_hybrid_m1206_s8.py (gen4)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m892_s1.py (gen5)
# born: 2026-05-29T23:53:03Z

"""
Module hybrid_voronoi_ssim_decision_fusion: A hybrid algorithm combining the Voronoi partition
and radial-basis surrogate model from hybrid_hybrid_voronoi_parti_hybrid_rbf_surrogate_m85_s0.py
with the decision-making and SSIM-based quality assessment from hybrid_hybrid_hybrid_model__hybrid_hybrid_hybrid_m1206_s8.py.
The mathematical bridge between the two structures lies in the use of Voronoi cell centers as 
surrogate model centers for decision-making, and the application of SSIM-based quality assessment 
to evaluate feature contributions in the Voronoi-partitioned space.

"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import Tuple, List, Dict
from dataclasses import dataclass

Vector = List[float]

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
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

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * np.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = np.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = np.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  

TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")

def generate_gpu_memory_signal(length: int = 256,
                               base_mb: int = 4096,
                               variance_mb: int = 512,
                               seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    sinusoid = np.sin(2 * math.pi * t / length)
    noise = rng.normal(0, variance_mb * 0.05, size=length)
    signal = base_mb + variance_mb * sinusoid + noise
    return signal.astype(np.float64)

def _ssim_component(x: np.ndarray, y: np.ndarray, C1: float, C2: float) -> Tuple[float, float, float]:
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    l = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    c = (2 * math.sqrt(sigma_x) * math.sqrt(sigma_y) + C2) / (sigma_x + sigma_y + C2)
    s = (sigma_xy + C2 / 2) / (math.sqrt(sigma_x) * math.sqrt(sigma_y) + C2 / 2)
    return l, c, s

def voronoi_ssim_decision_fusion(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], tier: ModelTier, seed: int | None = None) -> Tuple[np.ndarray, Dict[int, List[Tuple[float, float]]]]:
    rng = np.random.default_rng(seed)
    regions = assign(points, seeds)
    signals = []
    for region in regions.values():
        signal = generate_gpu_memory_signal(len(region), base_mb=tier.ram_mb, seed=seed)
        signals.append(signal)
    signals = np.array(signals)
    ssim_scores = []
    for i in range(len(signals)):
        for j in range(i+1, len(signals)):
            l, c, s = _ssim_component(signals[i], signals[j], C1=0.01, C2=0.03)
            ssim_scores.append((i, j, l, c, s))
    return np.array(ssim_scores), regions

def voronoi_decision_fusion(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], tier: ModelTier, seed: int | None = None) -> Dict[int, List[Tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    regions = assign(points, seeds)
    signals = []
    for region in regions.values():
        signal = generate_gpu_memory_signal(len(region), base_mb=tier.ram_mb, seed=seed)
        signals.append(signal)
    signals = np.array(signals)
    scores = []
    for i in range(len(signals)):
        for j in range(i+1, len(signals)):
            l, c, s = _ssim_component(signals[i], signals[j], C1=0.01, C2=0.03)
            scores.append((i, j, l, c, s))
    return regions

def developmental_rate_ssim(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], tier: ModelTier, seed: int | None = None) -> float:
    rng = np.random.default_rng(seed)
    regions = assign(points, seeds)
    signals = []
    for region in regions.values():
        signal = generate_gpu_memory_signal(len(region), base_mb=tier.ram_mb, seed=seed)
        signals.append(signal)
    signals = np.array(signals)
    ssim_scores = []
    for i in range(len(signals)):
        for j in range(i+1, len(signals)):
            l, c, s = _ssim_component(signals[i], signals[j], C1=0.01, C2=0.03)
            ssim_scores.append((i, j, l, c, s))
    developmental_rate_scores = []
    for temp_k in np.linspace(273.15, 307.15, 100):
        score = developmental_rate(temp_k, SchoolfieldParams())
        for i in range(len(signals)):
            for j in range(i+1, len(signals)):
                l, c, s = _ssim_component(signals[i], signals[j], C1=0.01, C2=0.03)
                developmental_rate_scores.append((temp_k, score, l, c, s))
    return developmental_rate_scores

if __name__ == "__main__":
    points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    seeds = [(0.5, 0.5), (1.5, 1.5)]
    tier = TIER_T2_REASONING
    seed = 42
    _, regions = voronoi_ssim_decision_fusion(points, seeds, tier, seed)
    print(regions)
    print(developmental_rate_ssim(points, seeds, tier, seed))