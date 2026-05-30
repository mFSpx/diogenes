# DARWIN HAMMER — match 1139, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_pheromone_hyb_hybrid_hybrid_hybrid_m152_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s1.py (gen3)
# born: 2026-05-29T23:33:00Z

import math
import numpy as np
import random
import sys
import pathlib

class Morphology:
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def pheromone_decay(v0: float, half_life_seconds: int, delta_t: int) -> float:
    tau = half_life_seconds / 3600
    return v0 * (0.5 ** (delta_t / tau))

def ssim(x: list[float], y: list[float], dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    c1 = (dynamic_range ** 2) * (k1 ** 2)
    c2 = (dynamic_range ** 2) * (k2 ** 2)
    k = sigma_x * sigma_y + c1 * (mu_x - mu_y) ** 2 + c2 * (sigma_x ** 2 + sigma_y ** 2)
    return (2 * mu_x * mu_y + c1 * (mu_x - mu_y) ** 2 + c2 * (sigma_x ** 2 + sigma_y ** 2)) / (
        mu_x ** 2 + mu_y ** 2 + c1 * (mu_x - mu_y) ** 2 + c2 * (sigma_x ** 2 + sigma_y ** 2)
    )

def hybrid_ssim_pheromone_decay(x: list[float], y: list[float], v0: float, half_life_seconds: int, delta_t: int, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    ssim_value = ssim(x, y, dynamic_range, k1, k2)
    return pheromone_decay(v0, half_life_seconds, delta_t) * ssim_value

def hybrid_righting_time_ssim(m: Morphology, x: list[float], y: list[float], dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    righting_time = righting_time_index(m)
    ssim_value = ssim(x, y, dynamic_range, k1, k2)
    return righting_time * ssim_value

def hybrid_recovery_priority_ssim(m: Morphology, x: list[float], y: list[float], dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    recovery_priority_value = recovery_priority(m)
    ssim_value = ssim(x, y, dynamic_range, k1, k2)
    return recovery_priority_value * ssim_value

if __name__ == "__main__":
    m = Morphology(1.0, 2.0, 3.0, 4.0)
    x = [1.0, 2.0, 3.0, 4.0]
    y = [5.0, 6.0, 7.0, 8.0]
    v0 = 1.0
    half_life_seconds = 3600
    delta_t = 1
    print(hybrid_ssim_pheromone_decay(x, y, v0, half_life_seconds, delta_t))
    print(hybrid_righting_time_ssim(m, x, y))
    print(hybrid_recovery_priority_ssim(m, x, y))