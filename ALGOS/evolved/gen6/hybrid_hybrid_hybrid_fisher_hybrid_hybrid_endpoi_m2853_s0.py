# DARWIN HAMMER — match 2853, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_hybrid_m1864_s2.py (gen5)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_hybrid_diffus_m1445_s2.py (gen5)
# born: 2026-05-29T23:46:14Z

import json
import math
import random
import sys
from pathlib import Path
import numpy as np

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x:
        raise ValueError('samples must not be empty')
    n = len(x)
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))

def fractional_decay(alpha: float, t: float) -> float:
    return math.exp(-alpha * t)

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: np.ndarray, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if np.any(m <= 0) or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m[0], m[1], m[2])
    return (np.prod(m) ** b) * np.exp(k * fi) / neck_lever

def recovery_priority(m: np.ndarray, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def hybrid_operation(theta: np.ndarray, m: np.ndarray) -> np.ndarray:
    similarity = ssim(theta, m, k1=0.02, k2=0.04)
    weighted_fisher = fisher_score(theta, np.mean(m), np.std(m), eps=1e-14) * similarity
    return weighted_fisher

def adaptive_routing(theta: np.ndarray, m: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    decay = fractional_decay(alpha, np.sum(theta))
    return decay * m

def network_optimization(theta: np.ndarray, m: np.ndarray, max_index: float = 10.0) -> np.ndarray:
    priority = recovery_priority(m, max_index)
    return adaptive_routing(theta, m, priority)

if __name__ == "__main__":
    theta = np.array([1.0, 2.0, 3.0])
    m = np.array([4.0, 5.0, 6.0])
    result = network_optimization(theta, m)
    print(result)