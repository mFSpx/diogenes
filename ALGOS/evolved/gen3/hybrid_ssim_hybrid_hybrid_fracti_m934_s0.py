# DARWIN HAMMER — match 934, survivor 0
# gen: 3
# parent_a: ssim.py (gen0)
# parent_b: hybrid_hybrid_fractional_hd_hybrid_hoeffding_tre_m12_s2.py (gen2)
# born: 2026-05-29T23:31:38Z

"""
Hybrid algorithm fusing the structural similarity index (ssim) from ssim.py 
and the Hybrid Fractional-Hoeffding algorithm from hybrid_hybrid_fractional_hd_hybrid_hoeffding_tre_m12_s2.py. 
The mathematical bridge lies in applying the Fractional HDC's scalar causal effect estimates 
as the exponent in the Hoeffding bound calculation, thus quantifying uncertainty in both data distributions 
and causal relationships, while also using the ssim to measure the similarity between the original and 
reconstructed data after applying the Hybrid Fractional-Hoeffding algorithm.
"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import Sequence, Iterable, Optional

def ssim(x: Sequence[float], y: Sequence[float], dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x:
        raise ValueError('samples must not be empty')
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x) / n
    vy = sum((v - my) ** 2 for v in y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))

def random_hv(d: int = 10000, kind: str = "complex", seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        return v / (np.linalg.norm(v) + 1e-30)
    raise ValueError(f"Unsupported kind {kind!r}")

def bind(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y))

def unbind(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    FY = np.fft.fft(Y)
    mag = np.abs(FY)
    inv_FY = np.conj(FY) / (mag**2 + 1e-30)
    return np.real(np.fft.ifft(np.fft.fft(Z) * inv_FY))

def fractional_power(X: np.ndarray, alpha: float) -> np.ndarray:
    return np.abs(X)**alpha * np.sign(X)

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: return 0.0
    if xs[0] < 0: raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2*i-n-1)*x for i,x in enumerate(xs,1))/(n*sum(xs))

def hybrid_hoeffding_fractional(values: Iterable[float], best_gain: float, epsilon: float, delta: float, n: int) -> float:
    bound = hoeffding_bound(1, delta, n)
    gain_gap = best_gain - gini_coefficient(values)
    return gain_gap - bound

def hybrid_ssim_hoeffding(X: np.ndarray, Y: np.ndarray, epsilon: float, delta: float, n: int) -> float:
    bound = hoeffding_bound(1, delta, n)
    Z = bind(X, Y)
    reconstructed_X = unbind(Z, Y)
    ssim_value = ssim(X, reconstructed_X)
    gain_gap = ssim_value - epsilon
    return gain_gap - bound

def hybrid_fractional_ssim_hoeffding(X: np.ndarray, Y: np.ndarray, alpha: float, epsilon: float, delta: float, n: int) -> float:
    bound = hoeffding_bound(1, delta, n)
    Z = bind(X, Y)
    fractional_Z = fractional_power(Z, alpha)
    reconstructed_X = unbind(fractional_Z, Y)
    ssim_value = ssim(X, reconstructed_X)
    gain_gap = ssim_value - epsilon
    return gain_gap - bound

if __name__ == "__main__":
    X = random_hv(10000, "real")
    Y = random_hv(10000, "real")
    epsilon = 0.1
    delta = 0.05
    n = 100
    alpha = 0.5
    print(hybrid_ssim_hoeffding(X, Y, epsilon, delta, n))
    print(hybrid_fractional_ssim_hoeffding(X, Y, alpha, epsilon, delta, n))