# DARWIN HAMMER — match 2639, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s1.py (gen3)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s5.py (gen1)
# born: 2026-05-29T23:43:10Z

"""
This module presents a hybrid algorithm that combines the TTT-Linear weight matrix 
from the hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s1.py with the Caputo power-law 
kernel from the hybrid_caputo_fractional_minimum_cost_tree_m35_s5.py. The mathematical 
bridge between these structures is found in their common use of weighted sums and 
matrix operations. The hybrid module defines a new class HybridMatrix, which 
incorporates the Caputo power-law kernel into the TTT-Linear weight matrix. This 
allows for the optimization of matrix operations under the influence of fractional 
memory.

Functions:
  - HybridMatrix: a class representing a weighted matrix with Caputo power-law kernel
  - hybrid_ttt_step: perform a single step of the TTT-Linear algorithm with Caputo power-law 
    kernel
  - hybrid_ssim: compute the SSIM metric with Caputo power-law kernel
  - caputo_ttt_derivative: compute the Caputo derivative of the TTT-Linear weight matrix
"""

import numpy as np
import math
import random
import sys
import pathlib
import json
from datetime import datetime, timezone
from pathlib import Path

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def parse_context(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target=None):
    if target is None:
        target = x
    return np.sum((W @ x - target) ** 2)

def ttt_grad(W, x, target=None):
    if target is None:
        target = x
    return 2 * (W @ x - target) @ x.T

def ttt_step(W, x, eta, target=None):
    grad = ttt_grad(W, x, target)
    return W - eta * grad

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x.size:
        raise ValueError('sample must not be empty')

def gamma_lanczos(z):
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
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    else:
        return math.sqrt(2 * math.pi) * (z + _LANCZOS_G + 0.5) ** (z + 0.5) \
               * math.exp(-(z + _LANCZOS_G + 0.5)) \
               * _LANCZOS_C.sum(axis=0) / math.prod(z + i for i in range(_LANCZOS_G + 1))

def caputo_derivative(alpha, f, t, dt):
    return (1 / (math.gamma(1 - alpha) * dt ** alpha)) * (f(t) - sum([((t - i * dt) ** (-alpha - 1)) / math.gamma(-alpha) * f(i * dt) for i in range(1, int(t / dt) + 1)]))

def hybrid_ttt_step(W, x, eta, alpha, target=None):
    grad = ttt_grad(W, x, target)
    return W - eta * caputo_derivative(alpha, lambda t: grad, 1, 1)

def hybrid_ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03, alpha: float = 1.0) -> float:
    return ssim(x, y, dynamic_range, k1, k2) * caputo_derivative(alpha, lambda t: 1, 1, 1)

def caputo_ttt_derivative(W, x, alpha, target=None):
    return caputo_derivative(alpha, lambda t: ttt_grad(W, x, target), 1, 1)

if __name__ == "__main__":
    W = init_ttt(10)
    x = np.random.rand(10)
    eta = 0.01
    alpha = 0.5
    print(hybrid_ttt_step(W, x, eta, alpha))
    print(hybrid_ssim(x, x, alpha=alpha))
    print(caputo_ttt_derivative(W, x, alpha))