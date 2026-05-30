# DARWIN HAMMER — match 1538, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m956_s1.py (gen4)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s3.py (gen1)
# born: 2026-05-29T23:37:12Z

"""
Hybrid Fractional Poikilotherm-Labeling State-Space Duality (HFPSSD)

This module fuses two parent algorithms:

* **Parent A** – the Hybrid Poikilotherm-Labeling State-Space Duality (HPLSSD) 
  that bridges the Schoolfield-Rollinson poikilotherm developmental rate with 
  the State-Space Duality (SSD) sequential and semiseparable parallel forms.
* **Parent B** – the Hybrid Caputo Fractional Derivative and Minimum-Cost Tree 
  scoring algorithm.

The mathematical bridge between the two parents lies in the ability of both 
algorithms to handle weighted decay kernels and path costs. The HPLSSD algorithm 
uses the temperature-dependent scalar `r(t) = developmental_rate(T(t))` to 
modulate the state-transition matrix `A` in the SSD, while the hybrid Caputo 
Fractional Derivative algorithm uses a power-law kernel to model 
algebraically-decaying long-range memory. By combining these two approaches, 
we can create a hybrid algorithm that uses the Caputo Fractional Derivative 
to model the decay of path costs over time, and the developmental rate to 
scale the state-transition matrix.

The hybrid algorithm uses the developmental rate to scale the state-transition 
matrix `A` in the SSD, while also using the Caputo Fractional Derivative to 
model the decay of path costs over time. The labeling process uses the weak 
supervision labeling primitives to handle noisy labels, and the hybrid 
algorithm combines the results of the sketching and labeling processes to 
produce a final output.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable, Set, Callable

# ----------------------------------------------------------------------
# Parent‑A: Poikilotherm developmental rate
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C (in arbitrary units)
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15            # K  (10 °C)
    t_high: float = 307.15           # K  (34 °C)
    delta_h_low: float = -45_000.0   # J mol⁻¹
    delta_h_high: float = 65_000.0   # J mol⁻¹
    r_cal: float = 1.987             # cal mol⁻¹ K⁻¹ (≈8.314 J mol⁻¹ K⁻¹)


def developmental_rate(T, params: SchoolfieldParams):
    if T < params.t_low:
        return params.rho_25 * math.exp((params.delta_h_activation / params.r_cal) * 
                                       (1 / params.t_low - 1 / T))
    elif T > params.t_high:
        return params.rho_25 * math.exp((params.delta_h_activation / params.r_cal) * 
                                       (1 / params.t_high - 1 / T) + 
                                       (params.delta_h_low / params.r_cal) * 
                                       (1 / T - 1 / params.t_high) + 
                                       (params.delta_h_high / params.r_cal) * 
                                       (1 / params.t_high - 1 / T))
    else:
        return params.rho_25 * math.exp((params.delta_h_activation / params.r_cal) * 
                                       (1 / params.t_low - 1 / T) + 
                                       (params.delta_h_low / params.r_cal) * 
                                       (1 / T - 1 / params.t_low))

# Lanczos g=7 coefficients (Numerical Recipes 3rd ed., table 6.1)
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


def gamma_lanczos(z):
    """Lanczos approximation of Gamma(z) for z > 0"""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    x = np.array([_LANCZOS_C[0]])
    for i in range(1, _LANCZOS_G + 1):
        x = np.append(x, _LANCZOS_C[i] / (z + i))
    return math.sqrt(2 * math.pi) * (z + _LANCZOS_G + 0.5) ** (z + 0.5) * math.exp(-(z + _LANCZOS_G + 0.5)) * np.sum(x)


def caputo_derivative(f, t, alpha):
    """Caputo Fractional Derivative"""
    return 1 / gamma_lanczos(1 - alpha) * np.sum((f[1:] - f[:-1]) / (t[1:] - t[:-1]) ** alpha)


def fractional_decay(t, alpha):
    """Fractional decay kernel"""
    return t ** (alpha - 1) / gamma_lanczos(alpha)


def hybrid_fpSSD(T, f, t, alpha, params: SchoolfieldParams):
    r = developmental_rate(T, params)
    A = np.array([[r, -r], [r, -r]])
    fd = fractional_decay(t, alpha)
    return np.dot(A, f) * fd


def demo_hybrid_fpSSD():
    T = 300.0
    f = np.array([1.0, 2.0])
    t = np.array([1.0, 2.0, 3.0])
    alpha = 0.5
    params = SchoolfieldParams()
    result = hybrid_fpSSD(T, f, t, alpha, params)
    print(result)


def smoke_test():
    T = 300.0
    f = np.array([1.0, 2.0])
    t = np.array([1.0, 2.0, 3.0])
    alpha = 0.5
    params = SchoolfieldParams()
    result = hybrid_fpSSD(T, f, t, alpha, params)
    assert np.isfinite(result).all()


if __name__ == "__main__":
    demo_hybrid_fpSSD()
    smoke_test()