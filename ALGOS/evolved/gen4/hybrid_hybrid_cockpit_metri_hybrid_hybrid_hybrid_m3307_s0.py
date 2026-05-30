# DARWIN HAMMER — match 3307, survivor 0
# gen: 4
# parent_a: hybrid_cockpit_metrics_rectified_flow_m10_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s4.py (gen3)
# born: 2026-05-29T23:49:04Z

"""
Hybrid module unifying the cockpit metrics with rectified flow and Caputo kernel utilities.
The mathematical bridge between the two parents is the use of the Caputo kernel to model 
the fractional memory in the rectified flow, while the cockpit metrics provide a scalar 
*trust* value to modulate the velocity field. This results in a trust-weighted velocity 
field that incorporates both the linear transport of rectified flow and the evidence-coverage 
quality of the cockpit.
"""

import math
from typing import Callable, Tuple
import numpy as np
import random
import sys
from pathlib import Path

# Parent A – cockpit metrics
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, 
                claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known-good, clamped to [0, 1]."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))


def audit_debt(exports_missing_audit_step: int, total_exports: int) -> float:
    """Ratio of exports missing audit step, clamped to [0, 1]."""
    return 1.0 if total_exports <= 0 else max(0.0, min(1.0, 
                exports_missing_audit_step / total_exports))


# Parent B – Caputo kernel utilities
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857
])


def _gamma(z: float) -> float:
    """Lanczos approximation of the Gamma function."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * _gamma(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x


def caputo_kernel(alpha: float, t: np.ndarray) -> np.ndarray:
    """
    Compute the raw (unnormalized) Caputo kernel values for a vector of time indices.
    The kernel is t^{alpha-1} / Gamma(alpha).
    """
    if alpha <= 0:
        raise ValueError("Fractional order alpha must be positive.")
    # avoid t=0 when alpha<1 (singularity) by starting at 1
    t = np.where(t == 0, 1e-12, t)
    return t ** (alpha - 1) / _gamma(alpha)


# Hybrid functions
def hybrid_velocity_field(x0: np.ndarray, x1: np.ndarray, trust: float) -> np.ndarray:
    """Trust-weighted velocity field."""
    return trust * (x1 - x0)


def hybrid_caputo_kernel(alpha: float, t: np.ndarray, trust: float) -> np.ndarray:
    """
    Compute the trust-weighted Caputo kernel values for a vector of time indices.
    """
    return trust * caputo_kernel(alpha, t)


def hybrid_flow_target(x0: np.ndarray, x1: np.ndarray, trust: float, alpha: float, t: np.ndarray) -> np.ndarray:
    """
    Compute the hybrid flow target, incorporating both the rectified flow and Caputo kernel.
    """
    velocity_field = hybrid_velocity_field(x0, x1, trust)
    caputo_kernel_values = hybrid_caputo_kernel(alpha, t, trust)
    return velocity_field + caputo_kernel_values


if __name__ == "__main__":
    x0 = np.array([1.0, 2.0])
    x1 = np.array([3.0, 4.0])
    trust = 0.5
    alpha = 0.7
    t = np.array([1.0, 2.0, 3.0])
    print(hybrid_flow_target(x0, x1, trust, alpha, t))