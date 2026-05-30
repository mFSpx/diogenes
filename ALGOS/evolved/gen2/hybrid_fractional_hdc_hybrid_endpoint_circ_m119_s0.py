# DARWIN HAMMER — match 119, survivor 0
# gen: 2
# parent_a: fractional_hdc.py (gen0)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s2.py (gen1)
# born: 2026-05-29T23:25:40Z

#!/usr/bin/env python3
"""Hybrid Power Binding with End-Point Morphology Fusion.

This module fuses two mathematical algorithms:

* Fractional Power Binding in Hyperdimensional Computing (HDC)
  (fractional_hdc.py)
* Hybrid Endpoint Morphology Pool (hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s2.py)

The mathematical bridge is a health score that multiplies a normalized
circuit-breaker reliability term with the complementary recovery priority
derived from the endpoint's morphology.  The health score therefore encodes
both operational reliability and intrinsic self-righting ability.

The fusion combines the fractional power binding from HDC with the geometric
indices from the endpoint morphology pool.  The resulting health score is
computed as a dot product between the fractional power bound vector and the
normalized geometric indices vector.

Clifford geometric product in geometric_product.py is richer than circular
convolution, but both operate in the same paradigm of coordinate-free
superposition algebras.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def random_hv(d=10000, kind="complex", seed=None):
    """Generate a random hypervector of dimension d.

    Parameters
    ----------
    d:
        Dimension of the hypervector.
    kind:
        "complex"  — unit-magnitude complex vector (each component e^{i*theta},
                     theta ~ Uniform[0, 2pi]).  These are the natural carriers
                     for phase-based fractional binding.
        "bipolar"  — real vector with each component in {+1, -1}.
        "real"     — Gaussian sample normalized to unit L2 norm.
    seed:
        Integer seed for reproducibility; None for random.

    Returns
    -------
    np.ndarray
        Shape (d,).  dtype=complex128 for kind="complex", float64 otherwise.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    elif kind == "bipolar":
        return np.random.choice([-1, 1], size=d)
    else:
        return (rng.normal(size=d) / np.linalg.norm(rng.normal(size=d)))

def health_score(xv: np.ndarray, yv: np.ndarray, alpha: float) -> float:
    """Compute the health score as a dot product between fractional power bound vector
    and normalized geometric indices vector.

    Parameters
    ----------
    xv:
        Fractional power bound vector.
    yv:
        Normalized geometric indices vector.
    alpha:
        Fractional power binding exponent.

    Returns
    -------
    float:
        Health score.
    """
    return np.dot(xv, yv) * np.exp(-1j * alpha * np.angle(yv))

def geometric_indices(xv: np.ndarray) -> np.ndarray:
    """Compute geometric indices (flatness, sphericity) from a vector.

    Parameters
    ----------
    xv:
        Input vector.

    Returns
    -------
    np.ndarray:
        Shape (2,).  Geometric indices.
    """
    # Compute dot products between the input vector and its outer products.
    dot_products = np.einsum('i,ij->j', xv, xv[:, None])[:, None]
    dot_products = np.concatenate((np.eye(xv.size), 1 / 2 * np.ones((1, xv.size))), axis=0)
    dot_products = dot_products / dot_products.sum(axis=-1, keepdims=True)
    return dot_products

def endpoint_morphology(xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
    """Compute the endpoint morphology as geometric indices.

    Parameters
    ----------
    xv:
        Input vector.
    yv:
        Input vector.

    Returns
    -------
    np.ndarray:
        Shape (2,).  Geometric indices.
    """
    return geometric_indices(xv + yv)

def hybrid_power_binding(xv: np.ndarray, yv: np.ndarray, alpha: float) -> np.ndarray:
    """Compute the hybrid power binding between two vectors.

    Parameters
    ----------
    xv:
        Input vector.
    yv:
        Input vector.
    alpha:
        Fractional power binding exponent.

    Returns
    -------
    np.ndarray:
        Shape (d,).  Hybrid power bound vector.
    """
    return xv * (yv ** alpha)

def cleanup(xv: np.ndarray, yv: np.ndarray, alpha: float) -> np.ndarray:
    """Perform nearest-stored-vector lookup using the hybrid power bound vector.

    Parameters
    ----------
    xv:
        Input vector.
    yv:
        Input vector.
    alpha:
        Fractional power binding exponent.

    Returns
    -------
    np.ndarray:
        Shape (d,).  Cleaned up vector.
    """
    return np.argmin(np.abs(np.dot(xv, hybrid_power_binding(xv, yv, alpha))))

def encode_sequence(xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
    """Encode a sequence of vectors using the endpoint morphology.

    Parameters
    ----------
    xv:
        Input vector.
    yv:
        Input vector.

    Returns
    -------
    np.ndarray:
        Shape (n, d).  Encoded sequence.
    """
    return np.array([endpoint_morphology(xv, yv)])

def hybrid_endpoint_morphology(xv: np.ndarray, yv: np.ndarray, alpha: float) -> np.ndarray:
    """Compute the hybrid endpoint morphology as a dot product between the
    fractional power bound vector and the normalized geometric indices vector.

    Parameters
    ----------
    xv:
        Input vector.
    yv:
        Input vector.
    alpha:
        Fractional power binding exponent.

    Returns
    -------
    np.ndarray:
        Shape (1,).  Hybrid endpoint morphology.
    """
    return health_score(xv, yv, alpha)

if __name__ == "__main__":
    np.random.seed(0)
    xv = random_hv(d=100, kind="complex")
    yv = random_hv(d=100, kind="complex")
    alpha = 0.5
    hybrid_power_bound_vector = hybrid_power_binding(xv, yv, alpha)
    endpoint_morphology_vector = endpoint_morphology(xv, yv)
    hybrid_endpoint_morphology_value = hybrid_endpoint_morphology(xv, yv, alpha)
    print(hybrid_power_bound_vector)
    print(endpoint_morphology_vector)
    print(hybrid_endpoint_morphology_value)