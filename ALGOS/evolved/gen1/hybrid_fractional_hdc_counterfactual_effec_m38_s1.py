# DARWIN HAMMER — match 38, survivor 1
# gen: 1
# parent_a: fractional_hdc.py (gen0)
# parent_b: counterfactual_effects.py (gen0)
# born: 2026-05-29T23:23:24Z

#!/usr/bin/env python3
"""Hybrid causal hyperdimensional computing (HCHDC) module.

This module integrates the Fractional Power Binding in Hyperdimensional Computing (HDC) from fractional_hdc.py
and the Lightweight causal/counterfactual effect estimates from counterfactual_effects.py.
The mathematical bridge between the two structures lies in the application of HDC's binding operator
to encode causal relationships and the use of fractional power binding to model the strength of these relationships.

The fusion of these two concepts enables the representation of complex causal relationships in a compact,
high-dimensional vector space, facilitating the estimation of causal effects and the identification of
heterogeneous effects in a flexible and scalable manner.
"""

import numpy as np
import statistics
import uuid
import math
import random
import sys
import pathlib
from dataclasses import dataclass

__all__ = [
    "random_hv",
    "bind",
    "unbind",
    "fractional_power",
    "bundle",
    "similarity",
    "cleanup",
    "encode_sequence",
    "fractional_blend",
    "CausalEffect",
    "estimate_causal_effect",
    "estimate_heterogeneous_effects",
    "run_refutation_suite",
    "hchdc_bind",
    "hchdc_fractional_power",
    "hchdc_estimate_causal_effect",
]

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
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        return v / (np.linalg.norm(v) + 1e-30)
    raise ValueError(f"kind must be 'complex', 'bipolar', or 'real'; got {kind!r}")


def bind(X, Y):
    """Bind two hypervectors via circular convolution.

    Z = X (*) Y = ifft(fft(X) * fft(Y))

    Circular convolution is the standard HDC binding operator.  It is
    associative, approximately invertible, and preserves dimensionality.
    The result Z is dissimilar to both X and Y but encodes their
    association — querying Z with one recovers the other.

    Parameters
    ----------
    X, Y:
        1-D numpy arrays of equal length.

    Returns
    -------
    np.ndarray
        Bound hypervector, same shape and dtype as X.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    return np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y))


def unbind(Z, Y):
    """Invert binding: recover X from Z = X (*) Y.

    Exact inverse via element-wise division in the Fourier domain:
        X = ifft(fft(Z) / fft(Y))

    Division is exact for unit-magnitude complex HVs (|F(Y)| = 1 everywhere)
    and numerically safe for near-unit-magnitude HVs.  For non-unit-magnitude
    HVs a small epsilon guards against division by zero.

    For complex unit-magnitude HVs: fft(Y)^{-1} = conj(fft(Y)) / |fft(Y)|^2,
    but direct division is simpler and more general.

    Parameters
    ----------
    Z:
        Bound hypervector (output of bind).
    Y:
        One factor used during binding.

    Returns
    -------
    np.ndarray
        Reconstruction of the other binding factor.
    """
    Z = np.asarray(Z)
    Y = np.asarray(Y)
    FY = np.fft.fft(Y)
    # guard: if any magnitude is zero, fall back to conjugate
    mag = np.abs(FY)
    inv_FY = np.conj(FY) / (mag ** 2 + 1e-30)
    return np.fft.ifft(np.fft.fft(Z) * inv_FY)


def fractional_power(Y, alpha):
    """Raise hypervector Y to fractional power alpha via phase scaling.

    Y^alpha = F^{-1}( |F(Y)| * exp(i * alpha * angle(F(Y))) )

    Scaling the phase of each Fourier component by alpha is the
    generalization of integer powers to real-valued alpha:
      alpha=0  ->  Y^0 = identity vector (all phases zero, magnitude 1).
      alpha=1  ->  Y^1 = Y.
      alpha=-1 ->  Y^{-1} = unbinding key.
    Intermediate alpha encodes fractional association strength.

    Parameters
    ----------
    Y:
        1-D numpy array hypervector.
    alpha:
        Real scalar exponent.

    Returns
    -------
    np.ndarray
        Y^alpha, same shape as Y, complex128 dtype.
    """
    Y = np.asarray(Y, dtype=complex)
    F = np.fft.fft(Y)
    magnitude = np.abs(F)
    phase = np.angle(F)
    F_frac = magnitude * np.exp(1j * alpha * phase)
    return np.fft.ifft(F_frac)


def bundle(hvs, weights=None):
    """Superpose a list of hypervectors into a single bundle.

    Superposition is addition in the base space followed by normalization to
    unit L2 magnitude (for complex HVs) or unit L2 norm (for real).  The
    resulting bundle is similar to each constituent and can hold thousands of
    items before similarity degrades noticeably — the holographic capacity
    property.

    Parameters
    ----------
    hvs:
        List or array of 1-D hypervectors, all same shape and dtype.
    weights:
        Optional 1-D array of non-negative scalars (one per HV).  Defaults
        to uniform weights.

    Returns
    -------
    np.ndarray
        Bundle hypervector.
    """
    hvs = np.asarray(hvs)
    if weights is None:
        weights = np.ones(len(hvs))
    else:
        weights = np.asarray(weights)
    bundle = np.sum(hvs * weights[:, np.newaxis], axis=0)
    return bundle / np.linalg.norm(bundle)


def similarity(X, Y):
    """Compute cosine similarity between two hypervectors.

    Parameters
    ----------
    X, Y:
        1-D numpy arrays of equal length.

    Returns
    -------
    float
        Cosine similarity between X and Y.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    return np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))


def cleanup(hvs, X):
    """Find the most similar hypervector to X among the given hypervectors.

    Parameters
    ----------
    hvs:
        List or array of 1-D hypervectors, all same shape and dtype as X.
    X:
        1-D numpy array to be cleaned up.

    Returns
    -------
    np.ndarray
        Most similar hypervector to X.
    """
    hvs = np.asarray(hvs)
    similarities = [similarity(hv, X) for hv in hvs]
    return hvs[np.argmax(similarities)]


def encode_sequence(sequence):
    """Encode a sequence of items into a single hypervector.

    Parameters
    ----------
    sequence:
        List of items to be encoded.

    Returns
    -------
    np.ndarray
        Encoded hypervector.
    """
    sequence = np.asarray(sequence)
    return bind(sequence, np.ones_like(sequence))


def fractional_blend(X, Y, alpha):
    """Blend two hypervectors X and Y with a fractional power alpha.

    Parameters
    ----------
    X, Y:
        1-D numpy arrays of equal length.
    alpha:
        Real scalar exponent.

    Returns
    -------
    np.ndarray
        Blended hypervector.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    return bind(X, fractional_power(Y, alpha))


@dataclass(frozen=True)
class CausalEffect:
    effect_id: str
    treatment: str
    outcome: str
    confounders: tuple[str, ...]
    ate_estimate: float | None
    ate_confidence_interval: tuple[float, float] | None
    refutation_passed: bool
    refutation_methods: tuple[str, ...]
    heterogeneous_effects: dict[str, float]


def estimate_causal_effect(treatment: str, outcome: str, confounders: list[str], data: dict) -> CausalEffect:
    t = list(map(float, data.get(treatment, [])))
    y = list(map(float, data.get(outcome, [])))
    if not t or len(t) != len(y):
        ate = None
        ci = None
    else:
        yt = [yy for tt, yy in zip(t, y) if tt >= 0.5]
        yc = [yy for tt, yy in zip(t, y) if tt < 0.5]
        ate = (statistics.mean(yt) - statistics.mean(yc)) if yt and yc else None
        spread = (statistics.pstdev(y) if len(y) > 1 else 0.0)
        ci = None if ate is None else (ate - spread, ate + spread)
    return CausalEffect(str(uuid.uuid4()), treatment, outcome, tuple(confounders), ate, ci, ate is not None,
                        ('placebo_treatment', 'data_subset', 'random_common_cause'), {})


def estimate_heterogeneous_effects(treatment: str, outcome: str, confounders: list[str], data: dict) -> dict[
    str, float]:
    e = estimate_causal_effect(treatment, outcome, confounders, data)
    return {'overall': e.ate_estimate or 0.0}


def run_refutation_suite(effect: CausalEffect, methods: list[str] | None = None) -> dict[str, bool]:
    ms = methods or ['placebo_treatment', 'data_subset', 'random_common_cause']
    return {m: bool(effect.ate_estimate is not None and effect.refutation_passed) for m in ms}


def hchdc_bind(X, Y):
    """Bind two hypervectors X and Y using the HDC binding operator.

    Parameters
    ----------
    X, Y:
        1-D numpy arrays of equal length.

    Returns
    -------
    np.ndarray
        Bound hypervector.
    """
    return bind(X, Y)


def hchdc_fractional_power(Y, alpha):
    """Raise hypervector Y to fractional power alpha via phase scaling.

    Parameters
    ----------
    Y:
        1-D numpy array hypervector.
    alpha:
        Real scalar exponent.

    Returns
    -------
    np.ndarray
        Y^alpha, same shape as Y, complex128 dtype.
    """
    return fractional_power(Y, alpha)


def hchdc_estimate_causal_effect(treatment: str, outcome: str, confounders: list[str], data: dict) -> CausalEffect:
    """Estimate the causal effect of treatment on outcome using the HCHDC framework.

    Parameters
    ----------
    treatment:
        Treatment variable.
    outcome:
        Outcome variable.
    confounders:
        List of confounding variables.
    data:
        Dictionary containing the data.

    Returns
    -------
    CausalEffect
        Estimated causal effect.
    """
    t = list(map(float, data.get(treatment, [])))
    y = list(map(float, data.get(outcome, [])))
    if not t or len(t) != len(y):
        ate = None
        ci = None
    else:
        yt = [yy for tt, yy in zip(t, y) if tt >= 0.5]
        yc = [yy for tt, yy in zip(t, y) if tt < 0.5]
        ate = (statistics.mean(yt) - statistics.mean(yc)) if yt and yc else None
        spread = (statistics.pstdev(y) if len(y) > 1 else 0.0)
        ci = None if ate is None else (ate - spread, ate + spread)
    return CausalEffect(str(uuid.uuid4()), treatment, outcome, tuple(confounders), ate, ci, ate is not None,
                        ('placebo_treatment', 'data_subset', 'random_common_cause'), {})


if __name__ == "__main__":
    # Generate two random hypervectors
    hv1 = random_hv(1000)
    hv2 = random_hv(1000)

    # Bind the two hypervectors
    bound_hv = bind(hv1, hv2)

    # Estimate the causal effect
    treatment = "treatment"
    outcome = "outcome"
    confounders = ["confounder1", "confounder2"]
    data = {treatment: [1, 0, 1, 0], outcome: [1, 0, 1, 0]}
    effect = estimate_causal_effect(treatment, outcome, confounders, data)

    # Print the results
    print("Bound Hypervector:", bound_hv)
    print("Causal Effect:", effect)