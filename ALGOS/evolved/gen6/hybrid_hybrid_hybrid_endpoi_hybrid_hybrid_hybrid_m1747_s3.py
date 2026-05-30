# DARWIN HAMMER — match 1747, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m571_s1.py (gen5)
# born: 2026-05-29T23:38:40Z

import math
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Core mathematical utilities
# ----------------------------------------------------------------------
def _stable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    sum_exp = np.sum(exp_x)
    if sum_exp == 0:
        # fallback to uniform distribution
        return np.full_like(x, 1.0 / x.size, dtype=float)
    return exp_x / sum_exp


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """
    Gaussian “beam” intensity.

    Parameters
    ----------
    theta : float
        Evaluation point.
    center : float
        Beam centre.
    width : float
        Standard deviation (must be > 0).

    Returns
    -------
    float
        Intensity in the range (0, 1].
    """
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_information_gaussian(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single‑parameter Gaussian model with respect to its mean.

    The analytical expression is 1/σ², but we keep the full
    derivative‑based form to stay consistent with the original code
    while protecting against division by zero.

    Parameters
    ----------
    theta, center, width : float
        Same meaning as in :func:`gaussian_beam`.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Non‑negative Fisher information.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    # derivative of the log‑likelihood w.r.t. the mean μ
    dlog = -(theta - center) / (width * width)
    # Fisher information = E[(dlog)²] = (dlog)² because intensity integrates to 1
    return (dlog * dlog) * intensity


# ----------------------------------------------------------------------
# Domain objects
# ----------------------------------------------------------------------
class Action:
    """
    Simple container for an action used by the allocation routine.

    Attributes
    ----------
    name : str
        Human readable identifier.
    expected_value : float
        Expected reward or utility.
    cost : float
        Resource cost associated with the action.
    """

    __slots__ = ("name", "expected_value", "cost")

    def __init__(self, name: str, expected_value: float, cost: float):
        self.name = name
        self.expected_value = float(expected_value)
        self.cost = float(cost)


class Morphology:
    """
    Geometric description of a physical (or logical) entity.

    All dimensions must be strictly positive.
    """

    __slots__ = ("length", "width", "height", "mass")

    def __init__(self, length: float, width: float, height: float, mass: float):
        for name, val in (("length", length), ("width", width), ("height", height), ("mass", mass)):
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)
        self.mass = float(mass)


# ----------------------------------------------------------------------
# Hybrid algorithm components
# ----------------------------------------------------------------------
def regret_weighted_probabilities(actions: Sequence[Action],
                                  fisher_info: float) -> np.ndarray:
    """
    Compute regret‑weighted probabilities, modulated by a scalar Fisher
    information term.

    The Fisher term acts as a temperature‑like factor: higher information
    sharpens the distribution, lower information flattens it.

    Parameters
    ----------
    actions : Sequence[Action]
        Collection of candidate actions.
    fisher_info : float
        Non‑negative scalar derived from the Fisher information of the
        underlying observation model.

    Returns
    -------
    np.ndarray
        Probabilities that sum to one.
    """
    if fisher_info < 0:
        raise ValueError("fisher_info must be non‑negative")

    utilities = np.array([a.expected_value - a.cost for a in actions], dtype=float)

    # Regret is defined as loss relative to the best utility.
    regret = utilities - np.max(utilities)

    # Temperature scaling using Fisher information.
    # We map Fisher information to a temperature τ = 1 / (1 + FI)
    # to keep the exponent bounded.
    tau = 1.0 / (1.0 + fisher_info)
    logits = regret / tau

    return _stable_softmax(logits)


def ternary_lens(probabilities: np.ndarray) -> np.ndarray:
    """
    Map a probability vector onto a ternary “lens” {-1, 0, +1}.

    The mapping respects the distribution’s quantiles:
      * Bottom third → -1
      * Middle third →  0
      * Top third    → +1

    Parameters
    ----------
    probabilities : np.ndarray
        Probabilities that sum to one.

    Returns
    -------
    np.ndarray
        Integer array with values -1, 0, or +1.
    """
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be a 1‑D array")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("probabilities must sum to 1")

    # Compute quantile thresholds
    sorted_idx = np.argsort(probabilities)
    n = probabilities.size
    third = n // 3

    lens = np.zeros_like(probabilities, dtype=int)
    lens[sorted_idx[:third]] = -1
    lens[sorted_idx[-third:]] = +1
    # The middle third remains 0 (already set)

    return lens


def hybrid_geometry(morphology: Morphology, fisher_info: float) -> float:
    """
    Geometry score that blends simple Gaussian beams with Fisher‑information
    scaling.

    Each spatial dimension is evaluated against a Gaussian centred on the
    dimension itself; the width is taken from the next orthogonal dimension
    (length↔width↔height↔mass) to create a coupling that respects the original
    “simple geometry” intuition while being mathematically consistent.

    The final score is multiplied by a monotonic function of Fisher information
    (1 + FI) to deepen the integration between the statistical and geometric
    subsystems.

    Parameters
    ----------
    morphology : Morphology
        Geometric entity.
    fisher_info : float
        Fisher information scalar.

    Returns
    -------
    float
        Positive geometry score.
    """
    # Coupled Gaussian beams
    beam_l = gaussian_beam(morphology.length, morphology.length, morphology.width)
    beam_w = gaussian_beam(morphology.width, morphology.width, morphology.height)
    beam_h = gaussian_beam(morphology.height, morphology.height, morphology.mass)

    raw_score = beam_l + beam_w + beam_h

    # Fisher‑information scaling (ensures positivity and smooth growth)
    scale = 1.0 + fisher_info
    return raw_score * scale


def hybrid_resource_allocation(actions: Sequence[Action],
                               fisher_info: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute both regret‑weighted probabilities and their ternary lens.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (probabilities, ternary_lens)
    """
    probs = regret_weighted_probabilities(actions, fisher_info)
    lens = ternary_lens(probs)
    return probs, lens


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a sample morphology
    morph = Morphology(length=10.0, width=5.0, height=2.0, mass=1.0)

    # Fisher information derived from a Gaussian observation centred at 5 with σ=5
    fisher = fisher_information_gaussian(theta=5.0, center=5.0, width=5.0)

    # Geometry score
    geo_score = hybrid_geometry(morph, fisher)
    print(f"Geometry score: {geo_score:.6f}")

    # Sample actions
    actions = [
        Action(name="action1", expected_value=10.0, cost=1.0),
        Action(name="action2", expected_value=20.0, cost=2.0),
        Action(name="action3", expected_value=15.0, cost=1.5),
    ]

    # Resource allocation with ternary lens
    probs, lens = hybrid_resource_allocation(actions, fisher)
    print("Probabilities:", np.round(probs, 4))
    print("Ternary lens :", lens)