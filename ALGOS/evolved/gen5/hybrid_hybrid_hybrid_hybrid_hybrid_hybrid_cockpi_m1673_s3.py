# DARWIN HAMMER — match 1673, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m198_s1.py (gen4)
# parent_b: hybrid_hybrid_cockpit_metri_hybrid_hybrid_liquid_m488_s1.py (gen4)
# born: 2026-05-29T23:38:16Z

import math
import random
import sys
from pathlib import Path
from typing import Sequence, List

import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Parent A components – Gaussian beam & Fisher information
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity centred at *center* with standard deviation *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single-parameter Gaussian model.
    I(θ) = (∂ℓ/∂θ)² / ℓ, where ℓ = Gaussian intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def gaussian_filter(data: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a 1-D Gaussian smoothing kernel (sigma) to *data*."""
    kernel = np.array([gaussian_beam(x, 0.0, sigma) for x in data])
    kernel /= kernel.sum()
    return np.convolve(data, kernel, mode="same")


# ----------------------------------------------------------------------
# Parent B core components
# ----------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Fraction of claims that are backed by evidence, clipped to [0,1]."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Honesty = proportion of displayed claims that are known to be OK."""
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total))


def social_interaction(
    x: Vector,
    g_best: Vector,
    k: int = 1,
    r1: float | None = None,
    seed: int | str | None = None,
) -> float:
    """Particle-swarm-style attraction toward the global best."""
    if len(x) != len(g_best):
        raise ValueError("x and g_best must share dimension")
    if k not in (1, 2):
        raise ValueError("k is normally 1 or 2")
    rng = random.Random(seed)
    r = rng.random() if r1 is None else r1
    if not (0.0 <= r <= 1.0):
        raise ValueError("r must be in [0,1]")
    return r


# ----------------------------------------------------------------------
# Hybrid components
# ----------------------------------------------------------------------
def pruning_probability(honesty: float, anti_slop_ratio: float) -> float:
    """Construct pruning probability from honesty and slop ratio metrics."""
    return max(0.0, min(1.0, honesty * anti_slop_ratio))


def composite_cost(fisher_score: float, reconstruction_risk: float, ram: float) -> float:
    """Composite cost function for model selection under VRAM budget."""
    if ram <= 0.0:
        raise ValueError("RAM must be positive")
    return fisher_score * (reconstruction_risk / ram)


def modulate_diffusion(similarity: float, social_interaction: float) -> float:
    """Modulate diffusion intensity based on similarity and social interaction."""
    return max(0.0, min(1.0, similarity * social_interaction))


def hybrid_operation(
    theta: float,
    center: float,
    width: float,
    claims_with_evidence: int,
    total_claims_emitted: int,
    displayed_ok: int,
    unknown_displayed_as_ok: int,
    x: Vector,
    g_best: Vector,
    ram: float,
) -> float:
    """Run the hybrid operation."""
    fisher = fisher_score(theta, center, width)
    reconstruction_risk = 1.0 - (claims_with_evidence / total_claims_emitted)
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    anti_slop = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    pruning_prob = pruning_probability(honesty, anti_slop)
    similarity = modulate_diffusion(honesty, social_interaction(x, g_best))
    composite = composite_cost(fisher, reconstruction_risk, ram)
    return composite * pruning_prob * similarity


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    theta = 0.5
    center = 1.0
    width = 0.5
    claims_with_evidence = 10
    total_claims_emitted = 20
    displayed_ok = 5
    unknown_displayed_as_ok = 3
    x = [1.0, 2.0, 3.0]
    g_best = [4.0, 5.0, 6.0]
    ram = 0.5

    hybrid = hybrid_operation(
        theta,
        center,
        width,
        claims_with_evidence,
        total_claims_emitted,
        displayed_ok,
        unknown_displayed_as_ok,
        x,
        g_best,
        ram,
    )
    print(f"Hybrid composite cost: {hybrid}")