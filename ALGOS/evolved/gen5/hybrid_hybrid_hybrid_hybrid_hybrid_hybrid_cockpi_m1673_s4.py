# DARWIN HAMMER — match 1673, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m198_s1.py (gen4)
# parent_b: hybrid_hybrid_cockpit_metri_hybrid_hybrid_liquid_m488_s1.py (gen4)
# born: 2026-05-29T23:38:16Z

import math
import random
from pathlib import Path
from typing import Sequence, List, Tuple

import numpy as np

Vector = Sequence[float]


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Return numerator/denominator, falling back to *fallback* if denominator is zero."""
    return numerator / denominator if denominator != 0 else fallback


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0,1] for non‑negative vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


# ----------------------------------------------------------------------
# Parent A – Gaussian beam & Fisher information
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity centred at *center* with standard deviation *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single‑parameter Gaussian model.

    I(θ) = (∂ℓ/∂θ)² / ℓ, where ℓ = Gaussian intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def gaussian_filter(data: np.ndarray, sigma: float) -> np.ndarray:
    """1‑D Gaussian smoothing with standard deviation *sigma*."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    # Build a symmetric kernel covering ±3σ (enough for practical purposes)
    radius = int(math.ceil(3 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(data, kernel, mode="same")


# ----------------------------------------------------------------------
# Parent B – Claim‑quality metrics
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
    *,
    inertia: float = 0.5,
    cognitive: float = 1.5,
    social: float = 1.5,
    rng: random.Random | None = None,
) -> float:
    """
    Particle‑swarm‑style scalar attraction toward the global best.

    Returns a normalized attraction strength in [0,1] that can be used
    to modulate diffusion or cost terms.
    """
    if len(x) != len(g_best):
        raise ValueError("x and g_best must have the same dimension")

    if rng is None:
        rng = random.Random()

    # Convert to numpy for vector arithmetic
    x_arr = np.asarray(x, dtype=float)
    g_arr = np.asarray(g_best, dtype=float)

    # PSO velocity update (scalarized)
    r1, r2 = rng.random(), rng.random()
    cognitive_component = cognitive * r1 * (g_arr - x_arr)
    social_component = social * r2 * (g_arr - x_arr)  # using g_best for both terms for simplicity
    velocity = inertia * (cognitive_component + social_component)

    # Use the L2 norm of the velocity as a proxy for “interaction strength”
    strength = float(np.linalg.norm(velocity))
    # Normalise by a heuristic max (distance between extreme points in unit cube)
    max_possible = math.sqrt(len(x)) * (inertia * (cognitive + social))
    return max(0.0, min(1.0, strength / max_possible))


# ----------------------------------------------------------------------
# Hybrid integration utilities
# ----------------------------------------------------------------------
def pruning_probability(honesty: float, anti_slop: float) -> float:
    """Probability of pruning a candidate based on honesty and anti‑slop."""
    return max(0.0, min(1.0, honesty * anti_slop))


def modulate_diffusion(similarity: float, interaction_strength: float) -> float:
    """
    Diffusion intensity in [0,1] that blends similarity of token sets
    with the strength of the social interaction.
    """
    return max(0.0, min(1.0, similarity * interaction_strength))


def weighted_fisher(
    fisher: float,
    honesty: float,
    anti_slop: float,
    diffusion_factor: float,
) -> float:
    """
    Deeply integrate the three quality signals into a single Fisher‑like term.

    The base Fisher information is scaled by a credibility factor
    (average of honesty and anti‑slop) and further attenuated by the
    diffusion factor, which reflects how much the surrounding evidence
    supports the current claim.
    """
    credibility = (honesty + anti_slop) / 2.0
    return fisher * credibility * diffusion_factor


def composite_cost(
    weighted_fisher: float,
    reconstruction_risk: float,
    ram: float,
    pruning_prob: float,
) -> float:
    """
    Final cost for model selection.

    The cost grows with Fisher information (more informative models are
    preferred), reconstruction risk, and RAM usage, but it is discounted
    by the pruning probability – a high pruning probability indicates
    that the model should be penalised.
    """
    base = weighted_fisher * (reconstruction_risk * ram)
    # Inverse relationship: larger pruning probability → larger penalty
    penalty = 1.0 + pruning_prob
    return base * penalty


# ----------------------------------------------------------------------
# Main hybrid operation
# ----------------------------------------------------------------------
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
    *,
    rng_seed: int | str | None = None,
) -> float:
    """
    Execute the fused algorithm.

    Returns a scalar cost that respects VRAM constraints while
    incorporating statistical fidelity (Fisher), claim‑quality metrics,
    and a diffusion‑modulated interaction term.
    """
    # ---- Metric extraction -------------------------------------------------
    fisher = fisher_score(theta, center, width)
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    anti_slop = anti_slop_ratio(claims_with_evidence, total_claims_emitted)

    # Reconstruction risk is the complement of the anti‑slop ratio:
    reconstruction_risk = 1.0 - anti_slop

    # ---- Interaction & diffusion -------------------------------------------
    rng = random.Random(rng_seed)
    interaction_strength = social_interaction(x, g_best, rng=rng)

    # Similarity derived from cosine similarity of the two token vectors
    similarity = _cosine_similarity(np.asarray(x, dtype=float), np.asarray(g_best, dtype=float))

    diffusion_factor = modulate_diffusion(similarity, interaction_strength)

    # ---- Deep integration ---------------------------------------------------
    w_fisher = weighted_fisher(fisher, honesty, anti_slop, diffusion_factor)

    # Pruning probability influences the final penalty
    prune_prob = pruning_probability(honesty, anti_slop)

    # ---- Final cost ---------------------------------------------------------
    cost = composite_cost(w_fisher, reconstruction_risk, ram, prune_prob)
    return cost


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

    result = hybrid_operation(
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
        rng_seed=42,
    )
    print(f"Hybrid composite cost: {result:.6f}")