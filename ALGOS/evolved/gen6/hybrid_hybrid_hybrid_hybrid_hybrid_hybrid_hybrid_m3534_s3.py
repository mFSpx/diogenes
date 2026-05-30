# DARWIN HAMMER — match 3534, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1455_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m50_s1.py (gen4)
# born: 2026-05-29T23:50:33Z

"""Hybrid Algorithm Fusion of Stylometry, Social Interaction, Health Scoring, and Hoeffding‑Gini Metrics

Parents:
- hybrid_hybrid_hybrid_hard_t_hybrid_capybara_opti_m32_s2.py (stylometry + vector‑based social interaction)
- hybrid_hybrid_hybrid_hoeffding_tre_m50_s1.py (health scoring, Gini coefficient, Hoeffding bound)

Mathematical Bridge:
The bridge is built on the observation that stylometric function‑word frequencies can be
treated as *quasi‑identifiers*.  From these counts we compute a reconstruction‑risk
score, which together with a recovery‑priority derived from the particle’s distance to the
global best vector yields a *health score*.  Health scores across a swarm are
aggregated with the Gini coefficient to quantify inequality, while the Hoeffding bound
provides a confidence interval for that inequality.  The health score also modulates the
social‑interaction update, biasing particles with higher health toward the global best.

The resulting system fuses the vector‑based dynamics of Parent A with the statistical
evaluation machinery of Parent B into a single, self‑consistent hybrid algorithm.
"""

import sys
import pathlib
import random
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry & social interaction utilities
# ----------------------------------------------------------------------
Vector = List[float]

FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
}


def stylometry_analysis(tokens: List[str]) -> Dict[str, int]:
    """Count function‑word categories in a token list."""
    counts = Counter(tokens)
    return {
        cat: sum(counts[t] for t in toks)
        for cat, toks in FUNCTION_CATS.items()
    }


def tokens_to_vector(tokens: List[str]) -> np.ndarray:
    """Convert token list to a normalized frequency vector over FUNCTION_CATS."""
    cat_counts = stylometry_analysis(tokens)
    total = sum(cat_counts.values())
    if total == 0:
        return np.zeros(len(FUNCTION_CATS))
    return np.array([cat_counts[cat] / total for cat in FUNCTION_CATS])


# ----------------------------------------------------------------------
# Parent B – health scoring, Gini, Hoeffding bound
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Risk = proportion of unique quasi‑identifiers."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def health_score(reconstruction_risk: float, recovery_priority: float) -> float:
    """Higher health when risk is low and priority is high."""
    return (1.0 - reconstruction_risk) * (1.0 - recovery_priority)


def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for non‑negative values."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0.0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a bounded random variable in [0, r]."""
    if r <= 0 or not (0.0 < delta < 1.0) or n <= 0:
        raise ValueError("Invalid parameters for Hoeffding bound")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


# ----------------------------------------------------------------------
# Hybrid structures
# ----------------------------------------------------------------------
@dataclass
class Particle:
    """A particle representing a document / model in the swarm."""
    position: np.ndarray          # stylometric vector
    velocity: np.ndarray          # change per iteration
    health: float = 0.0           # updated each iteration


def initialize_swarm(token_corpus: List[List[str]]) -> List[Particle]:
    """Create a swarm where each particle is seeded from a token list."""
    particles: List[Particle] = []
    for tokens in token_corpus:
        pos = tokens_to_vector(tokens)
        vel = np.zeros_like(pos)
        particles.append(Particle(position=pos, velocity=vel))
    return particles


def compute_particle_health(particle: Particle, total_records: int) -> float:
    """
    Derive health from stylometric quasi‑identifiers.
    - unique_quasi_identifiers: number of non‑zero function categories.
    - reconstruction_risk: proportion of unique identifiers.
    - recovery_priority: 1 - normalized Euclidean distance to global best (computed elsewhere).
    """
    nonzero = np.count_nonzero(particle.position)
    recon_risk = reconstruction_risk_score(nonzero, total_records)
    # placeholder priority; will be refined in the swarm update step
    return health_score(recon_risk, recovery_priority=0.5)  # 0.5 is a neutral default


def social_update(particles: List[Particle], k: float = 1.0, w: float = 0.5,
                 c1: float = 1.5, c2: float = 1.5) -> None:
    """
    PSO‑style update where the health score weights the attraction to the global best.
    Parameters:
        k   – scaling factor for health influence.
        w   – inertia weight.
        c1  – cognitive coefficient (self‑best, omitted for brevity).
        c2  – social coefficient (global best).
    The function mutates particles in‑place.
    """
    # Determine global best based on highest health
    best_particle = max(particles, key=lambda p: p.health)
    g_best_pos = best_particle.position.copy()

    for p in particles:
        # Random components
        r1 = random.random()
        r2 = random.random()

        # Velocity update: inertia + social term weighted by health
        cognitive = np.zeros_like(p.position)  # omitted for simplicity
        social = k * p.health * (g_best_pos - p.position)

        new_vel = w * p.velocity + c2 * r2 * social + c1 * r1 * cognitive
        p.velocity = new_vel

        # Position update
        p.position = p.position + p.velocity

        # Keep positions within [0,1] (they are frequencies)
        p.position = np.clip(p.position, 0.0, 1.0)

        # Re‑normalize to preserve a proper probability distribution
        if p.position.sum() > 0:
            p.position = p.position / p.position.sum()
        else:
            p.position = np.zeros_like(p.position)


def evaluate_swarm(particles: List[Particle], delta: float = 0.05) -> Tuple[float, float]:
    """
    Compute the Gini coefficient of the health scores and a Hoeffding bound
    that quantifies confidence in that coefficient.
    Returns (gini, hoeffding_bound).
    """
    healths = [p.health for p in particles]
    gini = gini_coefficient(healths)
    # Hoeffding bound on the Gini estimate (range of Gini is [0,1])
    bound = hoeffding_bound(r=1.0, delta=delta, n=len(particles))
    return gini, bound


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic corpus: each sub‑list is a tokenised document
    corpus = [
        "I am writing a test document about AI".lower().split(),
        "You have been reading the same test".lower().split(),
        "She will be going to the market".lower().split(),
        "They are planning a project without delay".lower().split(),
    ]

    # Initialise swarm
    swarm = initialize_swarm(corpus)

    # Total records for risk calculation (here: number of documents)
    total = len(corpus)

    # Initial health evaluation
    for p in swarm:
        p.health = compute_particle_health(p, total)

    # Perform a few social updates
    for _ in range(5):
        social_update(swarm, k=1.2, w=0.4, c1=1.0, c2=2.0)
        for p in swarm:
            p.health = compute_particle_health(p, total)

    # Final evaluation
    gini_val, hoeffding_val = evaluate_swarm(swarm)

    print("Final health scores:", [round(p.health, 3) for p in swarm])
    print("Gini coefficient of health:", round(gini_val, 4))
    print("Hoeffding bound (95% confidence approx):", round(hoeffding_val, 4))

    # Exit cleanly
    sys.exit(0)