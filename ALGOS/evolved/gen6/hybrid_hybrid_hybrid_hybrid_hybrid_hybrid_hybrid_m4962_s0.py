# DARWIN HAMMER — match 4962, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_percyp_hybrid_hybrid_hybrid_m1389_s4.py (gen4)
# parent_b: hybrid_hybrid_hybrid_physar_hybrid_hybrid_ternar_m2736_s3.py (gen5)
# born: 2026-05-29T23:59:10Z

"""Hybrid Algorithm: Morphology‑Geography ↔ Bayesian Bandit Fusion

Parent A (hybrid_hybrid_hybrid_percyp_hybrid_hybrid_hybrid_m1389_s4.py) supplies
geometric utilities (sphericity, haversine distance) that quantify similarity
between physical objects (Morphology, Entity).  

Parent B (hybrid_hybrid_hybrid_physar_hybrid_hybrid_ternar_m2736_s3.py) supplies a
bandit‑style Bayesian update mechanism that maintains per‑edge priors
(_edge_priors) and computes flux through a Physarum‑inspired network.

**Mathematical Bridge**

For every ordered pair of entities (i, j) we define an *edge* with

* geometric conductance  g₍ᵢⱼ₎  =  f( sphericity_i , sphericity_j , mass_i , mass_j )
* geographic length      ℓ₍ᵢⱼ₎  =  haversine_distance( pos_i , pos_j )
* Bayesian prior          p₍ᵢⱼ₎  =  _edge_priors[(i,j)]

The *expected cost* of traversing the edge is

    C₍ᵢⱼ₎ = ℓ₍ᵢⱼ₎ / ( g₍ᵢⱼ₎ · p₍ᵢⱼ₎ + ε )

where ε prevents division by zero.  The cost feeds the bandit reward signal:
a lower cost yields a higher reward, which in turn updates p₍ᵢⱼ₎ via Bayes’
rule.  This creates a closed loop that couples the physical topology (A) with
the learning dynamics (B).

The module implements three core hybrid functions:
    • compute_conductance – maps morphologies to a conductance value.
    • expected_edge_cost   – combines conductance, distance and Bayesian prior.
    • hybrid_step          – a single iteration that evaluates fluxes,
      computes rewards, and updates the Bayesian policy.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Data structures (union of both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class Entity:
    id: str
    lat: float   # decimal degrees, positive north, negative west
    lon: float   # decimal degrees, positive east, negative west
    category: str
    morphology: Morphology
    score: float = 0.0
    address_signature: str = ""


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str          # node identifier (source entity)
    action_id: str           # target entity identifier
    reward: float
    confidence_bound: float = 1.0  # default confidence if not supplied


# ----------------------------------------------------------------------
# Global stores (mirroring parent B)
# ----------------------------------------------------------------------
_POLICY: dict[str, List[float]] = {}
_STORE: dict[str, float] = {}
_edge_priors: dict[Tuple[str, str], float] = {}

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Classical sphericity ψ ∈ (0,1] for a rectangular solid."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be strictly positive.")
    volume = length * width * height
    surface_area = 2 * (length * width + width * height + height * length)
    return (volume * (36 * math.pi) ** (1 / 3)) / surface_area


def haversine_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in kilometres between two (lat, lon) pairs."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    sin_dlat = math.sin(dlat / 2.0)
    sin_dlon = math.sin(dlon / 2.0)

    aa = sin_dlat ** 2 + math.cos(lat1) * math.cos(lat2) * sin_dlon ** 2
    c = 2.0 * math.atan2(math.sqrt(aa), math.sqrt(1.0 - aa))
    return 6371.0 * c  # Earth radius in kilometres


def gini_coefficient(values: List[float]) -> float:
    """Standard Gini coefficient for a non‑negative list."""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    if np.any(arr < 0):
        raise ValueError("Gini coefficient is undefined for negative values.")
    sorted_arr = np.sort(arr)
    n = len(arr)
    cumulative = np.cumsum(sorted_arr)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return gini


# ----------------------------------------------------------------------
# Bayesian helpers (derived from Parent B)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability of evidence."""
    return prior * likelihood + (1 - prior) * false_positive


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior probability via Bayes' rule."""
    if marginal == 0:
        return prior  # avoid division by zero; keep prior unchanged
    return (prior * likelihood) / marginal


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_conductance(morph_a: Morphology, morph_b: Morphology) -> float:
    """
    Conductance g₍ᵢⱼ₎ is proportional to the geometric mean of the two
    sphericities and to the square‑root of the product of masses.
    This mirrors the Physarum principle where thicker tubes (higher mass)
    and more spherical bodies (lower surface‑to‑volume ratio) conduct better.
    """
    psi_a = sphericity_index(morph_a.length, morph_a.width, morph_a.height)
    psi_b = sphericity_index(morph_b.length, morph_b.width, morph_b.height)
    mass_factor = math.sqrt(morph_a.mass * morph_b.mass)
    return math.sqrt(psi_a * psi_b) * mass_factor + 1e-6  # epsilon to stay >0


def expected_edge_cost(
    src: Entity,
    dst: Entity,
    conductance: float,
    eps: float = 1e-6,
) -> float:
    """
    Expected cost C₍ᵢⱼ₎ = ℓ / (g·p + ε) where:
        ℓ = geographic distance,
        g = conductance,
        p = Bayesian prior for edge (src.id, dst.id).
    """
    distance = haversine_distance((src.lat, src.lon), (dst.lat, dst.lon))
    prior = _edge_priors.get((src.id, dst.id), 0.5)  # default uninformed prior
    return distance / (conductance * prior + eps)


def flux(
    conductance: float,
    edge_length: float,
    pressure_a: float,
    pressure_b: float,
    eps: float = 1e-6,
) -> float:
    """
    Physarum‑style flux equation:
        Φ = g * (P_a - P_b) / (ℓ + ε)
    """
    return conductance * (pressure_a - pressure_b) / (edge_length + eps)


def reset_policy() -> None:
    """Clears all stored statistics and priors."""
    _POLICY.clear()
    _STORE.clear()
    _edge_priors.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    """
    Incorporates observed rewards into the bandit statistics and updates the
    Bayesian edge priors using the bridge formula described in the module
    docstring.
    """
    # ---- Bandit statistics (running average of rewards) ----
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])  # [total_reward, count]
        stats[0] += float(u.reward)
        stats[1] += 1.0

    # ---- Bayesian edge‑prior update ----
    for u in updates:
        edge = (u.context_id, u.action_id)
        prior = _edge_priors.get(edge, 0.5)

        # Likelihood is interpreted as normalized reward.
        # To keep it in [0,1] we clip the reward.
        reward_norm = max(0.0, min(1.0, float(u.reward)))
        likelihood = reward_norm

        false_positive = 0.1  # hyper‑parameter: assumed noise level
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal)
        _edge_priors[edge] = posterior


def hybrid_step(
    entities: List[Entity],
    pressures: Dict[str, float],
    updates: List[BanditUpdate],
    eps: float = 1e-6,
) -> Dict[Tuple[str, str], float]:
    """
    Executes one hybrid iteration:
        1. Compute conductances and distances for every ordered pair.
        2. Derive fluxes using current pressures.
        3. Translate flux efficiency into rewards (higher flux ⇒ higher reward).
        4. Update the Bayesian policy with those rewards.
        5. Return a mapping edge→flux for inspection.
    """
    # 1. Conductance & distance matrix
    conductance_map: Dict[Tuple[str, str], float] = {}
    distance_map: Dict[Tuple[str, str], float] = {}

    for src in entities:
        for dst in entities:
            if src.id == dst.id:
                continue
            key = (src.id, dst.id)
            g = compute_conductance(src.morphology, dst.morphology)
            d = haversine_distance((src.lat, src.lon), (dst.lat, dst.lon))
            conductance_map[key] = g
            distance_map[key] = d

    # 2. Flux computation
    flux_map: Dict[Tuple[str, str], float] = {}
    for (src_id, dst_id), g in conductance_map.items():
        p_a = pressures.get(src_id, 0.0)
        p_b = pressures.get(dst_id, 0.0)
        ℓ = distance_map[(src_id, dst_id)]
        Φ = flux(g, ℓ, p_a, p_b, eps)
        flux_map[(src_id, dst_id)] = Φ

    # 3. Reward generation (inverse of absolute cost)
    reward_updates: List[BanditUpdate] = []
    for (src_id, dst_id), Φ in flux_map.items():
        # Cost derived from the bridge equation
        cost = expected_edge_cost(
            src=next(e for e in entities if e.id == src_id),
            dst=next(e for e in entities if e.id == dst_id),
            conductance=conductance_map[(src_id, dst_id)],
            eps=eps,
        )
        # Reward is high when cost is low; we normalise to [0,1].
        reward = 1.0 / (1.0 + cost)  # sigmoid‑like scaling
        reward_updates.append(
            BanditUpdate(
                context_id=src_id,
                action_id=dst_id,
                reward=reward,
                confidence_bound=abs(Φ) + eps,
            )
        )

    # 4. Policy update
    update_policy(reward_updates)

    # 5. Return fluxes for external analysis
    return flux_map


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny network of three entities with simple morphologies.
    entities = [
        Entity(
            id="A",
            lat=34.0522,
            lon=-118.2437,
            category="city",
            morphology=Morphology(length=10.0, width=8.0, height=5.0, mass=1200.0),
        ),
        Entity(
            id="B",
            lat=36.1699,
            lon=-115.1398,
            category="city",
            morphology=Morphology(length=9.0, width=7.0, height=6.0, mass=1100.0),
        ),
        Entity(
            id="C",
            lat=40.7128,
            lon=-74.0060,
            category="city",
            morphology=Morphology(length=12.0, width=9.0, height=4.0, mass=1300.0),
        ),
    ]

    # Initialise arbitrary pressures (could be any scalar field).
    pressures = {"A": 1.0, "B": 0.5, "C": 0.0}

    # No external updates on first iteration.
    dummy_updates: List[BanditUpdate] = []

    # Run a single hybrid step.
    fluxes = hybrid_step(entities, pressures, dummy_updates)

    # Print a concise summary.
    print("Fluxes (source → target):")
    for (src, dst), Φ in fluxes.items():
        print(f"  {src} → {dst}: {Φ:.6f}")

    # Show updated priors.
    print("\nUpdated edge priors:")
    for edge, prior in _edge_priors.items():
        print(f"  {edge}: {prior:.4f}")

    # Verify that the policy statistics are populated.
    print("\nBandit policy statistics (action_id → [total_reward, count]):")
    for aid, stats in _POLICY.items():
        print(f"  {aid}: {stats}")