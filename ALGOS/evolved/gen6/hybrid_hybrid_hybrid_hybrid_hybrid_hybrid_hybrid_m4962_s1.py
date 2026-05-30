# DARWIN HAMMER — match 4962, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_percyp_hybrid_hybrid_hybrid_m1389_s4.py (gen4)
# parent_b: hybrid_hybrid_hybrid_physar_hybrid_hybrid_ternar_m2736_s3.py (gen5)
# born: 2026-05-29T23:59:10Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
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
    context_id: str
    action_id: str
    reward: float
    confidence_bound: float = 1.0


_POLICY: dict[str, List[float]] = {}
_STORE: dict[str, float] = {}
_edge_priors: dict[Tuple[str, str], float] = {}


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be strictly positive.")
    volume = length * width * height
    surface_area = 2 * (length * width + width * height + height * length)
    return (volume * (36 * math.pi) ** (1 / 3)) / surface_area


def haversine_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    sin_dlat = math.sin(dlat / 2.0)
    sin_dlon = math.sin(dlon / 2.0)

    aa = sin_dlat ** 2 + math.cos(lat1) * math.cos(lat2) * sin_dlon ** 2
    c = 2.0 * math.atan2(math.sqrt(aa), math.sqrt(1.0 - aa))
    return 6371.0 * c


def gini_coefficient(values: List[float]) -> float:
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


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    return prior * likelihood + (1 - prior) * false_positive


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal == 0:
        return prior
    return (prior * likelihood) / marginal


def compute_conductance(morph_a: Morphology, morph_b: Morphology) -> float:
    psi_a = sphericity_index(morph_a.length, morph_a.width, morph_a.height)
    psi_b = sphericity_index(morph_b.length, morph_b.width, morph_b.height)
    mass_factor = math.sqrt(morph_a.mass * morph_b.mass)
    return math.sqrt(psi_a * psi_b) * mass_factor + 1e-6


def expected_edge_cost(
    src: Entity,
    dst: Entity,
    conductance: float,
    eps: float = 1e-6,
) -> float:
    distance = haversine_distance((src.lat, src.lon), (dst.lat, dst.lon))
    prior = _edge_priors.get((src.id, dst.id), 0.5)
    return distance / (conductance * prior + eps)


def flux(
    conductance: float,
    edge_length: float,
    pressure_a: float,
    pressure_b: float,
    eps: float = 1e-6,
) -> float:
    return conductance * (pressure_a - pressure_b) / (edge_length + eps)


def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()
    _edge_priors.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

    for u in updates:
        edge = (u.context_id, u.action_id)
        prior = _edge_priors.get(edge, 0.5)
        reward_norm = max(0.0, min(1.0, float(u.reward)))
        likelihood = reward_norm
        false_positive = 0.1
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal)
        _edge_priors[edge] = posterior


def hybrid_step(src: Entity, dst: Entity, updates: List[BanditUpdate]) -> None:
    conductance = compute_conductance(src.morphology, dst.morphology)
    cost = expected_edge_cost(src, dst, conductance)
    updates.append(BanditUpdate(src.id, dst.id, cost))
    update_policy(updates)


def main():
    # Example usage:
    entity1 = Entity("1", 37.7749, -122.4194, "category1", Morphology(1.0, 1.0, 1.0, 1.0))
    entity2 = Entity("2", 37.7859, -122.4364, "category2", Morphology(1.0, 1.0, 1.0, 1.0))
    updates = []
    hybrid_step(entity1, entity2, updates)


if __name__ == "__main__":
    main()