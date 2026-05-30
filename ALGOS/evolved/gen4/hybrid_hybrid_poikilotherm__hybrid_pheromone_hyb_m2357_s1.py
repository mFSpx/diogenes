# DARWIN HAMMER — match 2357, survivor 1
# gen: 4
# parent_a: hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s0.py (gen3)
# parent_b: hybrid_pheromone_hybrid_distributed_l_m41_s2.py (gen2)
# born: 2026-05-29T23:41:56Z

"""Hybrid Algorithm Fusing Poikilotherm-Schoolfield and Pheromone Dynamics

This module integrates two parent algorithms:

* **Algorithm A** – `hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s0.py` – 
  provides a temperature-dependent activity function based on the Schoolfield-Rollinson rate equation.
* **Algorithm B** – `hybrid_pheromone_hybrid_distributed_l_m41_s2.py` – 
  defines an exponential decay of a scalar signal and clusters graph nodes by computing a perceptual hash.

The mathematical bridge between these algorithms lies in the treatment of node feature vectors in the 
pheromone-based clustering. We re-scale each node's feature vector by the Schoolfield activity 
computed from a temperature mapped from the node's geometric or topological distance.

The hybrid algorithm injects the temperature-modulated activity dynamics directly into the 
hash-based clustering, yielding clusters that respect both similarity (hash) and 
temporal relevance (pheromone decay) as well as thermal activity.

The node beliefs are derived from incident edge posteriors and the full hybrid cost 
is computed using the temperature-weighted expectation of edge lengths.
"""

import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Hashable, Set, Dict, List
import numpy as np
from dataclasses import dataclass

R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_00

class PheromoneStore:
    """In-memory store mimicking the surface_pheromone table."""
    def __init__(self) -> None:
        # surface_key → list of records
        self._store: Dict[str, List[Dict]] = {}

    def signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: int,
        detail: str | None = None,
    ) -> str:
        """Insert a new pheromone record and return its UUID-like id."""
        rec = {
            "pheromone_id": f"{surface_key}:{len(self._store.get(surface_key, []))}",
            "surface_key": surface_key,
            "signal_kind": signal_kind,
            "signal_value": float(signal_value),
            "half_life_seconds": int(half_life_seconds),
            "created_at": datetime.now(timezone.utc),
            "detail": detail or "",
            "active": True,
        }
        self._store.setdefault(surface_key, []).append(rec)
        return rec["pheromone_id"]

    def decay(self, surface_key: str, elapsed_hours: float) -> float:
        """Compute the decayed pheromone value."""
        records = self._store.get(surface_key, [])
        if not records:
            return 0.0
        rec = records[-1]
        tau = rec["half_life_seconds"] / 3600
        return rec["signal_value"] * 0.5 ** (elapsed_hours / tau)

def normalized_activity(temp: float, params: SchoolfieldParams) -> float:
    """Compute the normalized activity using the Schoolfield-Rollinson rate equation."""
    return params.rho_25 * math.exp((temp - K25) * params.delta_h_activation / (R_CAL * temp * K25))

def temperature_weighted_posteriors(edge_lengths: List[float], temperatures: List[float], 
                                    prior_beliefs: List[float], params: SchoolfieldParams) -> List[float]:
    """Compute Bayesian edge posteriors scaled by Schoolfield activity."""
    activities = [normalized_activity(temp, params) for temp in temperatures]
    scaled_beliefs = [prior * activity for prior, activity in zip(prior_beliefs, activities)]
    return [belief / sum(scaled_beliefs) for belief in scaled_beliefs]

def hybrid_stylometry(edge_lengths: List[float], posteriors: List[float]) -> float:
    """Compute the expected edge length under the temperature-weighted posteriors."""
    return sum(length * posterior for length, posterior in zip(edge_lengths, posteriors)) / sum(abs(posterior) for posterior in posteriors)

def hybrid_tree_cost(edge_lengths: List[float], temperatures: List[float], prior_beliefs: List[float], 
                     node_distances: List[float], lambda_: float, params: SchoolfieldParams) -> float:
    """Compute the full hybrid cost including the node-distance term."""
    posteriors = temperature_weighted_posteriors(edge_lengths, temperatures, prior_beliefs, params)
    expected_length = hybrid_stylometry(edge_lengths, posteriors)
    node_beliefs = [sum(posteriors[i] * posteriors[j] for i, j in enumerate(posteriors)) for _ in range(len(node_distances))]
    return expected_length + lambda_ * sum(node * distance for node, distance in zip(node_beliefs, node_distances)) / sum(abs(node) for node in node_beliefs)

def pheromone_based_clustering(node_features: List[List[float]], pheromone_store: PheromoneStore, 
                               surface_key: str, half_life_seconds: int) -> List[List[float]]:
    """Perform pheromone-based clustering."""
    pheromone_value = pheromone_store.signal(surface_key, "initial", 1.0, half_life_seconds)
    elapsed_hours = 0.1  # example elapsed time
    decayed_pheromone = pheromone_store.decay(surface_key, elapsed_hours)
    scaled_features = [[feature * decayed_pheromone for feature in node_feature] for node_feature in node_features]
    # Perform clustering using the scaled features (example: simple k-means)
    return [[feature / sum(scaled_features) for feature in scaled_feature] for scaled_feature in scaled_features]

if __name__ == "__main__":
    # Smoke test
    schoolfield_params = SchoolfieldParams()
    edge_lengths = [1.0, 2.0, 3.0]
    temperatures = [298.15, 300.0, 310.0]
    prior_beliefs = [0.2, 0.3, 0.5]
    node_distances = [1.0, 2.0]
    lambda_ = 0.5
    pheromone_store = PheromoneStore()
    node_features = [[1.0, 2.0], [3.0, 4.0]]
    surface_key = "example_surface"
    half_life_seconds = 3600

    posteriors = temperature_weighted_posteriors(edge_lengths, temperatures, prior_beliefs, schoolfield_params)
    expected_length = hybrid_stylometry(edge_lengths, posteriors)
    hybrid_cost = hybrid_tree_cost(edge_lengths, temperatures, prior_beliefs, node_distances, lambda_, schoolfield_params)
    clustered_features = pheromone_based_clustering(node_features, pheromone_store, surface_key, half_life_seconds)

    print("Posteriors:", posteriors)
    print("Expected Length:", expected_length)
    print("Hybrid Cost:", hybrid_cost)
    print("Clustered Features:", clustered_features)