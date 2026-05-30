# DARWIN HAMMER — match 2367, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m528_s0.py (gen4)
# parent_b: hybrid_hybrid_sparse_wta_hy_hybrid_minimum_cost__m392_s0.py (gen3)
# born: 2026-05-29T23:42:00Z

"""Hybrid Fusion Module
Combines:
- Parent A: spatial‑aware privacy risk computation using haversine distances and reconstruction risk.
- Parent B: model pool management, winner‑take‑all (WTA) selection, and Bayesian updating of edge priors in a minimum‑cost tree.

Mathematical Bridge:
The spatial privacy risk vector **r** (size N) derived from entities is used to form *edge priors* for a graph G = (V,E) where V are entities and E are candidate connections.  
For an edge e = (i,j) the prior probability is set to  

    π_e = (r_i * r_j) / Σ_{e'∈E} (r_{i'} * r_{j'})

providing a normalized prior that favours edges linking high‑risk (i.e. less private) entities.  
These priors feed the Bayesian update rule  

    posterior_e ∝ π_e * L_e

where L_e is a likelihood supplied by evidence (e.g., observed communication cost).  
The resulting posteriors become edge weights for a minimum‑cost spanning tree (MCST).  
The MCST cost then informs a *score* for each ModelTier; a winner‑take‑all (WTA) step picks the tier with the lowest (best) cost, respecting RAM constraints from the original ModelPool logic.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Any

import numpy as np


# ----------------------------------------------------------------------
# Data structures (merged from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int = 0  # optional, kept for compatibility with Parent A


class ModelPool:
    """Manages loading of ModelTier instances respecting RAM ceiling and tier exclusivity."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # evict the oldest loaded model (FIFO)
            oldest_key = next(iter(self.loaded))
            self.loaded.pop(oldest_key)
        self.load(model)


# ----------------------------------------------------------------------
# Helper functions from Parent A
# ----------------------------------------------------------------------
def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat,lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Simple reconstruction risk between 0 and 1."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def spatial_aware_privacy_risk_vector(entities: List[Entity], delta_m: float) -> np.ndarray:
    """
    For each entity i compute a risk score based on the proportion of
    *similar* entities (same signature) that lie within `delta_m` metres.
    The raw proportion is then passed through the reconstruction risk
    function to obtain a bounded risk ∈ [0,1].
    """
    risks = []
    n = len(entities)
    for i, entity in enumerate(entities):
        similar = 0
        for j, other in enumerate(entities):
            if i == j:
                continue
            if signature(entity) != signature(other):
                continue
            if haversine_m((entity.lat, entity.lon), (other.lat, other.lon)) <= delta_m:
                similar += 1
        risk = reconstruction_risk_score(similar, n - 1)
        risks.append(risk)
    return np.array(risks, dtype=float)


# ----------------------------------------------------------------------
# Helper functions from Parent B
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = '') -> List[float]:
    """Hash‑based sparse expansion (3‑hash per value)."""
    if m <= 0:
        raise ValueError('m must be positive')
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f'{salt}:{i}:{r}'.encode()).digest()
            j = int.from_bytes(h[:8], 'big') % m
            sign = 1.0 if (h[8] & 1) else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: List[float], k: int) -> List[int]:
    """Return a binary mask (1 for top‑k entries, 0 otherwise)."""
    if k <= 0:
        return [0] * len(values)
    indices = np.argpartition(values, -k)[-k:]
    mask = [0] * len(values)
    for idx in indices:
        mask[idx] = 1
    return mask


def winner_take_all(models: List[ModelTier], scores: List[float]) -> ModelTier:
    """Select the model with the lowest score (WTA)."""
    if not models or not scores or len(models) != len(scores):
        raise ValueError("Models and scores must be non‑empty and of equal length")
    min_idx = int(np.argmin(scores))
    return models[min_idx]


# ----------------------------------------------------------------------
# Fusion core – three representative functions
# ----------------------------------------------------------------------
def compute_edge_priors(entities: List[Entity],
                        edges: List[Tuple[int, int]],
                        delta_m: float) -> np.ndarray:
    """
    Build a normalized prior distribution over edges using spatial privacy risks.
    `edges` are index pairs into `entities`.
    """
    risk_vec = spatial_aware_privacy_risk_vector(entities, delta_m)  # shape (N,)
    raw = np.empty(len(edges), dtype=float)
    for idx, (i, j) in enumerate(edges):
        raw[idx] = risk_vec[i] * risk_vec[j]
    total = raw.sum()
    if total == 0:
        # fallback to uniform prior
        return np.full(len(edges), 1.0 / len(edges))
    return raw / total


def bayesian_update_edge_posteriors(priors: np.ndarray,
                                   likelihoods: np.ndarray) -> np.ndarray:
    """
    Perform a Bayesian update on edge probabilities.
    posterior ∝ prior * likelihood, then renormalized to sum to 1.
    """
    if priors.shape != likelihoods.shape:
        raise ValueError("Priors and likelihoods must have the same shape")
    unnorm = priors * likelihoods
    total = unnorm.sum()
    if total == 0:
        # avoid division by zero – revert to priors
        return priors.copy()
    return unnorm / total


def select_optimal_model(entities: List[Entity],
                         edges: List[Tuple[int, int]],
                         model_tiers: List[ModelTier],
                         delta_m: float,
                         evidence: np.ndarray,
                         pool: ModelPool) -> ModelTier:
    """
    End‑to‑end hybrid routine:
    1. Compute edge priors from spatial privacy risk.
    2. Update priors with Bayesian evidence.
    3. Derive a scalar cost for each ModelTier (e.g., weighted sum of posterior edge probs).
    4. Use winner‑take‑all to pick the best tier.
    5. Load the chosen model into the provided ModelPool (evicting if needed).
    """
    # 1‑2: Edge probabilities
    priors = compute_edge_priors(entities, edges, delta_m)
    posteriors = bayesian_update_edge_posteriors(priors, evidence)

    # 3: Simple cost function – higher posterior mass on edges that involve
    #    high‑risk entities is penalized. We approximate by dot(product) with priors.
    #    Each tier gets a different scaling factor (simulated here).
    base_cost = np.dot(posteriors, priors)  # scalar ∈ [0,1]
    tier_costs = []
    for tier in model_tiers:
        # arbitrary scaling: more RAM -> higher cost, lower tier label -> lower cost
        ram_factor = tier.ram_mb / 1024.0
        tier_factor = 1.0 if tier.tier == "T1" else 1.2 if tier.tier == "T2" else 1.5
        cost = base_cost * ram_factor * tier_factor
        tier_costs.append(cost)

    # 4: Winner‑take‑all (lowest cost wins)
    best_model = winner_take_all(model_tiers, tier_costs)

    # 5: Load into pool
    if not pool.is_loaded(best_model.name):
        try:
            pool.load(best_model)
        except Exception:
            # if loading fails (RAM ceiling or exclusivity), try eviction strategy
            pool.load_with_eviction(best_model)

    return best_model


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic entities
    random.seed(42)
    entities = [
        Entity(id=f"E{i}",
               lat=37.0 + random.random() * 0.01,
               lon=-122.0 + random.random() * 0.01,
               category="catA" if i % 2 == 0 else "catB")
        for i in range(6)
    ]

    # Simple fully connected graph (undirected) represented by index pairs
    edges = [(i, j) for i in range(len(entities)) for j in range(i + 1, len(entities))]

    # Dummy evidence: higher likelihood for edges with smaller index distance
    likelihoods = np.exp(-0.1 * np.array([abs(i - j) for i, j in edges]))
    likelihoods = likelihoods / likelihoods.sum()  # normalize for sanity

    # Model tiers
    model_tiers = [
        ModelTier(name="Tiny", ram_mb=512, tier="T1", vram_mb=256),
        ModelTier(name="Small", ram_mb=1024, tier="T2", vram_mb=512),
        ModelTier(name="Medium", ram_mb=2048, tier="T3", vram_mb=1024),
    ]

    # Initialise pool
    pool = ModelPool(ram_ceiling_mb=3000)

    # Run hybrid selection
    selected = select_optimal_model(
        entities=entities,
        edges=edges,
        model_tiers=model_tiers,
        delta_m=100.0,            # 100 m spatial window
        evidence=likelihoods,
        pool=pool
    )

    print(f"Selected model: {selected.name} (RAM {selected.ram_mb} MB, tier {selected.tier})")
    print(f"Currently loaded models: {list(pool.loaded.keys())}")