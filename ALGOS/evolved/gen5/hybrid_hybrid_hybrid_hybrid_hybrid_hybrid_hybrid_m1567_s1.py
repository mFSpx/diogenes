# DARWIN HAMMER — match 1567, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sheaf__m529_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m528_s4.py (gen4)
# born: 2026-05-29T23:37:29Z

"""Hybrid Sheaf‑Bayesian Scheduler
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sheaf__m529_s3.py (Sheaf‑cohomology with entropy)
- hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m528_s4.py (Bayesian spatial‑privacy VRAM scheduler)

Mathematical bridge:
The Shannon entropy measured on each Sheaf node (derived from the uncertainty of the
Hoeffding‑tree decision boundaries in Parent A) is used to modulate the spatial‑privacy
prior `p_i` of each entity in the Bayesian posterior of Parent B.  
Specifically, the prior is scaled by `exp(-H_i)` where `H_i` is the node entropy,
thereby giving lower weight to highly uncertain regions.  
The resulting posterior matrix drives VRAM allocation while preserving the
topological structure captured by the Sheaf.
"""

import math
import random
import sys
import pathlib
import hashlib
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import numpy as np


# ----------------------------------------------------------------------
# Core data structures (merged)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Entity:
    """Spatial entity with optional quasi‑identifier signature."""
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


@dataclass(frozen=True)
class ModelTier:
    """VRAM‑aware model tier used for scheduling."""
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


# ----------------------------------------------------------------------
# Sheaf implementation (from Parent A)
# ----------------------------------------------------------------------


class Sheaf:
    def __init__(self, node_dims: Dict[str, int], edge_list: List[Tuple[str, str]]):
        self.node_dims = dict(node_dims)          # node → dimension
        self.edges = list(edge_list)              # list of (src, dst)
        self._restrictions = {}                   # (u,v) → (src_map, dst_map)
        self._sections = {}                       # node → vector
        self._entropy = {}                        # node → float

    def set_restriction(self, edge: Tuple[str, str], src_map, dst_map):
        u, v = edge
        src_map = np.array(src_map, dtype=float)
        dst_map = np.array(dst_map, dtype=float)
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node: str, value):
        self._sections[node] = np.array(value, dtype=float)

    def set_entropy(self, node: str, entropy: float):
        self._entropy[node] = float(entropy)

    def get_entropy(self, node: str) -> float:
        return self._entropy.get(node, 0.0)

    def _edge_dim(self, u: str, v: str) -> int:
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        raise KeyError(f"No restriction map for edge ({u}, {v})")


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def shannon_entropy(probs: np.ndarray) -> float:
    """Return Shannon entropy (base‑2) of a probability vector."""
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1:
        raise ValueError("probs must be a 1‑D array")
    # Guard against zero entries
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat, lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


# ----------------------------------------------------------------------
# Hybrid construction: Sheaf from entities
# ----------------------------------------------------------------------


def build_sheaf_from_entities(entities: List[Entity]) -> Sheaf:
    """
    Construct a Sheaf where each entity is a node.
    - Node dimension = 3 (lat, lon, score)
    - Section vector = [lat, lon, score]
    - Entropy = Shannon entropy of the global categorical distribution,
      scaled per node (same value for all nodes in this simple example).
    No edges are created (empty edge list) because the focus is on the
    node‑wise prior modulation.
    """
    if not entities:
        raise ValueError("Entity list must not be empty")

    # Global category distribution
    cat_counts = defaultdict(int)
    for e in entities:
        cat_counts[e.category] += 1
    total = sum(cat_counts.values())
    cat_probs = np.array([cnt / total for cnt in cat_counts.values()], dtype=float)
    global_entropy = shannon_entropy(cat_probs)

    node_dims = {e.id: 3 for e in entities}
    sheaf = Sheaf(node_dims=node_dims, edge_list=[])

    for e in entities:
        sheaf.set_section(e.id, [e.lat, e.lon, e.score])
        # Simple scaling: more uncertain categories get higher entropy weight
        # Here we just reuse the global entropy for demonstration.
        sheaf.set_entropy(e.id, global_entropy)

    return sheaf


# ----------------------------------------------------------------------
# Hybrid Bayesian posterior using Sheaf entropy as prior modifier
# ----------------------------------------------------------------------


def compute_posterior(
    entities: List[Entity],
    model_tiers: List[ModelTier],
    sheaf: Sheaf,
) -> np.ndarray:
    """
    Compute the Bayesian posterior matrix Posterior_{ij} where
        p_i  = normalized (score_i) * exp(-H_i)
        L_ij = h_j * (1 - r_j)

    h_j (health‑derived reliability) is taken as inverse RAM usage:
        h_j = 1 / (ram_mb + 1)

    r_j (privacy reconstruction risk) is derived from VRAM pressure:
        r_j = vram_mb / (vram_mb + ram_mb + 1)

    The posterior is returned as a 2‑D NumPy array of shape (n_entities, n_models).
    """
    if not entities or not model_tiers:
        raise ValueError("Entities and model_tiers must be non‑empty")

    # ---------- Prior vector ----------
    raw_scores = np.array([e.score for e in entities], dtype=float)
    # Avoid division by zero
    if np.all(raw_scores == 0):
        raw_scores = np.ones_like(raw_scores)

    # Entropy‑modulated scaling
    entropies = np.array([sheaf.get_entropy(e.id) for e in entities], dtype=float)
    scaling = np.exp(-entropies)  # lower weight for high entropy
    prior_unnorm = raw_scores * scaling
    prior = prior_unnorm / prior_unnorm.sum()

    # ---------- Likelihood matrix ----------
    h_vals = np.array([1.0 / (mt.ram_mb + 1) for mt in model_tiers], dtype=float)
    r_vals = np.array([mt.vram_mb / (mt.vram_mb + mt.ram_mb + 1) for mt in model_tiers], dtype=float)
    likelihood = np.outer(np.ones_like(prior), h_vals * (1.0 - r_vals))  # shape (n_entities, n_models)

    # ---------- Posterior ----------
    numerator = prior[:, None] * likelihood
    denominator = numerator.sum()
    posterior = numerator / denominator
    return posterior


# ----------------------------------------------------------------------
# VRAM allocation based on posterior
# ----------------------------------------------------------------------


def allocate_vram(posterior: np.ndarray, model_tiers: List[ModelTier]) -> Dict[str, float]:
    """
    Allocate VRAM to each model tier proportionally to the summed posterior mass.
    Returns a mapping from model name to allocated VRAM (in MB).
    """
    if posterior.shape[1] != len(model_tiers):
        raise ValueError("Posterior column count must match number of model tiers")

    # Total posterior mass per model
    mass_per_model = posterior.sum(axis=0)  # shape (n_models,)

    # Total available VRAM is the sum of each tier's vram_mb
    total_vram = sum(mt.vram_mb for mt in model_tiers)

    allocation = {}
    for mt, mass in zip(model_tiers, mass_per_model):
        allocated = mass * total_vram
        allocation[mt.name] = allocated
    return allocation


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------


def demo_shannon_entropy():
    """Simple demo of the entropy utility."""
    probs = np.array([0.5, 0.3, 0.2])
    return shannon_entropy(probs)


def demo_build_and_inspect_sheaf():
    """Build a tiny sheaf from synthetic entities and return node entropies."""
    ents = [
        Entity(id="e1", lat=0.0, lon=0.0, category="A", score=1.0),
        Entity(id="e2", lat=1.0, lon=1.0, category="B", score=2.0),
        Entity(id="e3", lat=2.0, lon=2.0, category="A", score=3.0),
    ]
    sheaf = build_sheaf_from_entities(ents)
    return {nid: sheaf.get_entropy(nid) for nid in sheaf._sections}


def demo_full_pipeline():
    """Run the complete hybrid pipeline and return VRAM allocation."""
    # Synthetic data
    entities = [
        Entity(id="e1", lat=34.05, lon=-118.25, category="urban", score=0.8),
        Entity(id="e2", lat=40.71, lon=-74.00, category="urban", score=0.6),
        Entity(id="e3", lat=47.61, lon=-122.33, category="suburb", score=0.4),
        Entity(id="e4", lat=36.16, lon=-115.13, category="rural", score=0.2),
    ]

    model_tiers = [
        ModelTier(name="tiny", ram_mb=256, tier="T1", vram_mb=512),
        ModelTier(name="small", ram_mb=1024, tier="T2", vram_mb=2048),
        ModelTier(name="large", ram_mb=4096, tier="T3", vram_mb=8192),
    ]

    sheaf = build_sheaf_from_entities(entities)
    posterior = compute_posterior(entities, model_tiers, sheaf)
    allocation = allocate_vram(posterior, model_tiers)
    return allocation


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    print("Entropy demo:", demo_shannon_entropy())
    print("Sheaf node entropies:", demo_build_and_inspect_sheaf())
    print("VRAM allocation from full pipeline:", demo_full_pipeline())