# DARWIN HAMMER — match 4294, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1411_s1.py (gen5)
# parent_b: hybrid_hybrid_rectified_flo_hybrid_hybrid_gini_c_m1244_s2.py (gen6)
# born: 2026-05-29T23:54:43Z

"""Hybrid Privacy-Flow-Gini Fusion
================================

This module fuses the two parent algorithms:

* **Parent A** – spatial‑aware privacy reconstruction risk (haversine distance,
  reconstruction risk scoring) applied to ``Entity`` objects.
* **Parent B** – Rectified Flow straight‑line interpolant together with a Gini‑
  based impurity measure and tropical (max‑plus) belief propagation.

Mathematical bridge
-------------------
Each ``Entity`` is mapped to a *risk scalar* `r_i` that quantifies the privacy
reconstruction risk of that tier.  The vector of risks `r = [r_1,…,r_N]` is used as
the *target* in the Rectified Flow interpolant


Z_t = (1‑t)·0 + t·r   =  t·r


where the source state is the zero vector.  The interpolated latent vector `Z_t`
acts as a feature set for the Gini impurity calculation.  The impurity of a
candidate split of the latent space is then turned into a log‑probability and
combined with tropical algebra:

* tropical multiplication `⊗`  → ordinary addition (`+`)
* tropical addition      `⊕`  → maximum (`max`)

Thus belief scores are propagated through a graph of ``ModelTier`` nodes by
max‑plus updates, yielding a unified score that respects both privacy risk and
the flow‑based representation quality.

The three core functions below demonstrate this hybrid pipeline:
``compute_entity_risk`` → ``flow_embed`` → ``tropical_belief_propagation``.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np


# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """A spatial entity with a categorical label."""
    id: str
    lat: float
    lon: float
    category: str
    address_signature: str = ""


@dataclass(frozen=True)
class ModelTier:
    """A model tier that will receive a belief score."""
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


# ----------------------------------------------------------------------
# Helper utilities (Parent A)
# ----------------------------------------------------------------------
def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat, lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """
    Simple surrogate for the privacy reconstruction risk used in Parent A.
    The score grows with the proportion of quasi‑identifiers.
    """
    if total_records <= 0:
        return 0.0
    return unique_quasi_identifiers / total_records


# ----------------------------------------------------------------------
# Hybrid core – bridging Parent A and Parent B
# ----------------------------------------------------------------------
def compute_entity_risk(
    entities: List[Entity],
    reference_point: Tuple[float, float],
    max_quasi_identifiers: int = 10,
) -> np.ndarray:
    """
    Compute a privacy‑risk vector for a list of entities.

    For each entity we combine:
      * a distance‑based attenuation (farther → lower risk)
      * a random quasi‑identifier count to emulate the reconstruction risk
    The final risk is normalised to the interval [0, 1].
    """
    risks = []
    for e in entities:
        dist = haversine_m((e.lat, e.lon), reference_point)  # metres
        # distance attenuation: 1 / (1 + d / 1e5)  (scale of 100 km)
        dist_factor = 1.0 / (1.0 + dist / 1e5)

        # simulate quasi‑identifier count
        qid = random.randint(1, max_quasi_identifiers)
        recon = reconstruction_risk_score(qid, max_quasi_identifiers)

        risk = dist_factor * recon
        risks.append(risk)

    risk_vec = np.array(risks, dtype=np.float64)
    # Normalise to [0,1] to keep the flow stable
    if risk_vec.max() > 0:
        risk_vec /= risk_vec.max()
    return risk_vec


# ----------------------------------------------------------------------
# Rectified Flow utilities (Parent B)
# ----------------------------------------------------------------------
def interpolant(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    """
    Straight‑line interpolant Z_t = (1‑t)·x0 + t·x1.
    x0 and x1 must be broadcastable; t is a scalar in [0,1].
    """
    return (1.0 - t) * x0 + t * x1


def flow_target(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Target velocity field v = x1 - x0."""
    return x1 - x0


def flow_embed(risk_vector: np.ndarray, t: float = 0.5) -> np.ndarray:
    """
    Embed the privacy‑risk vector using the rectified‑flow interpolant.
    The source state is the zero vector; the target is the risk vector.
    """
    zero = np.zeros_like(risk_vector)
    return interpolant(zero, risk_vector, t)


# ----------------------------------------------------------------------
# Gini impurity and tropical belief propagation (Parent B)
# ----------------------------------------------------------------------
def gini_coefficient(labels: List[int]) -> float:
    """
    Compute the Gini impurity for a list of integer class labels.
    """
    if not labels:
        return 0.0
    total = len(labels)
    counts = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    impurity = 1.0 - sum((c / total) ** 2 for c in counts.values())
    return impurity


def tropical_update(belief: np.ndarray, edge: Tuple[int, int]) -> np.ndarray:
    """
    Perform a single tropical (max‑plus) belief update along an edge (src, dst).

    belief_new[dst] = max(belief_new[dst], belief[src] + weight)
    where weight is the log‑probability derived from Gini impurity.
    """
    src, dst = edge
    # Convert impurity to a log‑probability: higher impurity → lower confidence
    weight = -math.log1p(gini_coefficient([src, dst]))
    belief[dst] = max(belief[dst], belief[src] + weight)
    return belief


def tropical_belief_propagation(
    embeddings: np.ndarray,
    edges: List[Tuple[int, int]],
    iterations: int = 5,
) -> np.ndarray:
    """
    Propagate belief scores over a graph using tropical algebra.

    *embeddings* – 1‑D array where each entry is the latent value of a node.
    *edges*      – list of directed edges (src_index, dst_index).
    The initial belief of node i is set to the embedding value (treated as a
    log‑probability).  Updates are performed ``iterations`` times.
    """
    belief = embeddings.copy()
    for _ in range(iterations):
        for e in edges:
            belief = tropical_update(belief, e)
    return belief


# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_score_pipeline(
    entities: List[Entity],
    reference_point: Tuple[float, float],
    model_tiers: List[ModelTier],
    edges: List[Tuple[int, int]],
    flow_time: float = 0.5,
) -> Dict[str, float]:
    """
    End‑to‑end hybrid computation:

    1. Compute privacy risk per entity.
    2. Embed the risk vector with rectified flow.
    3. Use the embedded values as initial log‑beliefs for the model‑tier graph.
    4. Run tropical belief propagation.
    5. Return a mapping from ``ModelTier.name`` to its final belief score.
    """
    # Step 1 – privacy risk
    risk_vec = compute_entity_risk(entities, reference_point)

    # Step 2 – flow embedding
    embedded = flow_embed(risk_vec, t=flow_time)

    # Align embedding length with the number of model tiers.
    # If they differ, truncate or pad with zeros.
    n_tiers = len(model_tiers)
    if embedded.shape[0] < n_tiers:
        padded = np.concatenate([embedded, np.zeros(n_tiers - embedded.shape[0])])
    else:
        padded = embedded[:n_tiers]

    # Step 3‑4 – tropical belief propagation over the tier graph
    final_belief = tropical_belief_propagation(padded, edges)

    # Step 5 – map back to tier names
    return {tier.name: float(final_belief[i]) for i, tier in enumerate(model_tiers)}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic world
    random.seed(42)
    np.random.seed(42)

    entities = [
        Entity(id="e1", lat=37.7749, lon=-122.4194, category="A"),
        Entity(id="e2", lat=34.0522, lon=-118.2437, category="B"),
        Entity(id="e3", lat=40.7128, lon=-74.0060, category="A"),
    ]

    reference = (36.0, -120.0)  # Roughly central California

    model_tiers = [
        ModelTier(name="tier_small", ram_mb=2048, tier="S", vram_mb=1024),
        ModelTier(name="tier_medium", ram_mb=8192, tier="M", vram_mb=4096),
        ModelTier(name="tier_large", ram_mb=16384, tier="L", vram_mb=8192),
    ]

    # Simple directed chain graph: small → medium → large
    edges = [(0, 1), (1, 2)]

    scores = hybrid_score_pipeline(
        entities,
        reference_point=reference,
        model_tiers=model_tiers,
        edges=edges,
        flow_time=0.6,
    )

    for name, val in scores.items():
        print(f"{name}: {val:.4f}")

    # Ensure the script exits cleanly
    sys.exit(0)