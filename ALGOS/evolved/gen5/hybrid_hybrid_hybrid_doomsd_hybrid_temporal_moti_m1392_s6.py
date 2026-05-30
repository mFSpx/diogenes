# DARWIN HAMMER — match 1392, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_bayes_claim_k_m455_s2.py (gen4)
# parent_b: hybrid_temporal_motifs_possum_filter_m87_s0.py (gen1)
# born: 2026-05-29T23:35:53Z

"""Hybrid Doomsday Temporal Motif Fusion
====================================

This module merges the core mathematics of two parent algorithms:

* **Parent A** – *hybrid_hybrid_doomsday_cale_hybrid_bayes_claim_k_m455_s2.py*  
  Provides a Gini‑coefficient calculation over probability distributions,
  a Bayesian update rule for those distributions and a tree‑cost evaluator
  based on Euclidean geometry.

* **Parent B** – *hybrid_temporal_motifs_possum_filter_m87_s0.py*  
  Supplies spatial filtering of entities via the Haversine distance,
  sessionisation of time‑stamped events, burst detection and temporal‑motif
  mining.

**Mathematical Bridge**  
Both parents operate on *distributions* over discrete categories:

* A treats a probability vector **p** (e.g. belief over system states) and
  measures its inequality with the **Gini coefficient**.  
* B produces categorical frequencies from event streams (burst counts) and
  from spatially filtered entities.

The fusion therefore uses the **category frequency vector** derived from
entities as a *prior* distribution, updates it with **burst evidence** via a
Dirichlet‑style Bayesian rule, evaluates the resulting *posterior* with the
Gini coefficient and finally incorporates the spatial network cost (a
minimum‑spanning‑tree cost) to obtain a single hybrid score.

The following implementation realises this pipeline while keeping the
original mathematical intent of both parents."""
import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple, Iterable, Any
import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities (Parent A & B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat, lon) pairs."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

# ----------------------------------------------------------------------
# Gini coefficient (Parent A)
# ----------------------------------------------------------------------
def gini_coefficient(values: List[float]) -> float:
    """Return the Gini coefficient of a non‑negative value list."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

# ----------------------------------------------------------------------
# Simple entity model (Parent B)
# ----------------------------------------------------------------------
class Entity:
    __slots__ = ("id", "lat", "lon", "category", "score", "address_signature")
    def __init__(self, id: str, lat: float, lon: float,
                 category: str, score: float = 0.0,
                 address_signature: str = "") -> None:
        self.id = id
        self.lat = lat
        self.lon = lon
        self.category = category
        self.score = score
        self.address_signature = address_signature

# ----------------------------------------------------------------------
# Spatial filtering utilities (Parent B)
# ----------------------------------------------------------------------
def signature(e: Entity) -> str:
    """Canonical signature used for deduplication."""
    return (e.address_signature or e.category).strip().lower()

def keep_candidate(candidate: Entity, selected: List[Entity], delta_m: float) -> bool:
    """Return True if *candidate* is sufficiently far from all *selected*."""
    for existing in selected:
        same_kind = (
            signature(candidate) == signature(existing) or
            candidate.category.strip().lower() == existing.category.strip().lower()
        )
        if same_kind and haversine_m((candidate.lat, candidate.lon),
                                    (existing.lat, existing.lon)) <= delta_m:
            return False
    return True

def spatial_filter(entities: Iterable[Entity], delta_m: float) -> List[Entity]:
    """Deduplicate entities respecting a minimum Haversine separation."""
    ordered = list(entities)
    ordered.sort(key=lambda e: (-e.score, e.id))
    selected: List[Entity] = []
    for e in ordered:
        if keep_candidate(e, selected, delta_m):
            selected.append(e)
    return selected

# ----------------------------------------------------------------------
# Temporal utilities (Parent B)
# ----------------------------------------------------------------------
def sessionize_events(events: List[Dict[str, Any]], gap_seconds: float = 1800.0) -> List[List[Dict[str, Any]]]:
    """Group events into sessions; a new session starts after a gap > gap_seconds."""
    if not events:
        return []
    # Ensure chronological order
    events = sorted(events, key=lambda e: e["timestamp"])
    sessions: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [events[0]]
    for ev in events[1:]:
        if ev["timestamp"] - current[-1]["timestamp"] > gap_seconds:
            sessions.append(current)
            current = [ev]
        else:
            current.append(ev)
    sessions.append(current)
    return sessions

def detect_bursts(events: List[Dict[str, Any]], key: str = "category") -> List[Tuple[str, int, float]]:
    """Count occurrences per *key* and compute a simple z‑score burst metric."""
    counts: Dict[str, int] = {}
    for ev in events:
        k = ev.get(key, "")
        counts[k] = counts.get(k, 0) + 1
    values = np.array(list(counts.values()))
    if len(values) == 0:
        mean = std = 0.0
    else:
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    bursts = []
    for k, c in counts.items():
        z = (c - mean) / std if std > 0 else 0.0
        bursts.append((k, c, z))
    return bursts

def mine_temporal_motifs(sessions: List[List[Dict[str, Any]]],
                         key: str = "category",
                         min_support: int = 2) -> List[Tuple[Tuple[str, ...], int]]:
    """Extract length‑2 sequential patterns that appear in at least *min_support* sessions."""
    pattern_counter: Dict[Tuple[str, ...], int] = {}
    for sess in sessions:
        seq = [ev.get(key, "") for ev in sess]
        # Extract all adjacent pairs
        for i in range(len(seq) - 1):
            pat = (seq[i], seq[i + 1])
            pattern_counter[pat] = pattern_counter.get(pat, 0) + 1
    motifs = [(pat, sup) for pat, sup in pattern_counter.items() if sup >= min_support]
    motifs.sort(key=lambda x: -x[1])
    return motifs

# ----------------------------------------------------------------------
# Bayesian update (Parent A) – Dirichlet‑like formulation
# ----------------------------------------------------------------------
def bayesian_update(prior: List[float],
                    evidence: List[int],
                    alpha: float = 1.0) -> List[float]:
    """
    Perform a simple Dirichlet update:
        posterior_i ∝ prior_i * alpha + evidence_i
    Returns a normalized probability vector.
    """
    if len(prior) != len(evidence):
        raise ValueError("prior and evidence must have the same length")
    posterior_unnorm = [p * alpha + e for p, e in zip(prior, evidence)]
    total = sum(posterior_unnorm)
    if total == 0:
        # fallback to uniform distribution
        return [1.0 / len(prior)] * len(prior)
    return [x / total for x in posterior_unnorm]

# ----------------------------------------------------------------------
# Minimum‑spanning‑tree cost (adapted from Parent A)
# ----------------------------------------------------------------------
def mst_cost(nodes: Dict[str, Point], weight_factor: float = 0.2) -> float:
    """
    Compute the total length of a Minimum Spanning Tree (Prim's algorithm)
    over the supplied *nodes* (id → (x, y) or (lat, lon) coordinates).
    The final cost is multiplied by *weight_factor* to emulate the
    path‑weight scaling used in the original tree_cost routine.
    """
    if not nodes:
        return 0.0
    # Convert to list for index handling
    ids = list(nodes.keys())
    n = len(ids)
    in_tree = [False] * n
    min_edge = [math.inf] * n
    min_edge[0] = 0.0
    total = 0.0

    for _ in range(n):
        # Pick the vertex with the smallest connecting edge not yet in the tree
        u = -1
        best = math.inf
        for i in range(n):
            if not in_tree[i] and min_edge[i] < best:
                best = min_edge[i]
                u = i
        if u == -1:
            break
        in_tree[u] = True
        total += best
        # Update neighbouring vertices
        for v in range(n):
            if not in_tree[v]:
                d = length(nodes[ids[u]], nodes[ids[v]])
                if d < min_edge[v]:
                    min_edge[v] = d
    return total * weight_factor

# ----------------------------------------------------------------------
# Hybrid operation – three demonstrative functions
# ----------------------------------------------------------------------
def category_distribution(entities: List[Entity]) -> Tuple[List[str], List[float]]:
    """
    Build a categorical probability vector from *entities*.
    Returns (categories, probabilities) where probabilities sum to 1.
    """
    counter: Dict[str, int] = {}
    for e in entities:
        cat = signature(e)
        counter[cat] = counter.get(cat, 0) + 1
    total = sum(counter.values())
    categories = list(counter.keys())
    probs = [cnt / total for cnt in counter.values()] if total > 0 else [0.0] * len(categories)
    return categories, probs

def hybrid_score(entities: List[Entity],
                 events: List[Dict[str, Any]],
                 delta_m: float = 500.0,
                 gap_seconds: float = 1800.0,
                 alpha: float = 1.0,
                 weight_factor: float = 0.2) -> float:
    """
    Compute a unified hybrid metric:

    1. Spatially filter *entities*.
    2. Derive a prior categorical distribution.
    3. Sessionise *events* and detect bursts → evidence counts.
    4. Bayesian update of the prior with burst evidence.
    5. Evaluate inequality of the posterior via the Gini coefficient.
    6. Add the MST cost of the filtered entity locations.

    The returned float is a weighted sum: 0.7 * Gini + 0.3 * normalized MST cost.
    """
    # 1. Spatial filter
    filtered = spatial_filter(entities, delta_m)

    # 2. Prior distribution
    cats, prior = category_distribution(filtered)

    # 3. Sessionise and burst detection
    sessions = sessionize_events(events, gap_seconds)
    bursts = detect_bursts(events, key="category")
    # Align evidence with prior categories
    evidence = [0] * len(cats)
    burst_dict = {k: c for k, c, _ in bursts}
    for i, cat in enumerate(cats):
        evidence[i] = burst_dict.get(cat, 0)

    # 4. Bayesian update
    posterior = bayesian_update(prior, evidence, alpha)

    # 5. Gini of posterior
    gini = gini_coefficient(posterior)

    # 6. MST cost (using lat/lon as planar for simplicity)
    node_coords = {e.id: (e.lat, e.lon) for e in filtered}
    mst = mst_cost(node_coords, weight_factor)

    # Normalise MST cost to [0,1] (simple min‑max using a heuristic bound)
    # Assume Earth circumference ~40,000 km → max possible tree length ~ 40e6 m
    mst_norm = min(mst / 40_000_000.0, 1.0)

    # Weighted combination
    return 0.7 * gini + 0.3 * mst_norm

def demo_hybrid_pipeline() -> None:
    """Run a tiny demonstration of the hybrid algorithm."""
    # Synthetic entities
    ents = [
        Entity("A1", 40.7128, -74.0060, "restaurant", score=9.0),
        Entity("A2", 40.7130, -74.0055, "restaurant", score=8.5),
        Entity("B1", 34.0522, -118.2437, "cafe", score=7.0),
        Entity("C1", 51.5074, -0.1278, "museum", score=6.5),
        Entity("D1", 48.8566, 2.3522, "cafe", score=5.0),
    ]

    # Synthetic events (timestamps are seconds since epoch)
    base = 1_700_000_000
    evts = [
        {"timestamp": base + 10, "category": "restaurant"},
        {"timestamp": base + 20, "category": "cafe"},
        {"timestamp": base + 2000, "category": "museum"},
        {"timestamp": base + 2100, "category": "cafe"},
        {"timestamp": base + 4000, "category": "restaurant"},
        {"timestamp": base + 4100, "category": "restaurant"},
    ]

    score = hybrid_score(ents, evts)
    print(f"Hybrid metric: {score:.5f}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_hybrid_pipeline()