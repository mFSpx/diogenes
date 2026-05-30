# DARWIN HAMMER — match 1392, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_bayes_claim_k_m455_s2.py (gen4)
# parent_b: hybrid_temporal_motifs_possum_filter_m87_s0.py (gen1)
# born: 2026-05-29T23:35:53Z

"""Hybrid algorithm merging:
- hybrid_hybrid_doomsday_cale_hybrid_bayes_claim_k_m455_s2 (Gini coefficient, Bayesian update, tree cost)
- hybrid_temporal_motifs_possum_filter_m87_s0 (spatial entity filtering, burst detection, temporal motif mining)

Mathematical bridge:
Both parents manipulate probability distributions over discrete states.
Parent A measures inequality of a distribution with the Gini coefficient and refines it via a Bayesian update.
Parent B builds a distribution of categorical events (bursts) and filters spatial entities.
The hybrid therefore:
1. Constructs a categorical probability vector from entity scores.
2. Refines it with Bayesian evidence derived from burst‑signal statistics.
3. Uses the refined probabilities to weight a minimum‑cost spanning‑tree on the entities’ geographic positions,
   while the Gini of the weighted distribution quantifies the remaining uncertainty.
"""

import math
import random
import sys
import pathlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Types shared by both parents
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

@dataclass(frozen=True)
class BurstSignal:
    key: str
    count: int
    z_score: float

@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int

# ----------------------------------------------------------------------
# Parent A utilities
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def gini_coefficient(values: List[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

def bayesian_update(prior: Dict[Any, float], likelihood: Dict[Any, float]) -> Dict[Any, float]:
    """Return posterior ∝ prior * likelihood, normalized."""
    posterior_unnorm = {k: prior.get(k, 0.0) * likelihood.get(k, 0.0) for k in set(prior) | set(likelihood)}
    total = sum(posterior_unnorm.values())
    if total == 0:
        # fallback to uniform distribution over observed keys
        n = len(posterior_unnorm)
        return {k: 1.0 / n for k in posterior_unnorm}
    return {k: v / total for k, v in posterior_unnorm.items()}

def tree_cost(nodes: Dict[str, Point], edges: List[Edge], root: str, path_weight: float = 0.2) -> float:
    """Simple cost: sum of edge lengths multiplied by path_weight."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    visited = set()
    stack = [(root, 0.0)]
    total = 0.0
    while stack:
        node, acc = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        total += acc
        for neigh in adj[node]:
            if neigh not in visited:
                edge_len = length(nodes[node], nodes[neigh])
                stack.append((neigh, acc + edge_len * path_weight))
    return total

# ----------------------------------------------------------------------
# Parent B utilities
# ----------------------------------------------------------------------
def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()

def keep_candidate(candidate: Entity, selected: List[Entity], delta_m: float) -> bool:
    for existing in selected:
        same_kind = signature(candidate) == signature(existing) or \
                    candidate.category.strip().lower() == existing.category.strip().lower()
        if same_kind and haversine_m((candidate.lat, candidate.lon),
                                    (existing.lat, existing.lon)) <= delta_m:
            return False
    return True

def sessionize_events(events: List[Dict[str, Any]], gap_seconds: float = 1800.0) -> List[List[Dict[str, Any]]]:
    """Group events into sessions separated by > gap_seconds."""
    if not events:
        return []
    sorted_events = sorted(events, key=lambda e: e["timestamp"])
    sessions = []
    current = [sorted_events[0]]
    for ev in sorted_events[1:]:
        if ev["timestamp"] - current[-1]["timestamp"] > gap_seconds:
            sessions.append(current)
            current = [ev]
        else:
            current.append(ev)
    sessions.append(current)
    return sessions

def detect_bursts(events: List[Dict[str, Any]], key: str = "category") -> List[BurstSignal]:
    """Count occurrences per key and compute z‑score against global mean/std."""
    counts = Counter(ev[key] for ev in events)
    if not counts:
        return []
    vals = np.array(list(counts.values()), dtype=float)
    mean, std = vals.mean(), vals.std(ddof=0)
    std = std if std > 0 else 1.0
    bursts = [BurstSignal(k, c, (c - mean) / std) for k, c in counts.items()]
    return bursts

def mine_temporal_motifs(sessions: List[List[Dict[str, Any]]],
                         key: str = "category",
                         min_support: int = 2) -> List[TemporalMotif]:
    """Extract frequent ordered pairs (length‑2 motifs) across sessions."""
    pair_counter = Counter()
    for sess in sessions:
        seq = [ev[key] for ev in sess]
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counter[pair] += 1
    motifs = [TemporalMotif(p, cnt) for p, cnt in pair_counter.items() if cnt >= min_support]
    return motifs

# ----------------------------------------------------------------------
# Hybrid operations (the required three+ functions)
# ----------------------------------------------------------------------
def compute_category_distribution(entities: Iterable[Entity]) -> Dict[str, float]:
    """Build a probability distribution over categories weighted by entity scores."""
    cat_score = defaultdict(float)
    total = 0.0
    for e in entities:
        w = max(e.score, 0.0)  # ensure non‑negative
        cat_score[e.category] += w
        total += w
    if total == 0.0:
        # uniform fallback
        uniq = {e.category for e in entities}
        return {c: 1.0 / len(uniq) for c in uniq}
    return {c: v / total for c, v in cat_score.items()}

def hybrid_gini_bayes_tree(entities: List[Entity],
                           events: List[Dict[str, Any]],
                           delta_m: float = 500.0,
                           root_id: str = None) -> Tuple[float, Dict[str, float], float]:
    """
    1. Filter spatially redundant entities (Possum filter).
    2. Build prior category distribution from remaining entities.
    3. Detect bursts → likelihood per category.
    4. Bayesian update to obtain posterior.
    5. Compute Gini of posterior (uncertainty measure).
    6. Build a minimum‑cost tree on the filtered entities and weight it by posterior probabilities.
    Returns (gini, posterior_distribution, tree_cost).
    """
    # ---- 1. Spatial filtering ----
    selected: List[Entity] = []
    for e in sorted(entities, key=lambda x: (-x.score, x.id)):
        if keep_candidate(e, selected, delta_m):
            selected.append(e)

    if not selected:
        raise ValueError("No entities survive spatial filtering")

    # ---- 2. Prior distribution ----
    prior = compute_category_distribution(selected)

    # ---- 3. Burst detection -> likelihood ----
    bursts = detect_bursts(events, key="category")
    likelihood = {b.key: max(b.z_score, 0.0) + 1e-6 for b in bursts}  # shift to positive
    # Ensure all prior keys appear in likelihood (with small epsilon if missing)
    for k in prior:
        likelihood.setdefault(k, 1e-6)

    # ---- 4. Bayesian update ----
    posterior = bayesian_update(prior, likelihood)

    # ---- 5. Gini of posterior ----
    gini = gini_coefficient(list(posterior.values()))

    # ---- 6. Tree cost weighted by posterior ----
    nodes = {e.id: (e.lat, e.lon) for e in selected}
    # simple full‑graph edges (complete graph) for demonstration
    edges: List[Edge] = []
    ids = list(nodes.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            edges.append((ids[i], ids[j]))
    root = root_id if root_id and root_id in nodes else ids[0]
    # weight path by posterior probability of the root's category (as an example)
    root_entity = next(e for e in selected if e.id == root)
    path_weight = posterior.get(root_entity.category, 0.0) + 0.01  # avoid zero
    cost = tree_cost(nodes, edges, root, path_weight=path_weight)
    return gini, posterior, cost

def hybrid_motif_analysis(entities: List[Entity],
                          events: List[Dict[str, Any]],
                          gap_seconds: float = 1800.0,
                          min_support: int = 2) -> List[TemporalMotif]:
    """
    Combine spatial filtering with temporal motif mining:
    * Keep only entities that survive the Posium filter.
    * Sessionize events.
    * Mine frequent ordered pairs (motifs).
    Returns the list of motifs.
    """
    # Spatial filter (reuse keep_candidate)
    filtered = []
    for e in sorted(entities, key=lambda x: (-x.score, x.id)):
        if keep_candidate(e, filtered, delta_m=500.0):
            filtered.append(e)

    # Sessionize and mine motifs
    sessions = sessionize_events(events, gap_seconds=gap_seconds)
    motifs = mine_temporal_motifs(sessions, key="category", min_support=min_support)
    return motifs

def hybrid_entity_scoring(entities: List[Entity],
                          events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Produce a refined score for each entity by:
    1. Computing the posterior category distribution (as in hybrid_gini_bayes_tree).
    2. Assigning to each entity the posterior probability of its category multiplied by its original score.
    Returns a mapping id -> refined_score.
    """
    # Prior from all entities (no spatial filter for scoring)
    prior = compute_category_distribution(entities)
    bursts = detect_bursts(events, key="category")
    likelihood = {b.key: max(b.z_score, 0.0) + 1e-6 for b in bursts}
    for k in prior:
        likelihood.setdefault(k, 1e-6)
    posterior = bayesian_update(prior, likelihood)

    refined = {}
    for e in entities:
        refined[e.id] = e.score * posterior.get(e.category, 0.0)
    return refined

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy entities
    ents = [
        Entity(id="A", lat=40.0, lon=-74.0, category="alpha", score=10.0),
        Entity(id="B", lat=40.001, lon=-74.001, category="beta", score=8.0),
        Entity(id="C", lat=40.1, lon=-74.1, category="alpha", score=5.0),
        Entity(id="D", lat=41.0, lon=-75.0, category="gamma", score=7.0),
    ]

    # Dummy events (timestamp in seconds)
    evts = [
        {"timestamp": 1000, "category": "alpha"},
        {"timestamp": 1100, "category": "alpha"},
        {"timestamp": 2500, "category": "beta"},
        {"timestamp": 2600, "category": "beta"},
        {"timestamp": 5000, "category": "gamma"},
        {"timestamp": 5100, "category": "gamma"},
        {"timestamp": 5200, "category": "gamma"},
    ]

    gini, posterior, cost = hybrid_gini_bayes_tree(ents, evts, delta_m=200.0, root_id="A")
    print(f"Gini of posterior: {gini:.4f}")
    print(f"Posterior distribution: {posterior}")
    print(f"Weighted tree cost: {cost:.2f}")

    motifs = hybrid_motif_analysis(ents, evts)
    print(f"Detected motifs (min_support=2): {[m.pattern for m in motifs]}")

    refined_scores = hybrid_entity_scoring(ents, evts)
    print(f"Refined entity scores: {refined_scores}")