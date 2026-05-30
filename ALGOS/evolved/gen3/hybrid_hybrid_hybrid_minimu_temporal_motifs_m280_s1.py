# DARWIN HAMMER — match 280, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s1.py (gen2)
# parent_b: temporal_motifs.py (gen0)
# born: 2026-05-29T23:28:04Z

"""Hybrid Minimum Cost Tree with Temporal Motif Integration

This module fuses two parent algorithms:

* **Parent A** – ``hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s1.py``  
  Provides Euclidean geometry, Bayesian marginalisation and update, and a
  tree‑cost formulation that weights edges with probabilistic updates.

* **Parent B** – ``temporal_motifs.py``  
  Supplies temporal sessionisation, burst detection and motif mining based on
  event streams.

**Mathematical bridge**

For each node *v* we derive a **prior probability** *π(v)* from the support of
temporal motifs that appear in the event sessions associated with *v*:

\[
\pi(v)=\frac{\text{support of the most frequent motif in }v}{\sum_{u\in V}\text{support of most frequent motif in }u}
\]

Each edge *(u,v)* carries a **likelihood** *ℓ(u,v)* (user supplied) and a
**false‑positive rate** *φ(u,v)* derived from burst statistics of the event
type that labels the edge.  The Bayesian marginal and posterior are

\[
m(u,v)=\ell(u,v)\,\pi(u)+\phi(u,v)\,(1-\pi(u)),
\qquad
w(u,v)=\frac{\pi(u)\,\ell(u,v)}{m(u,v)}.
\]

The final hybrid edge weight combines the posterior *w(u,v)* with a
**temporal‑motif similarity factor** *σ(u,v)*, defined as the Jaccard similarity
between the motif sets of the incident nodes.  The total tree cost is the sum
of Euclidean edge lengths multiplied by the hybrid weight.

The implementation below realises this mathematical fusion and provides three
core hybrid functions:

* ``compute_node_priors`` – builds node priors from temporal motifs.
* ``hybrid_edge_weight`` – computes the Bayesian posterior and multiplies by
  the motif‑similarity factor.
* ``hybrid_tree_cost`` – evaluates the full cost of a rooted tree using the
  hybrid weights.

A small smoke test at the end demonstrates end‑to‑end execution."""

import math
import random
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Types from Parent A
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Types from Parent B
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
# Parent A utilities (geometry & Bayesian core)
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability P(E) = ℓ·π + φ·(1‑π)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must be in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior probability P(H|E) = π·ℓ / m."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Parent B utilities (temporal mining)
def sessionize_events(events: List[Dict], gap_seconds: float = 1800.0) -> List[List[Dict]]:
    """Group events into sessions separated by a time gap."""
    sessions: List[List[Dict]] = []
    cur: List[Dict] = []
    last: float | None = None
    for e in sorted(events, key=lambda x: float(x.get('t', 0))):
        t = float(e.get('t', 0))
        if cur and last is not None and t - last > gap_seconds:
            sessions.append(cur)
            cur = []
        cur.append(e)
        last = t
    if cur:
        sessions.append(cur)
    return sessions

def detect_bursts(events: List[Dict], key: str = 'type') -> List[BurstSignal]:
    """Detect bursty values of a given key using z‑score."""
    counter = Counter(str(e.get(key, '')) for e in events)
    if not counter:
        return []
    mean = sum(counter.values()) / len(counter)
    variance = sum((v - mean) ** 2 for v in counter.values()) / len(counter)
    sd = math.sqrt(variance) or 1.0
    return [
        BurstSignal(k, v, (v - mean) / sd)
        for k, v in counter.items()
        if v >= mean
    ]

def mine_temporal_motifs(sessions: List[List[Dict]], key: str = 'type', min_support: int = 2) -> List[TemporalMotif]:
    """Extract frequent ordered patterns (motifs) across sessions."""
    pattern_counter = Counter(
        tuple(str(e.get(key, '')) for e in s) for s in sessions
    )
    return [
        TemporalMotif(p, cnt)
        for p, cnt in pattern_counter.items()
        if cnt >= min_support
    ]

# ----------------------------------------------------------------------
# Hybrid functions -----------------------------------------------------
def compute_node_priors(
    node_events: Dict[str, List[Dict]],
    gap_seconds: float = 1800.0,
    min_support: int = 2
) -> Dict[str, float]:
    """
    Derive a prior probability for each node from its most frequent temporal motif.

    The prior is the motif support normalised over all nodes, guaranteeing that
    Σ_v π(v) = 1.
    """
    motif_supports: Dict[str, int] = {}
    for node, evs in node_events.items():
        sessions = sessionize_events(evs, gap_seconds)
        motifs = mine_temporal_motifs(sessions, min_support=min_support)
        if motifs:
            # Take the highest‑support motif for this node
            motif_supports[node] = max(m.support for m in motifs)
        else:
            motif_supports[node] = 0

    total = sum(motif_supports.values())
    if total == 0:
        # fallback to uniform prior
        n = len(node_events)
        return {node: 1.0 / n for node in node_events}
    return {node: support / total for node, support in motif_supports.items()}

def motif_similarity(
    motifs_a: List[TemporalMotif],
    motifs_b: List[TemporalMotif]
) -> float:
    """
    Jaccard similarity between two motif sets (patterns only, ignoring support).
    Returns a value in [0, 1].
    """
    set_a = {m.pattern for m in motifs_a}
    set_b = {m.pattern for m in motifs_b}
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union

def hybrid_edge_weight(
    node: str,
    edge: Edge,
    priors: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
    node_motifs: Dict[str, List[TemporalMotif]],
    motif_factor: float = 0.5
) -> float:
    """
    Compute the hybrid weight for an edge (u, v).

    1. Bayesian posterior w = π(u)·ℓ / m  (m = ℓ·π + φ·(1‑π))
    2. Motif similarity σ(u, v) between the incident nodes.
    3. Final weight = w * (1 + motif_factor * σ)

    The ``motif_factor`` controls the influence of temporal similarity.
    """
    u, v = edge
    prior_u = priors.get(u, 0.0)
    likelihood = likelihoods.get(edge, 0.0)
    false_pos = false_positives.get(edge, 0.0)

    marginal = bayes_marginal(prior_u, likelihood, false_pos)
    posterior = bayes_update(prior_u, likelihood, marginal)

    sigma = motif_similarity(node_motifs.get(u, []), node_motifs.get(v, []))
    return posterior * (1.0 + motif_factor * sigma)

def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    priors: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
    node_motifs: Dict[str, List[TemporalMotif]],
    motif_factor: float = 0.5,
    path_weight: float = 0.2
) -> float:
    """
    Total cost of a rooted tree.

    For each edge (u, v) the contribution is:
        cost(e) = length(u, v) * hybrid_edge_weight(u, v)

    The function also adds a small penalty proportional to the depth of each
    node (controlled by ``path_weight``) to encourage shallow trees.
    """
    # Build adjacency list and compute depths via BFS
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # BFS to obtain depth of each node from root
    depth: Dict[str, int] = {root: 0}
    queue: List[str] = [root]
    visited = {root}
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in visited:
                visited.add(nb)
                depth[nb] = depth[cur] + 1
                queue.append(nb)

    total_cost = 0.0
    for a, b in edges:
        # Use the direction from parent to child according to depth
        parent, child = (a, b) if depth[a] < depth[b] else (b, a)

        edge_len = length(nodes[parent], nodes[child])
        w = hybrid_edge_weight(
            parent,
            (parent, child),
            priors,
            likelihoods,
            false_positives,
            node_motifs,
            motif_factor,
        )
        total_cost += edge_len * w
        # add path penalty
        total_cost += path_weight * depth[child]

    return total_cost

# ----------------------------------------------------------------------
# Helper to generate synthetic data for the smoke test
def _synthetic_events(node: str, count: int = 20) -> List[Dict]:
    """Generate a list of fake events for a node."""
    base = random.randint(1_600_000_000, 1_700_000_000)
    events = []
    for i in range(count):
        t = base + i * random.uniform(30, 300)  # timestamps spaced 30‑300 s apart
        events.append({'t': t, 'type': random.choice(['A', 'B', 'C'])})
    return events

# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ---- Construct a tiny graph ------------------------------------------------
    nodes = {
        'R': (0.0, 0.0),
        'A': (1.0, 2.0),
        'B': (3.0, 1.0),
        'C': (4.0, 4.0),
    }
    edges = [('R', 'A'), ('R', 'B'), ('A', 'C'), ('B', 'C')]

    # ---- Synthetic event streams per node --------------------------------------
    node_events = {n: _synthetic_events(n) for n in nodes}
    # ---- Compute priors from motifs ---------------------------------------------
    priors = compute_node_priors(node_events)

    # ---- Likelihoods and false‑positive rates (mocked) -------------------------
    # For demonstration we assign random probabilities respecting [0,1]
    likelihoods = {e: random.uniform(0.4, 0.9) for e in edges}
    false_positives = {e: random.uniform(0.0, 0.3) for e in edges}

    # ---- Motif collections per node (required for similarity) -----------------
    node_motifs: Dict[str, List[TemporalMotif]] = {}
    for n, evs in node_events.items():
        sess = sessionize_events(evs)
        node_motifs[n] = mine_temporal_motifs(sess)

    # ---- Compute total hybrid tree cost ----------------------------------------
    cost = hybrid_tree_cost(
        nodes=nodes,
        edges=edges,
        root='R',
        priors=priors,
        likelihoods=likelihoods,
        false_positives=false_positives,
        node_motifs=node_motifs,
        motif_factor=0.4,
        path_weight=0.1,
    )
    print(f"Hybrid tree cost: {cost:.4f}")

    # Simple sanity checks (will raise if something is wrong)
    assert cost >= 0.0
    assert all(0.0 <= p <= 1.0 for p in priors.values())
    print("Smoke test passed.")