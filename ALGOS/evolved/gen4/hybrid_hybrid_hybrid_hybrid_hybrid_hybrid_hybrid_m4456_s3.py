# DARWIN HAMMER — match 4456, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_nlms_omni_cha_m1_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_temporal_motifs_m280_s0.py (gen3)
# born: 2026-05-29T23:55:50Z

"""Hybrid Minimum-Cost Tree + NLMS + Temporal Motif Fusion

Parents:
- hybrid_hybrid_hybrid_minimu_hybrid_nlms_omni_cha_m1_s0.py (minimum‑cost tree with NLMS‑driven edge weights)
- hybrid_hybrid_hybrid_minimu_temporal_motifs_m280_s0.py (temporal‑motif mining with Bayesian updates)

Mathematical bridge:
Both parents expose the Bayesian marginal and update formulas.  In this hybrid we
use the NLMS prediction error as a *likelihood* term for Bayesian updating of
edge weights in the minimum‑cost tree **and** for updating the posterior
probability of temporal motifs.  The edge weight `w_ij` becomes

    w_ij = d_ij · (1 – ℓ_ij) · P(E|e)_ij

where `d_ij` is Euclidean distance, `ℓ_ij` is the NLMS prediction error for the
feature vector of node *j*, and `P(E|e)_ij` is the Bayesian posterior obtained
from a prior belief about the edge and the NLMS‑derived likelihood.  The same
posterior is later used to weight motif support counts, yielding a unified
probabilistic scoring across spatial (tree) and temporal (motif) domains.
"""

import math
import random
import sys
import pathlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric and Bayesian utilities (shared by both parents)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
NodeId = str
Edge = Tuple[NodeId, NodeId]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(e|E)·P(E) + P(e|~E)·P(~E)"""
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """P(E|e) = P(e|E)·P(E) / P(E)"""
    if marginal <= 0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# NLMS (Normalized Least‑Mean‑Squares) core
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the NLMS prediction ŷ = w·x / (ε + ||x||²)."""
    eps = 1e-8
    norm_sq = np.dot(x, x) + eps
    return float(np.dot(weights, x) / norm_sq)

def nlms_update(weights: np.ndarray, x: np.ndarray, d: float, mu: float = 0.1) -> np.ndarray:
    """
    Perform one NLMS weight update.
    w_{new} = w + μ·(d - ŷ)·x / (ε + ||x||²)
    """
    eps = 1e-8
    y_hat = nlms_predict(weights, x)
    error = d - y_hat
    norm_sq = np.dot(x, x) + eps
    adaptation = (mu * error / norm_sq) * x
    return weights + adaptation

# ----------------------------------------------------------------------
# Data structures for graph + temporal events
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    id: NodeId
    pos: Point
    features: np.ndarray  # arbitrary dimensional feature vector

@dataclass(frozen=True)
class EdgeData:
    src: NodeId
    dst: NodeId
    base_distance: float
    weight: float  # dynamic weight after NLMS + Bayesian update

@dataclass(frozen=True)
class BurstSignal:
    key: str
    count: int
    z_score: float
    prior: float
    likelihood: float
    false_positive: float

@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[NodeId, ...]
    support: int
    posterior: float  # Bayesian posterior after update

# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def update_edge_weights(
    nodes: Dict[NodeId, Node],
    edges: List[Edge],
    nlms_weights: np.ndarray,
    prior_edge: float = 0.5,
    false_positive: float = 0.1,
) -> List[EdgeData]:
    """
    Compute dynamic edge weights using NLMS predictions as likelihoods and
    Bayesian updating of a prior belief about each edge.
    """
    edge_data: List[EdgeData] = []
    for src_id, dst_id in edges:
        src = nodes[src_id]
        dst = nodes[dst_id]
        dist = length(src.pos, dst.pos)

        # NLMS prediction on destination features (treated as desired output d=1)
        pred = nlms_predict(nlms_weights, dst.features)
        likelihood = max(0.0, min(1.0, pred))          # clamp to [0,1]
        marginal = bayes_marginal(prior_edge, likelihood, false_positive)
        posterior = bayes_update(prior_edge, likelihood, marginal)

        # Final weight: distance scaled by (1‑likelihood) and posterior belief
        weight = dist * (1.0 - likelihood) * posterior
        edge_data.append(EdgeData(src=src_id, dst=dst_id,
                                  base_distance=dist, weight=weight))
    return edge_data

def minimum_cost_spanning_tree(edge_data: List[EdgeData]) -> List[EdgeData]:
    """
    Simple Kruskal implementation returning a subset of edges with minimal total
    dynamic weight that connects all nodes (assumes the graph is undirected and
    fully connected).  For brevity we use a union‑find structure.
    """
    # Union‑Find helpers
    parent: Dict[NodeId, NodeId] = {}

    def find(x: NodeId) -> NodeId:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: NodeId, b: NodeId) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    # Initialise each node as its own component
    nodes = {e.src for e in edge_data} | {e.dst for e in edge_data}
    for n in nodes:
        parent[n] = n

    # Sort edges by dynamic weight
    sorted_edges = sorted(edge_data, key=lambda e: e.weight)
    mst: List[EdgeData] = []
    for e in sorted_edges:
        if union(e.src, e.dst):
            mst.append(e)
        if len(mst) == len(nodes) - 1:
            break
    return mst

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

def detect_bursts(
    events: List[Dict],
    key: str = 'type',
    prior: float = 0.5,
    likelihood: float = 0.8,
    false_positive: float = 0.1,
) -> List[BurstSignal]:
    """Detect bursty keys and apply Bayesian update to their confidence."""
    counter = Counter(str(e.get(key, '')) for e in events)
    if not counter:
        return []
    mean = sum(counter.values()) / len(counter)
    sd = math.sqrt(sum((v - mean) ** 2 for v in counter.values()) / len(counter)) or 1.0
    bursts: List[BurstSignal] = []
    for k, v in counter.items():
        z = (v - mean) / sd
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal)
        bursts.append(BurstSignal(key=k, count=v, z_score=z,
                                  prior=prior, likelihood=likelihood,
                                  false_positive=false_positive))
    return bursts

def mine_temporal_motifs(
    sessions: List[List[Dict]],
    min_support: int = 2,
    prior: float = 0.5,
    likelihood: float = 0.7,
    false_positive: float = 0.1,
) -> List[TemporalMotif]:
    """
    Extract simple sequential motifs (ordered node ids) from each session,
    count their support, and apply a Bayesian posterior to the support.
    """
    motif_counter: Counter[Tuple[NodeId, ...]] = Counter()
    for sess in sessions:
        pattern = tuple(str(e.get('node', '')) for e in sess if 'node' in e)
        if pattern:
            motif_counter[pattern] += 1

    motifs: List[TemporalMotif] = []
    for pat, sup in motif_counter.items():
        if sup >= min_support:
            marginal = bayes_marginal(prior, likelihood, false_positive)
            posterior = bayes_update(prior, likelihood, marginal)
            motifs.append(TemporalMotif(pattern=pat, support=sup, posterior=posterior))
    return motifs

def hybrid_pipeline(
    nodes: List[Node],
    edges: List[Edge],
    events: List[Dict],
    nlms_mu: float = 0.05,
    nlms_steps: int = 5,
) -> Tuple[List[EdgeData], List[EdgeData], List[TemporalMotif]]:
    """
    End‑to‑end hybrid execution:
    1. Initialise NLMS weights from the dimensionality of node features.
    2. Run a few NLMS adaptation steps on random node feature samples.
    3. Update edge weights using NLMS predictions + Bayesian update.
    4. Compute a minimum‑cost spanning tree on the dynamic weights.
    5. Sessionize events, detect bursts, and mine temporal motifs with Bayesian
       posterior weighting.
    Returns (all_edge_data, mst_edge_data, motifs).
    """
    if not nodes:
        raise ValueError("At least one node required")
    dim = nodes[0].features.shape[0]
    w = np.zeros(dim)

    # Simple synthetic NLMS training: predict constant d=1 for random nodes
    rng = np.random.default_rng(42)
    for _ in range(nlms_steps):
        sample = rng.choice(nodes)
        w = nlms_update(w, sample.features, d=1.0, mu=nlms_mu)

    node_dict = {n.id: n for n in nodes}
    all_edge_data = update_edge_weights(node_dict, edges, w)

    mst_edge_data = minimum_cost_spanning_tree(all_edge_data)

    sessions = sessionize_events(events)
    motifs = mine_temporal_motifs(sessions)

    return all_edge_data, mst_edge_data, motifs

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic graph
    nodes = [
        Node(id="A", pos=(0.0, 0.0), features=np.array([0.2, 0.5])),
        Node(id="B", pos=(1.0, 0.0), features=np.array([0.1, 0.4])),
        Node(id="C", pos=(0.0, 1.0), features=np.array([0.3, 0.6])),
        Node(id="D", pos=(1.0, 1.0), features=np.array([0.4, 0.7])),
    ]
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("B", "C")]

    # Synthetic event stream (node visits with timestamps)
    events = [
        {"t": 10, "node": "A", "type": "click"},
        {"t": 20, "node": "B", "type": "click"},
        {"t": 30, "node": "C", "type": "view"},
        {"t": 2000, "node": "A", "type": "click"},
        {"t": 2010, "node": "D", "type": "view"},
        {"t": 2025, "node": "C", "type": "click"},
    ]

    all_edges, mst_edges, motifs = hybrid_pipeline(nodes, edges, events)

    print("All dynamic edges:")
    for e in all_edges:
        print(f"  {e.src}->{e.dst}  weight={e.weight:.4f}")

    print("\nMinimum‑cost spanning tree edges:")
    for e in mst_edges:
        print(f"  {e.src}->{e.dst}  weight={e.weight:.4f}")

    print("\nTemporal motifs (posterior weighted):")
    for m in motifs:
        print(f"  pattern={m.pattern}  support={m.support}  posterior={m.posterior:.4f}")

    sys.exit(0)