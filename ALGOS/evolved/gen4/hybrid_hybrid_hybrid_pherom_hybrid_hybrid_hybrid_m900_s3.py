# DARWIN HAMMER — match 900, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_pheromone_hyb_chelydrid_ambush_m183_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_ternary_route_m46_s1.py (gen3)
# born: 2026-05-29T23:31:31Z

"""Hybrid Algorithm: hybrid_hybrid_pheromone_hyb_chelydrid_ambush_m183_s2 + hybrid_hybrid_decisi_hybrid_ternary_route_m46_s1
Mathematical Bridge
-------------------
Parent A provides a pheromone‑based surface signal representation together with a
burst‑admission model (pulse force series).  Parent B supplies a Shannon‑entropy
measure over categorical evidence and uses that entropy to scale edge priors in a
minimum‑cost tree that is later updated by Bayesian rules.

The fusion treats the *pheromone signal vector* of each node as categorical
evidence.  From this vector we compute a Shannon entropy **H** (Parent B).  The
entropy is then mapped to edge priors **πₑ = exp(-H) / Σₑ' exp(-H)** – exactly the
formula used in Parent B – and the priors are multiplied by a *burst‑admission
score* derived from the pulse‑force series of the same node (Parent A).  The
resulting weighted scores drive a leader‑election routine on a graph whose
edges are the minimum‑cost tree of Parent B.

The module therefore integrates:
* Pheromone hashing & Hamming distance (Parent A)
* Pulse‑force generation & simple impulse integration (Parent A)
* Shannon entropy of discretised pheromone vectors (Parent B)
* Entropy‑scaled edge priors for a cost‑graph (Parent B)
* Hybrid leader selection using the product of burst score and edge prior
"""

import sys
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Iterable
import numpy as np
import re
from collections import Counter

# ----------------------------------------------------------------------
# Parent A – pheromone utilities
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Binary hash of a pheromone vector based on the mean threshold."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()

def broadcast_probability(phase: int, step: int) -> float:
    """Probability function used in the original distributed leader election."""
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))

def pulse_force(peak_force: float, steps: int) -> List[float]:
    """Triangular pulse force series."""
    if peak_force < 0 or steps <= 0:
        raise ValueError("peak_force non‑negative and steps positive required")
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    return [peak_force * max(0.0, 1.0 - abs(i - mid) / denom) for i in range(steps)]

def integrate_strike(force_series: Iterable[float], dt: float, m_head: float, drag_cd: float) -> float:
    """
    Very simple impulse‑based integration of a strike.
    Returns an approximate “burst admission score” proportional to the
    net momentum delivered after accounting for a quadratic drag term.
    """
    v = 0.0
    for f in force_series:
        # acceleration = (force - drag * v^2) / mass
        a = (f - drag_cd * v * v) / m_head
        v += a * dt
    # Return the final momentum (mass * velocity) as the score
    return m_head * v

# ----------------------------------------------------------------------
# Parent B – evidence extraction & entropy based edge priors
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)

def extract_evidence_features(text: str) -> List[str]:
    """Extract words that match the evidence regular expression."""
    return EVIDENCE_RE.findall(text.lower())

def shannon_entropy(categories: List[str]) -> float:
    """Compute Shannon entropy of a categorical list."""
    if not categories:
        return 0.0
    counts = Counter(categories)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()], dtype=float)
    return -np.sum(probs * np.log2(probs + 1e-12))

def entropy_weighted_edge_priors(edges: List[Tuple[int, int, float]], entropy: float) -> List[float]:
    """
    Scale each edge's base cost by a prior derived from the global entropy.
    Prior for each edge: πₑ = exp(-H) / Σₑ' exp(-H)  (uniform because H is global)
    The returned list contains πₑ * base_cost for each edge.
    """
    if not edges:
        return []
    base = math.exp(-entropy)
    uniform_prior = base / (base * len(edges))  # simplifies to 1/len(edges)
    # Multiply by the original edge cost to obtain a weighted cost
    return [uniform_prior * cost for (_, _, cost) in edges]

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def pheromone_entropy(pheromone_vectors: List[List[float]]) -> float:
    """
    Treat each pheromone vector as a bag of categorical tokens.
    Tokens are the string representation of discretised bins.
    The global entropy over all tokens is returned.
    """
    # Discretise each value into 4 bins (0‑3) and flatten
    tokens = []
    for vec in pheromone_vectors:
        if not vec:
            continue
        arr = np.array(vec)
        # 4‑bin quantisation
        bins = np.digitize(arr, np.histogram(arr, bins=4)[1][1:-1])
        tokens.extend([f"b{b}" for b in bins])
    return shannon_entropy(tokens)

def compute_burst_scores(pheromone_vectors: List[List[float]],
                         peak_force: float = 10.0,
                         steps: int = 7,
                         dt: float = 0.1,
                         m_head: float = 1.0,
                         drag_cd: float = 0.05) -> List[float]:
    """
    For each pheromone vector generate a pulse force series whose amplitude
    is proportional to the vector's mean magnitude, then integrate to obtain
    a burst admission score.
    """
    scores = []
    for vec in pheromone_vectors:
        if not vec:
            scores.append(0.0)
            continue
        mean_val = float(np.mean(vec))
        # Scale peak force by the mean pheromone intensity
        pf = pulse_force(peak_force * mean_val, steps)
        score = integrate_strike(pf, dt, m_head, drag_cd)
        scores.append(score)
    return scores

def hybrid_leader_election(node_ids: List[int],
                           pheromone_vectors: List[List[float]],
                           edges: List[Tuple[int, int, float]]) -> Dict[int, int]:
    """
    Perform leader election on a graph whose nodes carry pheromone data.
    1. Compute a global entropy H from all pheromone vectors.
    2. Derive edge priors πₑ = exp(-H) / Σₑ' exp(-H) and obtain weighted edge costs.
    3. Cluster nodes by Hamming distance of their hashes (threshold = 2).
    4. Within each cluster select the node with the highest product:
           burst_score * (average weighted edge cost incident to the node)
       as the cluster leader.
    Returns a mapping {cluster_id: leader_node_id}.
    """
    # ---- Step 1: entropy ----
    H = pheromone_entropy(pheromone_vectors)

    # ---- Step 2: weighted edge costs ----
    weighted_costs = entropy_weighted_edge_priors(edges, H)
    # Build adjacency map with weighted costs
    adj: Dict[int, List[Tuple[int, float]]] = {nid: [] for nid in node_ids}
    for (u, v, _), wcost in zip(edges, weighted_costs):
        adj[u].append((v, wcost))
        adj[v].append((u, wcost))

    # ---- Step 3: clustering by hash similarity ----
    hashes = [compute_phash(vec) for vec in pheromone_vectors]
    clusters: List[List[int]] = []
    visited = set()
    threshold = 2  # max Hamming distance to belong to same cluster
    for i, h in enumerate(hashes):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j, h2 in enumerate(hashes):
            if j in visited:
                continue
            if hamming_distance(h, h2) <= threshold:
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)

    # ---- Step 4: burst scores ----
    burst_scores = compute_burst_scores(pheromone_vectors)

    # Leader selection
    leaders: Dict[int, int] = {}
    for cid, cluster in enumerate(clusters):
        best_node = None
        best_metric = -math.inf
        for idx in cluster:
            nid = node_ids[idx]
            # average weighted incident cost
            inc_costs = [c for (_, c) in adj[nid]]
            avg_inc = sum(inc_costs) / (len(inc_costs) + 1e-9)
            metric = burst_scores[idx] * avg_inc
            if metric > best_metric:
                best_metric = metric
                best_node = nid
        if best_node is not None:
            leaders[cid] = best_node
    return leaders

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic graph
    node_ids = [0, 1, 2, 3]
    pheromone_vectors = [
        [0.2, 0.3, 0.25, 0.4],
        [0.8, 0.75, 0.9, 0.85],
        [0.21, 0.19, 0.22, 0.18],
        [0.78, 0.81, 0.77, 0.79],
    ]
    # Simple fully‑connected graph with base costs equal to Euclidean distance of mean pheromones
    edges = []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            mean_i = np.mean(pheromone_vectors[i])
            mean_j = np.mean(pheromone_vectors[j])
            cost = abs(mean_i - mean_j) + 0.1  # add small epsilon to avoid zero
            edges.append((node_ids[i], node_ids[j], cost))

    leaders = hybrid_leader_election(node_ids, pheromone_vectors, edges)
    print("Cluster leaders (cluster_id -> node_id):")
    for cid, nid in leaders.items():
        print(f"  {cid} -> {nid}")

    # Verify that the functions run without error
    assert isinstance(leaders, dict)
    sys.exit(0)