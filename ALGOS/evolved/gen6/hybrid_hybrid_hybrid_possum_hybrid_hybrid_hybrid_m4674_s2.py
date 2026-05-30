# DARWIN HAMMER — match 4674, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_possum_filter_hybrid_caputo_fracti_m1220_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m1191_s0.py (gen5)
# born: 2026-05-29T23:57:20Z

"""Hybrid Spatial‑Aware Fractional Tree with Fisher‑Weighted RBF & Perceptual Hashing.

Parents:
* hybrid_hybrid_possum_filter_hybrid_caputo_fracti_m1220_s0.py (spatial‑aware
  Caputo fractional cost)
* hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m1191_s0.py (Fisher‑modulated
  RBF similarity + perceptual hashing)

Mathematical bridge:
The edge cost of the minimum‑cost spanning tree is defined as  

    w_ij =  κ_ij · φ_ij · (1 + H_ij)

where  

* κ_ij  – Caputo fractional kernel that incorporates the haversine distance
  d_ij and a privacy‑risk term r_i (from the Possum filter).  
* φ_ij  – Radial‑basis‑function weight  exp(‑ε·d_ij²) whose scale ε is
  modulated by the Fisher information score F_i of node i, thus coupling the
  fractional memory term with the Fisher‑weighted similarity of the
  perceptual model.  
* H_ij – normalized Hamming distance between perceptual hashes of the two
  nodes, providing a discrete similarity component.

The three functions below realise this fused cost and a Kruskal‑style
construction of the fractional minimum‑cost tree.  The implementation stays
within the allowed standard‑library and NumPy stack."""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Set

import numpy as np

# ----------------------------------------------------------------------
# Basic data structures
# ----------------------------------------------------------------------
Vector = List[float]
Node = str


@dataclass(frozen=True)
class Entity:
    """A data point with spatial coordinates, a categorical label and a
    numeric feature vector used for Fisher and perceptual hashing."""
    id: str
    lat: float
    lon: float
    category: str
    features: Vector
    # optional fields for extensions
    score: float = 0.0
    address_signature: str = ""


# ----------------------------------------------------------------------
# Parent‑A utilities (spatial & fractional)
# ----------------------------------------------------------------------
def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(
        dlon / 2
    ) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def reconstruction_risk_score(entity: Entity, total_records: int) -> float:
    """Placeholder privacy‑risk: inverse proportional to total records."""
    return 1.0 / max(1, total_records)


def caputo_fractional_weight(
    distance: float, risk: float, alpha: float = 0.8
) -> float:
    """Caputo‑type kernel mixing spatial distance and privacy risk.

    κ = risk * (distance)^{α‑1} / Γ(α)
    The Gamma function is approximated by math.gamma (available in std lib)."""
    if distance <= 0:
        distance = 1e-9
    return risk * (distance ** (alpha - 1)) / math.gamma(alpha)


# ----------------------------------------------------------------------
# Parent‑B utilities (Fisher, RBF, perceptual hashing)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """64‑bit perceptual hash based on mean threshold."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def fisher_information_scores(entities: List[Entity]) -> Dict[Node, float]:
    """Estimate a scalar Fisher score per entity as the Mahalanobis‑like
    distance of its feature vector from the global mean, using per‑dimension
    variance as the covariance approximation."""
    if not entities:
        return {}
    # Stack feature vectors
    mat = np.vstack([e.features for e in entities])
    mean = np.mean(mat, axis=0)
    var = np.var(mat, axis=0, ddof=1)
    # Avoid division by zero
    var[var == 0] = 1e-9
    scores = {}
    for e in entities:
        diff = np.array(e.features) - mean
        score = float(np.sum((diff ** 2) / var))
        scores[e.id] = score
    return scores


def fisher_modulated_rbf(
    distance: float, fisher_score: float, base_epsilon: float = 0.5
) -> float:
    """RBF where epsilon is scaled by the Fisher information of the source node."""
    epsilon = base_epsilon * (1.0 + fisher_score)
    return gaussian(distance, epsilon)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_edge_weight(
    e1: Entity,
    e2: Entity,
    total_records: int,
    alpha: float = 0.8,
    base_epsilon: float = 0.5,
) -> float:
    """Compute the fused edge weight w_ij = κ_ij * φ_ij * (1 + H_ij_norm)."""
    # Spatial part
    d_geo = haversine_m((e1.lat, e1.lon), (e2.lat, e2.lon))

    # Privacy risk (same for both ends; we use the larger risk)
    risk1 = reconstruction_risk_score(e1, total_records)
    risk2 = reconstruction_risk_score(e2, total_records)
    risk = max(risk1, risk2)

    # Caputo fractional kernel
    kappa = caputo_fractional_weight(d_geo, risk, alpha)

    # Fisher‑modulated RBF (use Fisher score of the source node)
    fisher_scores = fisher_information_scores([e1, e2])
    phi = fisher_modulated_rbf(d_geo, fisher_scores[e1.id], base_epsilon)

    # Perceptual hash Hamming term (normalised to [0,1])
    ph1 = compute_phash(e1.features)
    ph2 = compute_phash(e2.features)
    hamming = hamming_distance(ph1, ph2)
    h_norm = hamming / 64.0  # 64‑bit hash

    return kappa * phi * (1.0 + h_norm)


class UnionFind:
    """Simple Union‑Find (Disjoint Set) structure for Kruskal."""
    def __init__(self):
        self.parent: Dict[Node, Node] = {}
        self.rank: Dict[Node, int] = {}

    def make_set(self, x: Node):
        self.parent[x] = x
        self.rank[x] = 0

    def find(self, x: Node) -> Node:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: Node, y: Node) -> bool:
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return False
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        elif self.rank[xroot] > self.rank[yroot]:
            self.parent[yroot] = xroot
        else:
            self.parent[yroot] = xroot
            self.rank[xroot] += 1
        return True


def hybrid_fractional_minimum_cost_tree(
    entities: List[Entity],
    total_records: int,
    alpha: float = 0.8,
    base_epsilon: float = 0.5,
) -> List[Tuple[Node, Node, float]]:
    """
    Build a minimum‑cost spanning tree where each edge weight is the hybrid
    cost defined in `hybrid_edge_weight`. Returns a list of edges
    (src_id, dst_id, weight) forming the tree.
    """
    if len(entities) < 2:
        return []

    # Prepare Union‑Find
    uf = UnionFind()
    for e in entities:
        uf.make_set(e.id)

    # Compute all pairwise hybrid weights
    edges: List[Tuple[float, Node, Node]] = []
    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1 :]:
            w = hybrid_edge_weight(e1, e2, total_records, alpha, base_epsilon)
            edges.append((w, e1.id, e2.id))

    # Kruskal's algorithm (ascending order)
    edges.sort(key=lambda x: x[0])
    tree: List[Tuple[Node, Node, float]] = []
    for w, u, v in edges:
        if uf.union(u, v):
            tree.append((u, v, w))
            if len(tree) == len(entities) - 1:
                break
    return tree


def fractional_ssm_step(
    state: np.ndarray,
    history: List[np.ndarray],
    alpha: float = 0.8,
) -> np.ndarray:
    """
    Fractional state‑space update using Caputo‑type memory.
    state_{t+1} = state_t + Σ_{k=0}^{t} w_k * (history_k - state_t)
    where w_k = (t‑k+1)^{α‑1} / Γ(α).
    """
    t = len(history)
    if t == 0:
        return state
    weights = np.array(
        [(t - k + 1) ** (alpha - 1) / math.gamma(alpha) for k in range(t)]
    )
    diffs = np.stack(history) - state
    update = np.sum(weights[:, None] * diffs, axis=0)
    return state + update


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Generate synthetic entities
    N = 12
    entities: List[Entity] = []
    for i in range(N):
        lat = random.uniform(-90, 90)
        lon = random.uniform(-180, 180)
        category = random.choice(["A", "B", "C"])
        features = np.random.rand(128).tolist()  # 128‑dim feature vector
        entities.append(
            Entity(
                id=f"node_{i}",
                lat=lat,
                lon=lon,
                category=category,
                features=features,
            )
        )

    total_records = 1000
    tree = hybrid_fractional_minimum_cost_tree(
        entities, total_records, alpha=0.85, base_epsilon=0.3
    )
    print("Hybrid Minimum‑Cost Tree edges (src, dst, weight):")
    for src, dst, w in tree:
        print(f"{src} – {dst} : {w:.6e}")

    # Demonstrate fractional state update
    init_state = np.zeros(10)
    hist = [np.random.rand(10) for _ in range(5)]
    new_state = fractional_ssm_step(init_state, hist, alpha=0.9)
    print("\nState after fractional SSM step (first 5 values):")
    print(new_state[:5])