# DARWIN HAMMER — match 3799, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s3.py (gen3)
# born: 2026-05-29T23:51:51Z

import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A: Morphology utilities
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity (unitless, 0‑1)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness ratio (unitless, >0)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    """Physical proxy for the time needed to right the object."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


# ----------------------------------------------------------------------
# Parent‑B: Feature extraction & hygiene scoring
# ----------------------------------------------------------------------
H_MAX = math.log2(9)  # maximum Shannon entropy for a 9‑dimensional one‑hot vector


def extract_features(text: str) -> np.ndarray:
    """Light‑weight 9‑dimensional count vector from the input string."""
    tokens = [t.lower() for t in re.split(r"\W+", text) if t]
    counter: Dict[str, int] = {}
    for tok in tokens:
        if tok not in counter:
            counter[tok] = 0
        counter[tok] += 1
        if len(counter) >= 9:
            break
    vec = np.zeros(9, dtype=float)
    for i, count in enumerate(counter.values()):
        vec[i] = count
    return vec


def shannon_entropy(vec: np.ndarray) -> float:
    """Shannon entropy (base‑2) of a non‑negative vector."""
    if np.any(vec < 0):
        raise ValueError("entropy vector must be non‑negative")
    total = vec.sum()
    if total == 0:
        return 0.0
    p = vec / total
    mask = p > 0
    return -np.sum(p[mask] * np.log2(p[mask]))


def hybrid_hygiene_score(text: str) -> float:
    """
    Compute the hygiene score Sₕ = density · (1 + H / H_max).

    * density – token count per character, capped at 1.
    * H – Shannon entropy of the 9‑dimensional feature vector.
    """
    vec = extract_features(text)
    density = min(1.0, vec.sum() / max(len(text), 1))
    entropy = shannon_entropy(vec)
    return density * (1.0 + entropy / H_MAX)


# ----------------------------------------------------------------------
# Fusion utilities
# ----------------------------------------------------------------------
def compute_morphology_score(
    m: Morphology, α: float = 0.4, β: float = 0.3, γ: float = 0.3
) -> float:
    """
    Morphology term M = α·sphericity + β·flatness + γ·righting_time.
    Coefficients are assumed to sum to 1.
    """
    sph = sphericity_index(m.length, m.width, m.height)
    flat = flatness_index(m.length, m.width, m.height)
    rt = righting_time_index(m)
    return α * sph + β * flat + γ * rt


def _soft_normalise(value: float, scale: float = 10.0) -> float:
    """
    Soft‑cap normalisation using tanh.
    The scale parameter determines the point where saturation begins.
    """
    return math.tanh(value / scale)


def hybrid_node_score(
    text: str,
    morphology: Morphology,
    α: float = 0.4,
    β: float = 0.3,
    γ: float = 0.3,
    scale: float = 10.0,
) -> float:
    """
    Fuse hygiene and morphology into the hybrid prior Ŝ.

    Ŝ = Sₕ · (1 + tanh(M / scale))
    """
    S_h = hybrid_hygiene_score(text)
    M = compute_morphology_score(morphology, α, β, γ)
    M_norm = _soft_normalise(M, scale)
    return S_h * (1.0 + M_norm)


def bayes_marginal(prior: float, likelihood: float, false_pos: float) -> float:
    """
    Bayesian marginalisation:

        posterior = (prior * likelihood) /
                    (prior * likelihood + (1 - prior) * false_pos)

    The function clamps inputs to [0,1] and protects against division‑by‑zero.
    """
    eps = 1e-12
    prior = max(0.0, min(1.0, prior))
    likelihood = max(0.0, min(1.0, likelihood))
    false_pos = max(0.0, min(1.0, false_pos))

    numerator = prior * likelihood
    denominator = numerator + (1.0 - prior) * false_pos + eps
    return numerator / denominator


# ----------------------------------------------------------------------
# Minimum‑cost epistemic tree using hybrid priors
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    idx: int
    coord: Tuple[float, float]
    text: str
    morphology: Morphology


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def effective_edge_weight(
    node_i: Node,
    node_j: Node,
    flag: str,
    eps: float = 1e-9,
) -> float:
    """
    Compute the edge weight for Kruskal's algorithm.

    1. Physical distance `d`.
    2. Certainty factor `c` derived from the epistemic flag.
    3. Hybrid priors for both nodes → combined prior.
    4. Bayesian marginal → `marginal`.
    5. Weight = d * (1 - marginal)  (lower is better).
    """
    # 1. Physical cost
    d = euclidean_distance(node_i.coord, node_j.coord)

    # 2. Certainty factor c ∈ [0,1]
    flag_map = {
        "FACT": 0.95,
        "PROBABLE": 0.80,
        "POSSIBLE": 0.60,
        "SURE_MAYBE": 0.30,
        "BULLSHIT": 0.05,
    }
    c = flag_map.get(flag.upper(), 0.5)  # default moderate certainty

    # 3. Hybrid priors
    prior_i = hybrid_node_score(node_i.text, node_i.morphology)
    prior_j = hybrid_node_score(node_j.text, node_j.morphology)
    # Normalise to a probability‑like quantity; add eps to avoid 0/0.
    prior = prior_i / (prior_i + prior_j + eps)

    # 4. Bayesian components
    likelihood = 1.0 - c          # higher certainty → lower likelihood of error
    false_pos = c * 0.1           # small false‑positive rate scaled by certainty

    marginal = bayes_marginal(prior, likelihood, false_pos)

    # 5. Effective weight (lower → more attractive)
    return d * (1.0 - marginal) + eps


class _UnionFind:
    """Simple Union‑Find (Disjoint Set) data structure for Kruskal."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        xr = self.find(x)
        yr = self.find(y)
        if xr == yr:
            return False
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True


def build_epistemic_tree(
    nodes: List[Node],
    edge_flags: Dict[Tuple[int, int], str],
) -> List[Tuple[int, int, float]]:
    """
    Construct a minimum‑cost spanning tree over `nodes` using Kruskal's algorithm.
    `edge_flags` maps unordered node index pairs to an epistemic flag string.
    Missing entries default to the moderate certainty flag.

    Returns a list of edges (idx_i, idx_j, weight) describing the tree.
    """
    if not nodes:
        return []

    # 1. Generate all possible undirected edges with their weights.
    edges: List[Tuple[float, int, int]] = []
    n = len(nodes)
    idx_map = {node.idx: pos for pos, node in enumerate(nodes)}  # idx → position

    for i in range(n):
        for j in range(i + 1, n):
            node_i = nodes[i]
            node_j = nodes[j]
            # Retrieve flag; order‑independent key.
            key = (node_i.idx, node_j.idx) if (node_i.idx, node_j.idx) in edge_flags else (
                node_j.idx,
                node_i.idx,
            )
            flag = edge_flags.get(key, "POSSIBLE")
            w = effective_edge_weight(node_i, node_j, flag)
            edges.append((w, i, j))

    # 2. Sort edges by ascending weight.
    edges.sort(key=lambda e: e[0])

    # 3. Kruskal's MST selection.
    uf = _UnionFind(n)
    mst: List[Tuple[int, int, float]] = []
    for w, i, j in edges:
        if uf.union(i, j):
            mst.append((nodes[i].idx, nodes[j].idx, w))
            if len(mst) == n - 1:
                break

    return mst