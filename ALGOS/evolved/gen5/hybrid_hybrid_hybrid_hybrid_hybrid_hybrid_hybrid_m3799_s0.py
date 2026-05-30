# DARWIN HAMMER — match 3799, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s3.py (gen3)
# born: 2026-05-29T23:51:51Z

"""Hybrid Morphology‑Decision‑Epistemic System
================================================

Parents
-------
* **Parent A** – provides geometric morphology descriptors (`sphericity_index`,
  `flatness_index`, `righting_time_index`) and a `Morphology` data class.
* **Parent B** – extracts a 9‑dimensional feature vector from free‑text, computes a
  hygiene score `Sₕ`, and builds a minimum‑cost spanning tree where edge weights
  are modulated by epistemic certainty flags and Bayesian marginalisation.

Mathematical Bridge
-------------------
The bridge is a *node‑centric* score that fuses the morphology‑derived physical
indices with the decision‑hygiene score.  For a candidate node *n* we define


M(n) = α·sphericity(n) + β·flatness(n) + γ·righting_time(n)
Sₕ(n) = hygiene_score(text_n) · (1 + H(text_n)/H_max)
Ŝ(n) = Sₕ(n) · (1 + M(n)/M_max)                (1)


`Ŝ(n)` is the **hybrid node score** used as a Bayesian prior in the edge‑weight
formula of Parent B.  By scaling the prior with a morphology term we let the
physical shape of a candidate influence the epistemic tree, achieving a true
fusion of the two topologies.

The module implements:
* `compute_morphology_score` – evaluates `M(n)` from a `Morphology`.
* `hybrid_node_score` – combines the morphology score with the hygiene score.
* `build_epistemic_tree` – Kruskal’s MST using the hybrid prior in the Bayesian
  marginal weight calculation.

All functions are pure NumPy / Python and require no external scientific stack."""

import math
import random
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
    """Very light‑weight 9‑dimensional count vector from the input string."""
    # Tokenise on non‑alphanumerics, lower‑case, and count the first 9 distinct tokens.
    tokens = [t.lower() for t in re.split(r"\W+", text) if t]
    counter = {}
    for tok in tokens:
        counter[tok] = counter.get(tok, 0) + 1
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
    # avoid log(0) by masking zero entries
    mask = p > 0
    return -np.sum(p[mask] * np.log2(p[mask]))


def hybrid_hygiene_score(text: str) -> float:
    """
    Compute the hybrid hygiene score Sₕ as described in Parent B:

        Sₕ = s · (1 + H / H_max)

    where `s` is a simple token‑density score and `H` the Shannon entropy.
    """
    vec = extract_features(text)
    # token‑density: total tokens per 100 characters (capped at 1)
    density = min(1.0, vec.sum() / max(len(text), 1))
    entropy = shannon_entropy(vec)
    return density * (1.0 + entropy / H_MAX)


# ----------------------------------------------------------------------
# Fusion utilities
# ----------------------------------------------------------------------
def compute_morphology_score(m: Morphology, α: float = 0.4, β: float = 0.3, γ: float = 0.3) -> float:
    """
    Compute the morphology term M(n) = α·sphericity + β·flatness + γ·righting_time.
    The three coefficients sum to 1 and can be tuned.
    """
    sph = sphericity_index(m.length, m.width, m.height)
    flat = flatness_index(m.length, m.width, m.height)
    rt = righting_time_index(m)
    return α * sph + β * flat + γ * rt


def hybrid_node_score(
    text: str, morphology: Morphology, α: float = 0.4, β: float = 0.3, γ: float = 0.3
) -> float:
    """
    Combine the decision‑hygiene score with the morphology term as in equation (1).

    Returns the hybrid prior `Ŝ(n)` used later in the Bayesian marginal.
    """
    S_h = hybrid_hygiene_score(text)
    M = compute_morphology_score(morphology, α, β, γ)
    # Normalise M to [0,1] using a soft‑cap (empirically chosen)
    M_norm = math.tanh(M / 10.0)  # 10 is a scale factor; adjust for domain
    return S_h * (1.0 + M_norm)


def bayes_marginal(prior: float, likelihood: float, false_pos: float) -> float:
    """
    Simple Bayesian marginalisation:

        posterior = (prior * likelihood) / (prior * likelihood + (1-prior) * false_pos)

    The function is robust to division‑by‑zero via an epsilon.
    """
    eps = 1e-12
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
    Compute the edge weight according to Parent B, but replace the prior with the
    hybrid node score defined above.
    """
    # 1. Physical cost
    d = euclidean_distance(node_i.coord, node_j.coord)

    # 2. Epistemic certainty factor c(e) ∈ [0,1] derived from flag
    # Map the predefined flags to a certainty level.
    flag_map = {
        "FACT": 0.95,
        "PROBABLE": 0.80,
        "POSSIBLE": 0.60,
        "SURE_MAYBE": 0.30,
        "BULLSHIT": 0.05,
    }
    c = flag_map.get(flag.upper(), 0.5)  # default moderate certainty

    # 3. Bayesian components
    prior_i = hybrid_node_score(node_i.text, node_i.morphology)
    prior_j = hybrid_node_score(node_j.text, node_j.morphology)
    prior = prior_i / (prior_i + prior_j + eps)

    likelihood = 1.0 - c
    false_pos = c * 0.1

    marginal = bayes_marginal(prior, likelihood, false_pos)

    # 4. Effective weight (lower weight → more attractive)
    weight = d * (1.0 - marginal) + eps
    return weight


def build_epistemic_tree(
    nodes: List[Node],
    edge_flags: Dict[Tuple[int, int], str],
) -> List[Tuple[int, int, float]]:
    """
    Kruskal’s algorithm producing a minimum‑cost spanning tree.
    Returns a list of edges (i, j, weight).
    """
    # Union‑Find data structure
    parent = {node.idx: node.idx for node in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Build all candidate edges with their effective weights
    edge_list: List[Tuple[int, int, float]] = []
    n = len(nodes)
    idx_to_node = {node.idx: node for node in nodes}
    for i in range(n):
        for j in range(i + 1, n):
            flag = edge_flags.get((nodes[i].idx, nodes[j].idx), "PROBABLE")
            w = effective_edge_weight(nodes[i], nodes[j], flag)
            edge_list.append((nodes[i].idx, nodes[j].idx, w))

    # Sort edges by weight
    edge_list.sort(key=lambda e: e[2])

    mst: List[Tuple[int, int, float]] = []
    for i, j, w in edge_list:
        if find(i) != find(j):
            union(i, j)
            mst.append((i, j, w))
            if len(mst) == n - 1:
                break
    return mst


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic dataset of 4 candidates
    random.seed(42)
    np.random.seed(42)

    def random_morph() -> Morphology:
        return Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(1.0, 5.0),
        )

    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Data-driven insights reveal hidden patterns in the dataset.",
        "A robust algorithm must handle noise and uncertainty gracefully.",
        "Optimisation of free energy leads to better convergence.",
    ]

    nodes = [
        Node(
            idx=i,
            coord=(random.uniform(0, 10), random.uniform(0, 10)),
            text=sample_texts[i],
            morphology=random_morph(),
        )
        for i in range(len(sample_texts))
    ]

    # Randomly assign epistemic flags to each possible edge
    flags = {}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            flags[(nodes[i].idx, nodes[j].idx)] = random.choice(EPISTEMIC_FLAGS)

    mst = build_epistemic_tree(nodes, flags)

    print("Minimum‑cost epistemic tree (node_i, node_j, weight):")
    for edge in mst:
        print(edge)