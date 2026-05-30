# DARWIN HAMMER — match 749, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s3.py (gen3)
# parent_b: hybrid_rlct_grokking_hybrid_hybrid_hybrid_m52_s1.py (gen4)
# born: 2026-05-29T23:30:45Z

"""Hybrid Decision‑Hygiene & Multivector RLCT Tree

This module fuses the two parent algorithms:

* **Parent A** – extracts a 9‑dimensional feature vector from free‑text, computes a
  hygiene score `Sₕ` and builds an epistemic minimum‑cost spanning tree where
  edge weights are modulated by Bayesian marginalisation of an epistemic
  certainty flag.

* **Parent B** – provides a `Multivector` class and a Real Log‑Canonical
  Threshold (RLCT) update rule that uses the geometric (Clifford) product to
  represent a weight matrix `W`.

**Mathematical bridge**  
For every edge `(i, j)` we first obtain the Bayesian marginal `m(i,j)` exactly as
in Parent A.  We then construct a *multivector weight* `W_{ij}` whose scalar
components are the hybrid scores of the incident nodes.  The RLCT‑derived
scalar factor  


γ_{ij} = α · ‖W_{ij}‖²


(where `‖·‖` is the Euclidean norm of the multivector’s component vector and
`α` is a tunable constant) multiplies the Bayesian‑adjusted physical distance.
Thus the final effective edge weight becomes


w_{ij} = d(i,j) · (1 – m(i,j)) · γ_{ij} + ε .


Edges that connect high‑scoring, well‑documented nodes and that carry strong
epistemic certainty become cheap, while the multivector‑derived factor injects
the geometric‑algebraic richness of Parent B.

The module exposes three core functions that demonstrate the hybrid operation:
`extract_features`, `hybrid_hygiene_score`, and `build_epistemic_tree`.  The
`Multivector` class from Parent B is reused verbatim.

"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent B – Multivector definition (unchanged)
# ----------------------------------------------------------------------


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: Dict[Tuple[int, ...], float], n: int):
        # store only non‑zero components
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        """Return a new Multivector keeping only grade‑k components."""
        return Multivector({blade: val for blade, val in self.components.items() if len(blade) == k}, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product of two Multivectors."""
        result = Multivector({}, self.n)
        for blade_a, value_a in self.components.items():
            for blade_b, value_b in other.components.items():
                combined, sign = _multiply_blades(blade_a, blade_b)
                result.components[combined] = result.components.get(combined, 0.0) + sign * value_a * value_b
        return result

    def norm(self) -> float:
        """Euclidean norm of the component vector."""
        return math.sqrt(sum(v * v for v in self.components.values()))


def _multiply_blades(blade_a: Tuple[int, ...], blade_b: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """Geometric‑product helper: concatenate index lists and sort with sign."""
    # concatenate
    indices = list(blade_a) + list(blade_b)
    # bubble‑sort while tracking sign flips
    sign = 1
    n = len(indices)
    for i in range(n):
        for j in range(n - 1 - i):
            if indices[j] > indices[j + 1]:
                indices[j], indices[j + 1] = indices[j + 1], indices[j]
                sign *= -1
    # remove duplicate indices (square to +1 in Euclidean signature)
    cleaned = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == indices[i + 1]:
            # pair cancels out
            i += 2
        else:
            cleaned.append(indices[i])
            i += 1
    return tuple(cleaned), sign


# ----------------------------------------------------------------------
# Parent A – Feature extraction & hygiene scoring (concise re‑implementation)
# ----------------------------------------------------------------------


_KEYWORDS = [
    "privacy",
    "security",
    "compliance",
    "integrity",
    "availability",
    "audit",
    "risk",
    "governance",
    "policy",
]  # exactly 9 keywords → H_max = log2 9


def extract_features(text: str) -> np.ndarray:
    """
    Count occurrences of the nine predefined keywords (case‑insensitive) and
    return a 9‑dimensional integer vector.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = [tokens.count(k) for k in _KEYWORDS]
    return np.array(counts, dtype=int)


def hybrid_hygiene_score(v: np.ndarray) -> float:
    """
    Compute the hybrid hygiene score Sₕ for a feature count vector `v`.

    * `s` – average count (a proxy for raw hygiene).
    * `H` – Shannon entropy of the normalized count distribution.
    * `Sₕ = s * (1 + H / H_max)`,  where `H_max = log₂ 9`.
    """
    eps = 1e-12
    total = float(v.sum())
    s = total / v.size
    if total < eps:
        H = 0.0
    else:
        p = v / total
        H = -np.sum(p * np.log2(p + eps))
    H_max = math.log2(9)
    return s * (1.0 + H / H_max)


# ----------------------------------------------------------------------
# Bayesian marginal helper (from Parent A)
# ----------------------------------------------------------------------


def bayes_marginal(prior: float, likelihood: float, false_pos: float) -> float:
    """
    Simple Bayesian update:

        posterior = (likelihood * prior) /
                    (likelihood * prior + false_pos * (1 - prior) + ε)

    Returns a value in [0,1].
    """
    eps = 1e-12
    numerator = likelihood * prior
    denominator = numerator + false_pos * (1.0 - prior) + eps
    return numerator / denominator


# ----------------------------------------------------------------------
# RLCT‑derived multivector scaling (bridge to Parent B)
# ----------------------------------------------------------------------


def multivector_rlct_factor(score_i: float, score_j: float, alpha: float = 0.5) -> float:
    """
    Build a 2‑blade multivector from the two node scores and return the
    RLCT‑derived scaling factor γ = α · ‖W‖².

    The multivector has components:
        e1 → score_i
        e2 → score_j
    """
    components = { (1,): score_i, (2,): score_j }
    W = Multivector(components, n=2)
    return alpha * (W.norm() ** 2)


# ----------------------------------------------------------------------
# Core hybrid tree builder
# ----------------------------------------------------------------------


class Node:
    """Simple container for a graph node."""
    def __init__(self, node_id: int, text: str, coord: Tuple[float, float]):
        self.id = node_id
        self.text = text
        self.coord = np.array(coord, dtype=float)
        self.features = extract_features(text)
        self.score = hybrid_hygiene_score(self.features)


class Edge:
    """Undirected edge with an epistemic certainty flag c ∈ [0,1]."""
    def __init__(self, i: int, j: int, certainty: float):
        self.i = i
        self.j = j
        self.c = float(certainty)  # epistemic certainty


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def effective_weight(node_i: Node, node_j: Node, edge: Edge, eps: float = 1e-9) -> float:
    """
    Compute the hybrid effective weight for edge (i,j) using:

        prior   = Sₕ(i) / (Sₕ(i) + Sₕ(j) + ε)
        lik     = 1 - c
        fp      = c * 0.1
        m       = bayes_marginal(prior, lik, fp)
        d       = Euclidean distance between node coordinates
        γ       = multivector_rlct_factor(Sₕ(i), Sₕ(j))

        weight  = d * (1 - m) * γ + ε
    """
    prior = node_i.score / (node_i.score + node_j.score + eps)
    lik = 1.0 - edge.c
    fp = edge.c * 0.1
    m = bayes_marginal(prior, lik, fp)
    d = euclidean_distance(node_i.coord, node_j.coord)
    gamma = multivector_rlct_factor(node_i.score, node_j.score)
    return d * (1.0 - m) * gamma + eps


def build_epistemic_tree(nodes: List[Node], edges: List[Edge]) -> List[Tuple[int, int, float]]:
    """
    Return a list of edges sorted by their effective weight (ascending).
    The list entries are tuples (i, j, weight).
    """
    weight_list = []
    node_dict = {node.id: node for node in nodes}
    for e in edges:
        n_i = node_dict[e.i]
        n_j = node_dict[e.j]
        w = effective_weight(n_i, n_j, e)
        weight_list.append((e.i, e.j, w))
    # sort by weight – mimics extracting a minimum‑cost spanning tree
    weight_list.sort(key=lambda tup: tup[2])
    return weight_list


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a tiny synthetic graph of three nodes
    random.seed(42)
    np.random.seed(42)

    sample_texts = [
        "Privacy and security are core to our policy and governance.",
        "Risk, audit, and compliance drive the integrity of the system.",
        "Availability and governance ensure continuous service and risk mitigation."
    ]

    nodes = [
        Node(node_id=0, text=sample_texts[0], coord=(0.0, 0.0)),
        Node(node_id=1, text=sample_texts[1], coord=(1.5, 0.5)),
        Node(node_id=2, text=sample_texts[2], coord=(0.8, 1.2)),
    ]

    # Fully connected undirected graph with random epistemic certainty flags
    edges = [
        Edge(0, 1, certainty=random.uniform(0.6, 1.0)),
        Edge(0, 2, certainty=random.uniform(0.6, 1.0)),
        Edge(1, 2, certainty=random.uniform(0.6, 1.0)),
    ]

    sorted_edges = build_epistemic_tree(nodes, edges)

    print("Hybrid effective edge weights (ascending):")
    for i, j, w in sorted_edges:
        print(f"  ({i} ↔ {j}) : {w:.6f}")

    # Demonstrate that node scores are sensible
    print("\nNode hybrid hygiene scores:")
    for n in nodes:
        print(f"  Node {n.id}: score = {n.score:.4f}")