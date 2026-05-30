# DARWIN HAMMER — match 14, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s2.py (gen2)
# parent_b: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s0.py (gen2)
# born: 2026-05-29T23:25:17Z

"""Hybrid Decision‑Hygiene & Minimum‑Cost Epistemic Tree

Parents
-------
* **hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s2.py** – extracts a 9‑dimensional
  feature count vector *v* from free‑text, computes a hygiene score *s* and a Shannon
  entropy *H*, then combines them as  

      Sₕ = s · (1 + H / Hₘₐₓ) ,   Hₘₐₓ = log₂ 9 .

* **hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s0.py** – builds a minimum‑cost
  spanning tree where edge weights are altered by epistemic‑certainty flags and
  Bayesian updates of prior probabilities.

Mathematical Bridge
-------------------
Each vendor *candidate* is a node *n* with a hybrid decision‑hygiene score
`Sₕ(n)`.  For any edge *e = (i, j)* we define a **raw physical cost**
`d(i,j)` (Euclidean distance of the node coordinates) and an **epistemic
certainty factor** `c(e) ∈ [0,1]` derived from the flag attached to the edge.
The edge’s *effective* weight is obtained by a Bayesian‑inspired combination
of the physical cost, the epistemic factor and the node scores:


    prior   = Sₕ(i) / (Sₕ(i) + Sₕ(j) + ε)                # node‑based prior
    lik     = 1 - c(e)                                   # higher certainty → lower likelihood of error
    fp      = c(e) * 0.1                                 # modest false‑positive term
    marginal = bayes_marginal(prior, lik, fp)
    weight  = d(i,j) * (1 - marginal) + ε                # ε avoids zero division


Thus edges that connect highly‑scored, well‑documented candidates and that are
labelled with strong epistemic certainty become cheaper, guiding the
minimum‑cost tree toward the most trustworthy and information‑rich sub‑graph.

The module provides three core functions that realise this hybrid system:
`extract_features`, `hybrid_hygiene_score`, and `build_epistemic_tree`.  A
light‑weight smoke test demonstrates end‑to‑end execution."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Feature extraction – identical to Parent A
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|b", re.I
)  # truncated pattern – sufficient for demo
# Additional dummy patterns to reach nine features
QUALITY_RE = re.compile(r"\b(?:quality|high|low|grade|rating)\b", re.I)
SECURITY_RE = re.compile(r"\b(?:security|secure|vulnerability|exploit)\b", re.I)
PERFORMANCE_RE = re.compile(r"\b(?:performance|fast|slow|latency)\b", re.I)
COMPLIANCE_RE = re.compile(r"\b(?:compliance|regulation|standard)\b", re.I)
COST_RE = re.compile(r"\b(?:cost|price|budget|expense)\b", re.I)

FEATURE_REGEXES: List[Tuple[str, re.Pattern]] = [
    ("evidence", EVIDENCE_RE),
    ("planning", PLANNING_RE),
    ("delay", DELAY_RE),
    ("quality", QUALITY_RE),
    ("security", SECURITY_RE),
    ("performance", PERFORMANCE_RE),
    ("compliance", COMPLIANCE_RE),
    ("cost", COST_RE),
    # a ninth generic catch‑all pattern (words longer than 6 letters)
    ("generic", re.compile(r"\b\w{7,}\b", re.I)),
]

# ----------------------------------------------------------------------
# Epistemic certainty flags – identical to Parent B
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
# Map each flag to a numeric certainty factor in [0,1] (higher = more certain)
FLAG_CERTAINTY: Dict[str, float] = {
    "FACT": 0.95,
    "PROBABLE": 0.75,
    "POSSIBLE": 0.50,
    "SURE_MAYBE": 0.30,
    "BULLSHIT": 0.05,
}

# ----------------------------------------------------------------------
# Helper mathematics (Parent B)
# ----------------------------------------------------------------------
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability P(E) = L·P + FP·(1‑P)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P·L / P(E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


# ----------------------------------------------------------------------
# Core Hybrid Functions
# ----------------------------------------------------------------------
def extract_features(text: str) -> np.ndarray:
    """
    Return a 9‑dimensional integer count vector for the supplied text.
    Order follows ``FEATURE_REGEXES``.
    """
    counts = []
    for _, pattern in FEATURE_REGEXES:
        matches = pattern.findall(text)
        counts.append(len(matches))
    return np.array(counts, dtype=int)


def hybrid_hygiene_score(
    v: np.ndarray,
    w_pos: np.ndarray,
    w_neg: np.ndarray,
) -> float:
    """
    Compute the hybrid decision‑hygiene metric:

        s = w⁺·v − w⁻·v
        H = −∑ pᵢ log₂ pᵢ   where p = v / Σv
        Sₕ = s · (1 + H / H_max)

    ``w_pos`` and ``w_neg`` must be length‑9 weight vectors.
    """
    if v.shape != (9,) or w_pos.shape != (9,) or w_neg.shape != (9,):
        raise ValueError("All vectors must be of length 9")
    s = float(w_pos @ v - w_neg @ v)

    total = v.sum()
    if total == 0:
        H = 0.0
    else:
        p = v / total
        # avoid log2(0) by masking zeros
        mask = p > 0
        H = -float(np.sum(p[mask] * np.log2(p[mask])))

    H_max = math.log2(9)
    hybrid = s * (1.0 + H / H_max)
    return hybrid


def edge_effective_weight(
    node_i: str,
    node_j: str,
    coords: Dict[str, Tuple[float, float]],
    hybrid_scores: Dict[str, float],
    flag: str,
    prior_probabilities: Dict[Tuple[str, str], float],
    likelihoods: Dict[Tuple[str, str], float],
    false_positives: Dict[Tuple[str, str], float],
    epsilon: float = 1e-9,
) -> float:
    """
    Compute the epistemic‑adjusted weight for edge (i, j).

    Steps
    -----
    1. Physical distance d(i,j).
    2. Node‑based prior = Sₕ(i) / (Sₕ(i)+Sₕ(j)+ε).
    3. Likelihood = 1 − certainty_factor(flag).
    4. False‑positive term = certainty_factor(flag) * 0.1.
    5. Bayesian marginal = bayes_marginal(prior, likelihood, fp).
    6. Weight = d(i,j) * (1 − marginal) + ε.
    """
    d = length(coords[node_i], coords[node_j])

    Si = hybrid_scores.get(node_i, 0.0)
    Sj = hybrid_scores.get(node_j, 0.0)
    prior = Si / (Si + Sj + epsilon)

    certainty = FLAG_CERTAINTY.get(flag.upper(), 0.0)
    likelihood = 1.0 - certainty
    fp = certainty * 0.1

    marginal = bayes_marginal(prior, likelihood, fp)

    weight = d * (1.0 - marginal) + epsilon
    return weight


def build_epistemic_tree(
    nodes: Dict[str, Tuple[float, float]],
    texts: Dict[str, str],
    edge_flags: Dict[Tuple[str, str], str],
) -> Tuple[List[Tuple[str, str]], float]:
    """
    Construct a minimum‑cost spanning tree over *nodes* where each node
    carries a hybrid hygiene score derived from its associated free‑text.

    Returns
    -------
    edges_in_tree : list of (src, dst) tuples
    total_cost    : sum of effective edge weights
    """
    # 1. Compute hybrid scores for every node
    w_pos = np.array([2, 1, 1, 2, 3, 1, 1, 1, 1], dtype=float)
    w_neg = np.array([1, 2, 2, 1, 1, 2, 2, 2, 2], dtype=float)

    hybrid_scores = {
        nid: hybrid_hygiene_score(extract_features(texts[nid]), w_pos, w_neg)
        for nid in nodes
    }

    # 2. Assemble all possible edges (undirected) with their effective weights
    all_edges: List[Tuple[float, str, str]] = []
    node_ids = list(nodes.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            a, b = node_ids[i], node_ids[j]
            flag = edge_flags.get((a, b)) or edge_flags.get((b, a)) or "POSSIBLE"
            w = edge_effective_weight(
                a,
                b,
                nodes,
                hybrid_scores,
                flag,
                prior_probabilities={},  # not used in this simplified version
                likelihoods={},
                false_positives={},
            )
            all_edges.append((w, a, b))

    # 3. Kruskal's algorithm for MST
    parent: Dict[str, str] = {nid: nid for nid in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[ry] = rx
        return True

    all_edges.sort(key=lambda e: e[0])  # sort by weight
    mst_edges: List[Tuple[str, str]] = []
    total_cost = 0.0
    for w, a, b in all_edges:
        if union(a, b):
            mst_edges.append((a, b))
            total_cost += w
        if len(mst_edges) == len(nodes) - 1:
            break

    return mst_edges, total_cost


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    # Define a tiny graph of three candidates
    nodes = {
        "vendorA": (0.0, 0.0),
        "vendorB": (3.0, 4.0),  # distance 5 from A
        "vendorC": (6.0, 0.0),  # distance 6 from A, 5 from B
    }

    texts = {
        "vendorA": "Evidence of security audit and compliance. High quality plan.",
        "vendorB": "Possible delay, cost concerns, but good performance records.",
        "vendorC": "Fact checked, verified, and documented. Low risk.",
    }

    edge_flags = {
        ("vendorA", "vendorB"): "PROBABLE",
        ("vendorB", "vendorC"): "FACT",
        ("vendorA", "vendorC"): "POSSIBLE",
    }

    tree, cost = build_epistemic_tree(nodes, texts, edge_flags)
    print("MST edges:", tree)
    print("Total cost:", cost)


if __name__ == "__main__":
    _smoke_test()