# DARWIN HAMMER — match 14, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s2.py (gen2)
# parent_b: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s0.py (gen2)
# born: 2026-05-29T23:25:17Z

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
# Feature extraction 
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
    ("generic", re.compile(r"\b\w{7,}\b", re.I)),
]

# ----------------------------------------------------------------------
# Epistemic certainty flags 
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
FLAG_CERTAINTY: Dict[str, float] = {
    "FACT": 0.95,
    "PROBABLE": 0.75,
    "POSSIBLE": 0.50,
    "SURE_MAYBE": 0.30,
    "BULLSHIT": 0.05,
}

# ----------------------------------------------------------------------
# Helper mathematics 
# ----------------------------------------------------------------------
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


# ----------------------------------------------------------------------
# Core Hybrid Functions
# ----------------------------------------------------------------------
def extract_features(text: str) -> np.ndarray:
    counts = []
    for _, pattern in FEATURE_REGEXES:
        matches = pattern.findall(text)
        counts.append(len(matches))
    return np.array(counts, dtype=int)


def hybrid_hygiene_score(
    v: np.ndarray,
    w_pos: np.ndarray = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    w_neg: np.ndarray = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
) -> float:
    if v.shape != (9,):
        raise ValueError("All vectors must be of length 9")
    s = float(w_pos @ v - w_neg @ v)

    total = v.sum()
    if total == 0:
        H = 0.0
    else:
        p = v / total
        mask = p > 0
        H = -float(np.sum(p[mask] * np.log2(p[mask])))

    H_max = math.log2(9)
    hybrid = s * (1.0 + H / H_max) if H_max > 0 else s
    return hybrid


def edge_effective_weight(
    node_i: str,
    node_j: str,
    coords: Dict[str, Tuple[float, float]],
    hybrid_scores: Dict[str, float],
    flag: str,
    epsilon: float = 1e-9,
) -> float:
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
    hybrid_scores = {node: hybrid_hygiene_score(extract_features(text)) for node, text in texts.items()}

    edges = []
    for node_i in nodes:
        for node_j in nodes:
            if node_i < node_j:
                flag = edge_flags.get((node_i, node_j), "BULLSHIT")
                weight = edge_effective_weight(node_i, node_j, nodes, hybrid_scores, flag)
                edges.append((node_i, node_j, weight))

    edges.sort(key=lambda x: x[2])
    parent = {}
    rank = {}

    def make_set(node):
        parent[node] = node
        rank[node] = 0

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)

        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
                if rank[root1] == rank[root2]:
                    rank[root2] += 1

    for node in nodes:
        make_set(node)

    mst_edges = []
    mst_weight = 0.0
    for edge in edges:
        node_i, node_j, weight = edge
        if find(node_i) != find(node_j):
            union(node_i, node_j)
            mst_edges.append((node_i, node_j))
            mst_weight += weight

    return mst_edges, mst_weight