# DARWIN HAMMER — match 1333, survivor 0
# gen: 5
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_indy_learning_m816_s1.py (gen4)
# born: 2026-05-29T23:35:27Z

"""Hybrid algorithm combining minimum-cost tree scoring from minimum_cost_tree.py and Bayesian evidence update from bayes_update.py with the ternary lens audit and text-vector fusion from hybrid_hybrid_ternar_hybrid_indy_learning_m816_s1.py.

The bridge between the two structures is the notion of uncertainty in the tree edges and nodes, which is combined with the ternary lens audit and text-vector fusion. By assigning prior probabilities to the edges and nodes, we can update these probabilities based on new evidence using the Bayesian update rule. The ternary lens audit and text-vector fusion are used to derive a ternary audit vector and an ontology frequency vector, which are then combined with the uncertainty in the tree edges and nodes to obtain a hybrid score.

This module implements:
1. audit matrix construction,
2. ontology-aware token frequency extraction,
3. path-signature-style reduction,
4. hybrid scoring with Bayesian evidence update.
"""

import math
import random
import sys
import pathlib
import numpy as np

# ----------------------------------------------------------------------
# Constants & utilities (shared)
# ----------------------------------------------------------------------
CLASSIFICATIONS = {
    "usable_now": 1,
    "research_only": 0,
    "needs_conversion": -1,
    "unsafe_for_fastpath": -1,
    "unsupported": -1,
}
WORD_RE = re.compile(r"\S+")
DEFAULT_TERMS = (
    "ENTITY",
    "ATTRIBUTE",
    "RELATIONSHIP",
    "ACTION",
    "EVENT",
    "TIME",
    "EVIDENCE",
    "CLAIM",
    "HYPOTHESIS",
    "SIGNAL",
    "PATTERN",
    "TOOL",
    "ALGORITHM",
    "BOOK",
    "SOURCE",
    "LEAD",
    "LOCATION",
    "LAW",
    "RULE",
)

def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability using the Bayes' rule."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Update the prior probability using the Bayes' rule."""
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Compute the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]], root: str, edge_priors: Dict[Tuple[str, str], float], path_weight: float = 0.2) -> float:
    """Compute the minimum-cost tree score with uncertainty."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b]) * edge_priors[(a, b)]
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())

def ternary_audit_vector(classification: str) -> Tuple[float, float, float]:
    """Compute the ternary audit vector."""
    return (CLASSIFICATIONS[classification] + 1) / 2

def ontology_frequency_vector(terms: List[str]) -> np.ndarray:
    """Compute the ontology frequency vector."""
    return np.array([terms.count(term) / len(terms) for term in DEFAULT_TERMS])

def hybrid_score(audit_vector: Tuple[float, float, float], frequency_vector: np.ndarray, path_signature: float) -> float:
    """Compute the hybrid score."""
    return (audit_vector[0] * path_signature) * (frequency_vector.dot(np.array([1.0, 1.0, 1.0])))

def hybrid_tree_cost(nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]], root: str, edge_priors: Dict[Tuple[str, str], float], path_weight: float = 0.2) -> float:
    """Compute the hybrid tree score with Bayesian evidence update."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b]) * edge_priors[(a, b)]
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    path_signature = np.prod([ternary_audit_vector(nodes[node][0]) for node in adj[root]])
    frequency_vector = ontology_frequency_vector([WORD_RE.findall(text)[0] for text in nodes.values()])
    score = hybrid_score((path_signature, path_signature, path_signature), frequency_vector, path_signature)
    return material + path_weight * sum(dist.values()) + bayes_update(prior=edge_priors[(root, root)], likelihood=score, marginal=bayes_marginal(prior=edge_priors[(root, root)], likelihood=score, false_positive=1.0 - edge_priors[(root, root)]))

if __name__ == "__main__":
    nodes = {"A": ("0.0", "0.0"), "B": ("1.0", "1.0"), "C": ("2.0", "2.0")}
    edges = [("A", "B"), ("B", "C")]
    edge_priors = {("A", "B"): 0.5, ("B", "C"): 0.5}
    print(hybrid_tree_cost(nodes, edges, "A", edge_priors))