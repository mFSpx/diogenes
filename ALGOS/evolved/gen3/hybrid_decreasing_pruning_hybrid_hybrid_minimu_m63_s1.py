# DARWIN HAMMER — match 63, survivor 1
# gen: 3
# parent_a: decreasing_pruning.py (gen0)
# parent_b: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s1.py (gen2)
# born: 2026-05-29T23:25:28Z

"""
This module represents a hybrid algorithm, combining the principles of decreasing-rate pruning 
from decreasing_pruning.py and hybrid minimum-cost tree scoring with epistemic certainty computation 
from hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s1.py.
The mathematical bridge between these two systems is established by incorporating the epistemic 
certainty flags into the edge weights of the minimum-cost tree, allowing the tree to adapt and 
re-weight its edges based on both physical distances and epistemic certainty, and then applying 
a decreasing-rate pruning schedule to the resulting tree.
"""

import math
import numpy as np
import random
import sys
import pathlib
from collections.abc import Hashable

EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non-negative')
    return min(1.0, lam * math.exp(-alpha * t))

def prune_edges(edges: list[Hashable], t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> list[Hashable]:
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]

def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for Bayesian update."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform Bayesian update on the prior probability."""
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def certainty(label: str, *, confidence_bps: int, authority_class: str, rationale: str, evidence_refs: tuple[str, ...] = ()) -> dict:
    """Create an epistemic certainty flag."""
    if label not in EPISTEMIC_FLAGS:
        raise ValueError(f"unknown epistemic flag: {label!r}")
    if not 0 <= int(confidence_bps) <= 10000:
        raise ValueError("confidence_bps must be 0..10000")
    return {
        "label": label,
        "confidence_bps": int(confidence_bps),
        "authority_class": authority_class,
        "rationale": rationale,
        "evidence_refs": evidence_refs,
    }

def hybrid_tree_cost(nodes: dict[str, tuple[float, float]], edges: list[tuple[str, str]], root: str, 
                     prior_probabilities: dict[str, float], likelihoods: dict[tuple[str, str], float], 
                     false_positives: dict[tuple[str, str], float], certainty_flags: dict[tuple[str, str], dict]) -> float:
    total_cost = 0.0
    for edge in edges:
        node_a, node_b = edge
        distance = length(nodes[node_a], nodes[node_b])
        prior = prior_probabilities[node_a]
        likelihood = likelihoods[edge]
        false_positive = false_positives[edge]
        marginal = bayes_marginal(prior, likelihood, false_positive)
        updated_prior = bayes_update(prior, likelihood, marginal)
        certainty_flag = certainty_flags[edge]
        confidence_bps = certainty_flag["confidence_bps"]
        total_cost += distance * (1.0 - confidence_bps / 10000.0)
    return total_cost

def pruned_hybrid_tree_cost(nodes: dict[str, tuple[float, float]], edges: list[tuple[str, str]], root: str, 
                            prior_probabilities: dict[str, float], likelihoods: dict[tuple[str, str], float], 
                            false_positives: dict[tuple[str, str], float], certainty_flags: dict[tuple[str, str], dict], t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    pruned_edges = prune_edges(edges, t, lam, alpha)
    return hybrid_tree_cost(nodes, pruned_edges, root, prior_probabilities, likelihoods, false_positives, certainty_flags)

def pruned_hybrid_tree(nodes: dict[str, tuple[float, float]], edges: list[tuple[str, str]], root: str, 
                       prior_probabilities: dict[str, float], likelihoods: dict[tuple[str, str], float], 
                       false_positives: dict[tuple[str, str], float], certainty_flags: dict[tuple[str, str], dict], t: float, lam: float = 1.0, alpha: float = 0.2) -> dict:
    pruned_edges = prune_edges(edges, t, lam, alpha)
    return {
        "nodes": nodes,
        "edges": pruned_edges,
        "root": root,
        "prior_probabilities": prior_probabilities,
        "likelihoods": likelihoods,
        "false_positives": false_positives,
        "certainty_flags": certainty_flags,
    }

if __name__ == "__main__":
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    prior_probabilities = {
        "A": 0.5,
        "B": 0.3,
        "C": 0.2,
    }
    likelihoods = {
        ("A", "B"): 0.8,
        ("B", "C"): 0.7,
        ("C", "A"): 0.9,
    }
    false_positives = {
        ("A", "B"): 0.1,
        ("B", "C"): 0.2,
        ("C", "A"): 0.1,
    }
    certainty_flags = {
        ("A", "B"): certainty("FACT", confidence_bps=8000, authority_class="high", rationale="strong evidence"),
        ("B", "C"): certainty("PROBABLE", confidence_bps=6000, authority_class="medium", rationale="some evidence"),
        ("C", "A"): certainty("POSSIBLE", confidence_bps=4000, authority_class="low", rationale="weak evidence"),
    }
    t = 1.0
    lam = 1.0
    alpha = 0.2
    pruned_tree = pruned_hybrid_tree(nodes, edges, root, prior_probabilities, likelihoods, false_positives, certainty_flags, t, lam, alpha)
    print(pruned_tree)