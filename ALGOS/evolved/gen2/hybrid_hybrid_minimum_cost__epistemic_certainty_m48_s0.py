# DARWIN HAMMER — match 48, survivor 0
# gen: 2
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s0.py (gen1)
# parent_b: epistemic_certainty.py (gen0)
# born: 2026-05-29T23:23:58Z

"""
This module represents a hybrid algorithm, combining the principles of minimum-cost tree scoring 
from hybrid_minimum_cost_tree_bayes_update_m6_s0 and epistemic certainty from epistemic_certainty. 
The mathematical bridge between these two systems is established by incorporating the epistemic 
certainty flags into the edge weights of the minimum-cost tree, effectively allowing the tree to 
adapt and re-weight its edges based on both physical distances and epistemic certainty.

The core idea is to use the epistemic certainty flags to modify the path weights in the tree scoring 
function, thus creating a dynamic system where the tree structure and the epistemic certainty 
inform each other.
"""

import math
import numpy as np
import random
import sys
import pathlib

Point = tuple[float, float]
Edge = tuple[str, str]

EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

def length(a: Point, b: Point) -> float:
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

def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: tuple[str, ...] = (),
) -> dict[str, str]:
    return {
        "label": label,
        "confidence_bps": str(confidence_bps),
        "authority_class": authority_class,
        "rationale": rationale,
        "evidence_refs": evidence_refs,
    }

def hybrid_tree_cost(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
    prior_probabilities: dict[str, float],
    likelihoods: dict[Edge, float],
    false_positives: dict[Edge, float],
    certainty_flags: dict[Edge, dict[str, str]],
    path_weight: float = 0.2,
) -> float:
    """Calculate the cost of the tree incorporating Bayesian update and epistemic certainty in edge weights."""
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = 0.0
    bayes_weights = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        marginal = bayes_marginal(prior_probabilities[a], likelihoods[(a, b)], false_positives[(a, b)])
        updated_weight = bayes_update(prior_probabilities[a], likelihoods[(a, b)], marginal)
        bayes_weights[(a, b)] = updated_weight
        material += length(nodes[a], nodes[b]) * updated_weight * (1 + int(certainty_flags[(a, b)]["confidence_bps"]) / 10000)
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = (
                    dist[a]
                    + length(nodes[a], nodes[b])
                    * bayes_weights.get((a, b), 1.0)
                    * (1 + int(certainty_flags.get((a, b), {"confidence_bps": 0})["confidence_bps"]) / 10000)
                )
                stack.append(b)
    return material + path_weight * sum(dist.values())

def generate_random_tree(num_nodes: int, num_edges: int) -> (dict[str, Point], list[Edge], str):
    """Generate a random tree structure for testing."""
    nodes = {f"node_{i}": (random.uniform(0, 10), random.uniform(0, 10)) for i in range(num_nodes)}
    edges = []
    for _ in range(num_edges):
        a, b = random.sample(list(nodes.keys()), 2)
        edges.append((a, b))
    root = random.choice(list(nodes.keys()))
    return nodes, edges, root

def test_hybrid_tree() -> None:
    """Smoke test for the hybrid tree cost function."""
    nodes, edges, root = generate_random_tree(10, 15)
    prior_probabilities = {node: random.uniform(0, 1) for node in nodes}
    likelihoods = {(a, b): random.uniform(0, 1) for a, b in edges}
    false_positives = {(a, b): random.uniform(0, 1) for a, b in edges}
    certainty_flags = {
        (a, b): certainty(
            random.choice(EPISTEMIC_FLAGS),
            confidence_bps=random.randint(0, 10000),
            authority_class="random_authority",
            rationale="random_rationale",
        )
        for a, b in edges
    }
    cost = hybrid_tree_cost(
        nodes, edges, root, prior_probabilities, likelihoods, false_positives, certainty_flags
    )
    print(f"Hybrid tree cost: {cost}")

if __name__ == "__main__":
    test_hybrid_tree()