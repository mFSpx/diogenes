# DARWIN HAMMER — match 178, survivor 4
# gen: 2
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s0.py (gen1)
# parent_b: hybrid_infotaxis_minhash_m63_s2.py (gen1)
# born: 2026-05-29T23:27:23Z

"""Hybrid Minimum‑Cost Tree, Bayesian Update, and Infotaxis‑MinHash.

Parents:
- hybrid_minimum_cost_tree_bayes_update_m6_s0.py (minimum‑cost tree with Bayesian
  edge weighting)
- hybrid_infotaxis_minhash_m63_s2.py (entropy‑driven infotaxis using MinHash
  signatures)

Mathematical bridge:
The Bayesian posterior `P(edge | evidence)` obtained from the tree module is
interpreted as the probability `p_hit` of a “successful” action in the infotaxis
framework.  A MinHash signature of the current edge set provides a discrete
probability distribution over hash buckets; its Shannon entropy quantifies the
uncertainty of the tree’s structure.  For a candidate edge we evaluate the
expected post‑action entropy

    E[H] = p_hit * H(sig ∪ {edge}) + (1‑p_hit) * H(sig)

and greedily add the edge that minimises `E[H]`.  Thus the tree construction
simultaneously respects Bayesian evidence (edge probabilities) and infotaxis
principles (entropy reduction via MinHash similarity)."""

import math
import random
import sys
import pathlib
from pathlib import Path
from collections import Counter
from typing import Iterable, List, Tuple, Dict, Set

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Bayesian utilities (from parent A)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute marginal probability P(E) = L·π + FP·(1‑π)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must be in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Return posterior P(H|E) = π·L / P(E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


def bayes_edge_posterior(
    node: str,
    edge: Edge,
    priors: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
) -> float:
    """Posterior probability that `edge` is the true connection for `node`."""
    prior = priors[node]
    like = likelihoods[edge]
    fp = false_positives[edge]
    marginal = bayes_marginal(prior, like, fp)
    return bayes_update(prior, like, marginal)


# ----------------------------------------------------------------------
# MinHash utilities (from parent B)
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    # 8‑byte Blake2b digest → 64‑bit integer
    return int.from_bytes(
        np.frombuffer(
            np.frombuffer(hashlib.blake2b(data, digest_size=8).digest(), dtype=np.uint8),
            dtype=np.uint64,
        ),
        "big",
    )


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of length *k* for a token set."""
    tok_set: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not tok_set:
        return [MAX64] * k
    # For each seed compute the minimum hash over the token set
    sig: List[int] = []
    for i in range(k):
        min_hash = min(_hash(i, t) for t in tok_set)
        sig.append(min_hash)
    return sig


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Approximate Jaccard similarity via MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def entropy_of_signature(sig: List[int]) -> float:
    """Shannon entropy (bits) of the discrete distribution induced by a signature."""
    if not sig:
        return 0.0
    counts = Counter(sig)
    total = len(sig)
    ent = 0.0
    for cnt in counts.values():
        p = cnt / total
        ent -= p * math.log2(p)
    return ent


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def edge_token(edge: Edge) -> str:
    """Canonical string representation of an edge for MinHash."""
    a, b = edge
    return f"{min(a,b)}-{max(a,b)}"


def signature_from_edges(edges: List[Edge], k: int = 128) -> List[int]:
    """Compute MinHash signature of a set of edges."""
    return signature((edge_token(e) for e in edges), k=k)


def expected_entropy_after_edge(
    current_edges: List[Edge],
    candidate: Edge,
    priors: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
    k: int = 128,
) -> float:
    """Expected entropy if `candidate` were added to the tree."""
    # Posterior probability that the candidate edge is “correct”
    p_hit = bayes_edge_posterior(
        node=candidate[0],
        edge=candidate,
        priors=priors,
        likelihoods=likelihoods,
        false_positives=false_positives,
    )
    # Current signature & entropy
    sig_cur = signature_from_edges(current_edges, k=k)
    h_cur = entropy_of_signature(sig_cur)

    # Signature after successful addition
    sig_hit = signature_from_edges(current_edges + [candidate], k=k)
    h_hit = entropy_of_signature(sig_hit)

    # Expected entropy (infotaxis formula)
    return p_hit * h_hit + (1.0 - p_hit) * h_cur


def select_best_edge(
    visited: Set[str],
    candidate_edges: List[Edge],
    priors: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
    k: int = 128,
) -> Edge:
    """Choose the edge that minimises expected post‑action entropy."""
    best_edge = None
    best_val = math.inf
    for e in candidate_edges:
        # Ensure the edge connects visited ↔ unvisited
        a, b = e
        if (a in visited) ^ (b in visited):
            exp_ent = expected_entropy_after_edge(
                current_edges=list(visited_edges(visited, e)),  # placeholder, will be replaced
                candidate=e,
                priors=priors,
                likelihoods=likelihoods,
                false_positives=false_positives,
                k=k,
            )
            if exp_ent < best_val:
                best_val = exp_ent
                best_edge = e
    if best_edge is None:
        raise RuntimeError("No connectable edge found")
    return best_edge


def visited_edges(visited: Set[str], new_edge: Edge) -> List[Edge]:
    """Utility to collect edges whose both endpoints are already in `visited`.
    Used only for entropy calculation of the current tree."""
    # In the building loop we keep an explicit list of selected edges,
    # so this helper simply returns that list; for the selection step we pass
    # the already‑chosen edge list directly, avoiding recomputation.
    # The function exists to keep the signature of `select_best_edge` clean.
    raise NotImplementedError  # will be overridden in the main loop.


def hybrid_build_tree(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    priors: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
    k: int = 128,
) -> List[Edge]:
    """
    Construct a spanning tree by repeatedly adding the edge that yields the
    greatest expected reduction in MinHash‑signature entropy, where the edge
    probabilities are supplied by Bayesian updates.

    Returns the list of selected edges (size = |nodes|‑1).
    """
    if root not in nodes:
        raise ValueError("Root must be one of the nodes")
    # Initialise
    visited: Set[str] = {root}
    selected: List[Edge] = []

    # Pre‑compute adjacency for fast candidate lookup
    adjacency: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    # Helper to produce the current edge list for entropy computation
    def current_edge_list() -> List[Edge]:
        return selected.copy()

    while len(visited) < len(nodes):
        # Gather all edges crossing the frontier
        frontier: List[Edge] = []
        for v in visited:
            for nb in adjacency[v]:
                if nb not in visited:
                    frontier.append((v, nb))

        if not frontier:
            raise RuntimeError("Graph is not connected")

        # Choose the best frontier edge
        best_edge = None
        best_exp = math.inf
        cur_sig = signature_from_edges(current_edge_list(), k=k)
        h_cur = entropy_of_signature(cur_sig)

        for e in frontier:
            a, b = e
            # Posterior probability that this edge is the true connection
            p_hit = bayes_edge_posterior(
                node=a,
                edge=e,
                priors=priors,
                likelihoods=likelihoods,
                false_positives=false_positives,
            )
            # Entropy if edge is added
            sig_hit = signature_from_edges(current_edge_list() + [e], k=k)
            h_hit = entropy_of_signature(sig_hit)
            exp_ent = p_hit * h_hit + (1.0 - p_hit) * h_cur
            if exp_ent < best_exp:
                best_exp = exp_ent
                best_edge = e

        if best_edge is None:
            raise RuntimeError("Failed to select an edge")

        # Incorporate the chosen edge
        selected.append(best_edge)
        visited.update(best_edge)

    return selected


def hybrid_tree_cost(
    nodes: Dict[str, Point],
    tree_edges: List[Edge],
    priors: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
    path_weight: float = 0.2,
) -> float:
    """
    Compute a scalar cost for a fully built tree.
    The cost mixes Euclidean length with Bayesian‑adjusted weights and the
    final entropy of the MinHash signature.

    cost = Σ (length(e) * posterior(e) ** path_weight) + λ * H(sig)
    where λ is a small scaling factor.
    """
    total = 0.0
    for a, b in tree_edges:
        # Physical distance
        dist = math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
        # Bayesian posterior for this directed edge (using the first node as source)
        post = bayes_edge_posterior(a, (a, b), priors, likelihoods, false_positives)
        total += dist * (post ** path_weight)

    # Entropy regularisation
    sig = signature_from_edges(tree_edges)
    entropy = entropy_of_signature(sig)
    lambda_reg = 0.05
    return total + lambda_reg * entropy


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
    }
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("B", "C")]

    # Random but reproducible probabilities
    random.seed(42)
    priors = {n: random.uniform(0.4, 0.9) for n in nodes}
    likelihoods = {e: random.uniform(0.6, 0.95) for e in edges}
    false_positives = {e: random.uniform(0.05, 0.3) for e in edges}

    # Build hybrid tree
    tree = hybrid_build_tree(
        nodes=nodes,
        edges=edges,
        root="A",
        priors=priors,
        likelihoods=likelihoods,
        false_positives=false_positives,
        k=64,
    )
    print("Selected edges:", tree)

    # Compute cost
    cost = hybrid_tree_cost(
        nodes,
        tree,
        priors,
        likelihoods,
        false_positives,
        path_weight=0.25,
    )
    print("Hybrid tree cost:", cost)