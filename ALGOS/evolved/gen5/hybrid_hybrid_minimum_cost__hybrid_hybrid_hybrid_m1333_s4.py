# DARWIN HAMMER — match 1333, survivor 4
# gen: 5
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_indy_learning_m816_s1.py (gen4)
# born: 2026-05-29T23:35:27Z

"""Hybrid Minimum‑Cost Tree with Bayesian Edge Updates and Ternary‑Audit / Ontology Fusion.

Parents:
- hybrid_minimum_cost_tree_bayes_update_m6_s1.py (tree cost + Bayesian edge probability update)
- hybrid_hybrid_ternar_hybrid_indy_learning_m816_s1.py (ternary audit vectors, ontology frequency vectors, path‑signature reduction)

Mathematical bridge:
Each tree edge carries a prior probability *pₑ*.  Evidence about the edge (e.g. sensor
readings) yields a likelihood *ℓₑ* and a false‑positive rate *ϕₑ*.  Using Bayes’
rule we obtain an updated edge weight *wₑ = pₑ·ℓₑ / (ℓₑ·pₑ + ϕₑ·(1‑pₑ))*.
Every node *v* is also described by a ternary audit vector *aᵥ∈{‑1,0,1}³* (derived from
its classification) and an ontology frequency vector *fᵥ∈ℝᴺ* extracted from associated
text.  A global signature *s = ∏_v aᵥ* (element‑wise product) summarises the audit
topology, while a normalised term‑weight *w* summarises the ontology space.
The scalar node score

    σᵥ = (aᵥ·s)·(fᵥ·w)

modulates the contribution of a node to the tree’s path‑weight term.
The hybrid cost therefore is

    C = Σₑ length(e)·wₑ  +  λ· Σ_v dist(root,v)·σᵥ

where λ is a tunable path‑weight factor.

The implementation below provides the full pipeline:
1. Bayesian update of edge priors.
2. Construction of audit matrix and ontology vectors.
3. Signature and weight computation.
4. Hybrid tree cost evaluation.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]          # (node_a, node_b) – order does not matter
Classification = str

# ----------------------------------------------------------------------
# Geometry utilities (Parent A)
# ----------------------------------------------------------------------
def euclidean_length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Bayesian update utilities (Parent A)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = ℓ·p + ϕ·(1‑p)"""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, false_positive: float) -> float:
    """Posterior P(H|E) = p·ℓ / P(E)"""
    marginal = bayes_marginal(prior, likelihood, false_positive)
    if marginal == 0.0:
        raise ValueError("Marginal probability is zero; cannot divide.")
    return prior * likelihood / marginal

def update_edge_priors(
    edge_priors: Dict[Edge, float],
    evidence: Dict[Edge, Tuple[float, float]],
) -> Dict[Edge, float]:
    """
    Apply Bayesian update to every edge prior.

    Parameters
    ----------
    edge_priors : dict
        Mapping Edge → prior probability (0‑1).
    evidence : dict
        Mapping Edge → (likelihood, false_positive).

    Returns
    -------
    dict
        Edge → posterior probability.
    """
    updated = {}
    for e, prior in edge_priors.items():
        if e not in evidence:
            updated[e] = prior
            continue
        likelihood, false_positive = evidence[e]
        updated[e] = bayes_update(prior, likelihood, false_positive)
    return updated

# ----------------------------------------------------------------------
# Ternary audit & ontology utilities (Parent B)
# ----------------------------------------------------------------------
CLASSIFICATION_MAP = {
    "usable_now": 1,
    "research_only": 0,
    "needs_conversion": -1,
    "unsafe_for_fastpath": -1,
    "unsupported": -1,
}

def ternary_vector(classif: Classification) -> Tuple[int, int, int]:
    """
    Map a classification string to a 3‑dimensional ternary vector.
    For simplicity we reuse the same scalar three times; more elaborate
    schemes could split the classification into three orthogonal aspects.
    """
    val = CLASSIFICATION_MAP.get(classif, -1)
    return (val, val, val)

def build_audit_matrix(node_classes: Dict[str, Classification]) -> np.ndarray:
    """
    Construct audit matrix A ∈ {‑1,0,1}^(V×3) where V is the number of nodes.
    Row order follows the insertion order of ``node_classes``.
    """
    rows = [ternary_vector(cls) for cls in node_classes.values()]
    return np.array(rows, dtype=int)

def ontology_frequency_vector(texts: List[str], vocabulary: List[str]) -> np.ndarray:
    """
    Simple term‑frequency vector over a fixed vocabulary.
    """
    vocab_index = {term: i for i, term in enumerate(vocabulary)}
    vec = np.zeros(len(vocabulary), dtype=float)
    for txt in texts:
        for token in txt.split():
            if token in vocab_index:
                vec[vocab_index[token]] += 1.0
    return vec

def normalised_weight_vector(freq: np.ndarray) -> np.ndarray:
    """L2‑normalise the frequency vector to obtain term weights."""
    norm = np.linalg.norm(freq)
    return freq / norm if norm > 0 else freq

def path_signature(audit_matrix: np.ndarray) -> np.ndarray:
    """
    Element‑wise cumulative product across the candidate axis:
    s_j = ∏_i A_{i,j}
    """
    return np.prod(audit_matrix, axis=0)

def node_score(
    audit_vec: np.ndarray,
    signature: np.ndarray,
    ontology_vec: np.ndarray,
    term_weights: np.ndarray,
) -> float:
    """
    σ = (a·s)·(f·w)
    """
    audit_part = float(np.dot(audit_vec, signature))
    ontology_part = float(np.dot(ontology_vec, term_weights))
    return audit_part * ontology_part

# ----------------------------------------------------------------------
# Hybrid tree cost (integration of both parents)
# ----------------------------------------------------------------------
def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    node_classes: Dict[str, Classification],
    node_texts: Dict[str, List[str]],
    edge_priors: Dict[Edge, float],
    edge_evidence: Dict[Edge, Tuple[float, float]],
    vocabulary: List[str],
    path_lambda: float = 0.2,
) -> float:
    """
    Compute the hybrid cost:
        C = Σₑ length(e)·posterior_e  +  λ· Σ_v dist(root,v)·σ_v

    Parameters
    ----------
    nodes : dict
        node_id → (x, y) coordinates.
    edges : list of Edge
        Undirected edges.
    root : str
        Identifier of the root node.
    node_classes : dict
        node_id → classification string.
    node_texts : dict
        node_id → list of textual snippets (will be concatenated).
    edge_priors : dict
        Edge → prior probability.
    edge_evidence : dict
        Edge → (likelihood, false_positive).
    vocabulary : list
        Fixed list of ontology terms.
    path_lambda : float
        Weight for the distance‑score term.

    Returns
    -------
    float
        The hybrid cost.
    """
    # 1. Bayesian update of edge priors
    posteriors = update_edge_priors(edge_priors, edge_evidence)

    # 2. Build adjacency list and compute material cost
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        # Normalise edge key for dictionary look‑up
        key = (a, b) if (a, b) in posteriors else (b, a)
        weight = posteriors.get(key, 1.0)   # default weight 1 if missing
        length = euclidean_length(nodes[a], nodes[b])
        material += length * weight
        adj[a].append(b)
        adj[b].append(a)

    # 3. Compute distances from root (BFS)
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + euclidean_length(nodes[cur], nodes[nb])
                stack.append(nb)

    # 4. Audit matrix and signature
    audit_mat = build_audit_matrix(node_classes)                 # V×3
    sig = path_signature(audit_mat)                             # (3,)

    # 5. Global ontology term weights (same for all nodes in this simple demo)
    all_texts = [txt for txts in node_texts.values() for txt in txts]
    global_freq = ontology_frequency_vector(all_texts, vocabulary)
    term_weights = normalised_weight_vector(global_freq)

    # 6. Node scores σ_v
    node_scores: Dict[str, float] = {}
    for nid, cls in node_classes.items():
        a_vec = np.array(ternary_vector(cls), dtype=float)      # (3,)
        texts = node_texts.get(nid, [])
        f_vec = ontology_frequency_vector(texts, vocabulary)   # (N,)
        score = node_score(a_vec, sig, f_vec, term_weights)
        node_scores[nid] = score

    # 7. Path‑score term
    path_term = sum(dist[nid] * node_scores.get(nid, 0.0) for nid in nodes)

    return material + path_lambda * path_term

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple geometry
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
    }
    edges = [("A", "B"), ("A", "C")]

    # Edge priors and synthetic evidence
    edge_priors = {
        ("A", "B"): 0.8,
        ("A", "C"): 0.6,
    }
    edge_evidence = {
        ("A", "B"): (0.9, 0.1),   # high likelihood, low false‑positive
        ("A", "C"): (0.4, 0.2),   # weaker evidence
    }

    # Node classifications
    node_classes = {
        "A": "usable_now",
        "B": "research_only",
        "C": "needs_conversion",
    }

    # Node associated texts (very small example)
    node_texts = {
        "A": ["ENTITY ACTION"],
        "B": ["RELATIONSHIP EVENT"],
        "C": ["ATTRIBUTE TIME"],
    }

    # Vocabulary (ontology terms)
    vocab = [
        "ENTITY", "ATTRIBUTE", "RELATIONSHIP", "ACTION", "EVENT",
        "TIME", "EVIDENCE", "CLAIM", "HYPOTHESIS", "SIGNAL",
    ]

    cost = hybrid_tree_cost(
        nodes=nodes,
        edges=edges,
        root="A",
        node_classes=node_classes,
        node_texts=node_texts,
        edge_priors=edge_priors,
        edge_evidence=edge_evidence,
        vocabulary=vocab,
        path_lambda=0.3,
    )
    print(f"Hybrid cost: {cost:.4f}")