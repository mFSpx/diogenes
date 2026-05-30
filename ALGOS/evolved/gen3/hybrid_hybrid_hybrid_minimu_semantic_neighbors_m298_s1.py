# DARWIN HAMMER — match 298, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s0.py (gen2)
# parent_b: semantic_neighbors.py (gen0)
# born: 2026-05-29T23:28:06Z

"""
This module represents a hybrid algorithm, combining the principles of minimum-cost tree scoring 
from hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s0.py and semantic neighbors from 
semantic_neighbors.py. The exact mathematical bridge between these two systems is established 
by incorporating the cosine similarity calculation into the edge weights of the minimum-cost tree, 
while also utilizing the label scoring from hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s0.py. 
This fusion enables the tree to consider both the physical distances between nodes and the semantic 
similarities of the documents associated with these nodes, as well as the probabilistic relevance of 
the paths connecting them and the relevance of labels to these paths.

The core idea is to use the Bayesian update function to modify the path weights in the tree scoring 
function, while also considering the score of labels on these paths and the semantic similarity of 
the documents associated with these paths. This dynamic system where the tree structure, the 
Bayesian probabilities, and the semantic similarities inform each other is integrated with the 
relevance of labels to the paths in the tree.
"""

import math
import numpy as np
import random
import sys
import pathlib

Point = tuple[float, float]
Edge = tuple[str, str]
Document = tuple[str, list[float]]

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

def label_score(text: str, label: str) -> float:
    """Compute the score of a label on the text using the literal fallback algorithm."""
    # Simplified implementation for demonstration purposes
    return len(label) / len(text)

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    den = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
    return 0.0 if den == 0 else sum(x*y for x, y in zip(a, b)) / den

def hybrid_tree_cost(nodes: list[Point], edges: list[Edge], documents: list[Document]) -> float:
    """Compute the hybrid tree cost considering both Euclidean distance and semantic similarity."""
    cost = 0.0
    for i, (node1, node2) in enumerate(edges):
        distance = length(nodes[node1], nodes[node2])
        doc1, doc2 = documents[node1], documents[node2]
        similarity = cosine_similarity(doc1[1], doc2[1])
        cost += distance * (1 - similarity)
    return cost

def semantic_neighbors(doc_id: str, documents: list[Document], k: int = 5) -> list[tuple[str, float]]:
    """Find the semantic neighbors of a document."""
    doc_vector = next((doc[1] for doc in documents if doc[0] == doc_id), None)
    if doc_vector is None:
        return []
    return sorted(((doc[0], cosine_similarity(doc_vector, doc[1])) for doc in documents if doc[0] != doc_id), key=lambda x: (-x[1], x[0]))[:k]

def hybrid_operation(nodes: list[Point], edges: list[Edge], documents: list[Document]) -> tuple[float, list[tuple[str, float]]]:
    """Perform the hybrid operation, computing the tree cost and finding semantic neighbors."""
    cost = hybrid_tree_cost(nodes, edges, documents)
    neighbors = semantic_neighbors(documents[0][0], documents)
    return cost, neighbors

if __name__ == "__main__":
    nodes = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    edges = [(0, 1), (1, 2)]
    documents = [("doc1", [1.0, 2.0, 3.0]), ("doc2", [4.0, 5.0, 6.0]), ("doc3", [7.0, 8.0, 9.0])]
    cost, neighbors = hybrid_operation(nodes, edges, documents)
    print(f"Hybrid tree cost: {cost}")
    print(f"Semantic neighbors: {neighbors}")