# DARWIN HAMMER — match 2682, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_fisher_localization_m153_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_decisi_m1115_s0.py (gen4)
# born: 2026-05-29T23:43:33Z

import re
import math
import numpy as np
from collections.abc import Mapping, Hashable
from typing import List, Dict, Set, Tuple

# ----------------------------------------------------------------------
# Regular expressions for feature extraction
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
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|pa)\b", re.I)

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]

# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------
def extract_features(text: str) -> np.ndarray:
    """Return a numeric feature vector for a single piece of text."""
    counts = [
        len(EVIDENCE_RE.findall(text)),
        len(PLANNING_RE.findall(text)),
        len(DELAY_RE.findall(text)),
        len(SUPPORT_RE.findall(text)),
        len(BOUNDARY_RE.findall(text)),
        len(OUTCOME_RE.findall(text)),
        len(IMPULSIVE_RE.findall(text)),
    ]
    return np.array(counts, dtype=float)


# ----------------------------------------------------------------------
# Perceptual hash utilities
# ----------------------------------------------------------------------
def compute_phash(values: np.ndarray) -> int:
    """Simple perceptual hash: compare each entry to the median of the vector."""
    if values.size == 0:
        return 0
    median = np.median(values)
    bits = 0
    # limit to 64 bits for consistency with original code
    for v in values[:64]:
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers interpreted as bit‑strings."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------
def build_graph(feature_matrix: np.ndarray, max_hamming: int = 4) -> Graph:
    """
    Build an undirected graph where nodes are rows of ``feature_matrix``.
    Two nodes are connected if the Hamming distance of their perceptual
    hashes does not exceed ``max_hamming``.
    """
    n = feature_matrix.shape[0]
    hashes = [compute_phash(row) for row in feature_matrix]
    graph: Dict[str, Set[str]] = {str(i): set() for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(hashes[i], hashes[j]) <= max_hamming:
                graph[str(i)].add(str(j))
                graph[str(j)].add(str(i))
    return graph


# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature approximation
# ----------------------------------------------------------------------
def compute_ollivier_ricci(graph: Graph) -> Dict[str, float]:
    """
    Approximate Ollivier‑Ricci curvature using a simple degree‑based proxy.
    For an edge (i, j) the curvature κ(i,j) ≈ 1 - (|N(i) Δ N(j)| / max_deg),
    where Δ denotes symmetric difference and max_deg is the larger degree.
    Node curvature is the average of incident edge curvatures.
    """
    node_curvature: Dict[str, float] = {}
    for node, neighbors in graph.items():
        if not neighbors:
            node_curvature[node] = 0.0
            continue

        edge_curvs = []
        deg_i = len(neighbors)
        for nb in neighbors:
            deg_j = len(graph[nb])
            max_deg = max(deg_i, deg_j, 1)
            sym_diff = len(neighbors.symmetric_difference(graph[nb]))
            edge_curvs.append(1.0 - sym_diff / max_deg)

        node_curvature[node] = sum(edge_curvs) / len(edge_curvs)
    return node_curvature


# ----------------------------------------------------------------------
# Fisher information approximation
# ----------------------------------------------------------------------
def compute_fisher_information(features: np.ndarray) -> float:
    """
    Approximate the Fisher information of a set of feature vectors.
    We treat the empirical covariance Σ as the Fisher information matrix
    (up to a constant factor) and return its trace, which reflects total
    information content.
    """
    if features.shape[0] < 2:
        # With a single observation the covariance is undefined;
        # fall back to the sum of squares (original behaviour).
        return float(np.sum(features[0] ** 2))

    cov = np.cov(features, rowvar=False, bias=True)
    # Ensure numerical stability
    cov += np.eye(cov.shape[0]) * 1e-12
    fisher_trace = float(np.trace(np.linalg.inv(cov)))
    return fisher_trace


# ----------------------------------------------------------------------
# Integrated hybrid operation
# ----------------------------------------------------------------------
def hybrid_operation(texts: List[str], max_hamming: int = 4, curvature_weight: float = 0.5) -> Dict[str, float]:
    """
    Perform the hybrid computation on a collection of texts.

    Returns a dictionary with:
        - ``average_curvature``: mean node curvature over the graph.
        - ``fisher_information``: scalar Fisher information estimate.
        - ``combined_score``: weighted sum of the two metrics.
    """
    if not texts:
        raise ValueError("At least one text must be provided.")

    # Feature extraction
    feature_matrix = np.vstack([extract_features(t) for t in texts])

    # Graph and curvature
    graph = build_graph(feature_matrix, max_hamming=max_hamming)
    node_curv = compute_ollivier_ricci(graph)
    avg_curvature = float(np.mean(list(node_curv.values())))

    # Fisher information
    fisher_info = compute_fisher_information(feature_matrix)

    # Combine
    combined = curvature_weight * avg_curvature + (1 - curvature_weight) * fisher_info

    return {
        "average_curvature": avg_curvature,
        "fisher_information": fisher_info,
        "combined_score": combined,
    }


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "This is a test text with some evidence and planning keywords.",
        "The patient needs support and a clear boundary to stay safe.",
        "We have verified the source, but there is a delay in the schedule.",
        "Outcome was successful and the file was shipped.",
    ]
    result = hybrid_operation(sample_texts)
    for k, v in result.items():
        print(f"{k}: {v:.6f}")