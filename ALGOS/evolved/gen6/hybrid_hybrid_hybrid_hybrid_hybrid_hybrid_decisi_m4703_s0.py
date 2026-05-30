# DARWIN HAMMER — match 4703, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m1547_s0.py (gen5)
# parent_b: hybrid_hybrid_decision_hygi_hybrid_hybrid_model__m3_s2.py (gen3)
# born: 2026-05-29T23:57:34Z

"""
Hybrid Algorithm: Fusion of Distributed Perceptual Hashing & Decision Hygiene with
Shannon Entropy and Krampus‑Ollivier‑Ricci Curvature.

Parents:
- hybrid_hybrid_hybrid_distri_hybrid_hybrid_m1547_s0.py (perceptual hashing,
  geometric product weighting, Voronoi‑like clustering)
- hybrid_hybrid_decision_hygi_hybrid_hybrid_model__m3_s2.py (regex feature
  extraction, Shannon entropy weighting, curvature on graphs)

Mathematical Bridge:
The feature‑count vector extracted by the Decision‑Hygiene regexes is turned
into a probability distribution and its Shannon entropy 𝑯 is used as a scalar
weight.  For each node we also build a multivector 𝑣 (the raw numeric feature
vector).  The Clifford geometric product 𝑔 = 𝑣·𝑟 + 𝑣∧𝑟 with a global
reference vector 𝑟 yields a scalar part 𝑔₀ = 𝑣·𝑟.  The final weight
𝑤 = 𝑔₀·𝑯 modulates both the perceptual hash (by scaling the feature values)
and the edge weights fed to the Ollivier‑Ricci curvature computation.  This
creates a single unified system where feature‑based uncertainty and geometric
relationships jointly shape clustering and curvature analysis.
"""

import sys
import math
import random
import pathlib
import re
from collections import Counter, defaultdict
from typing import Mapping, Hashable, Sequence, List, Dict, Set, Tuple

import numpy as np

Node = Hashable
Graph = Mapping[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Parent A – perceptual hashing utilities
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash of a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:                     # deterministic length
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers."""
    return (a ^ b).bit_count()


def cluster_by_phash(hashes: Dict[Node, int], max_distance: int = 4) -> List[List[Node]]:
    """Greedy clustering of nodes whose hashes differ by ≤ max_distance bits."""
    clusters: List[List[Node]] = []
    for node, h in hashes.items():
        placed = False
        for cluster in clusters:
            if any(hamming_distance(h, hashes[other]) <= max_distance for other in cluster):
                cluster.append(node)
                placed = True
                break
        if not placed:
            clusters.append([node])
    return clusters


# ----------------------------------------------------------------------
# Parent B – decision‑hygiene feature extraction and entropy
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


def extract_feature_counts(text: str) -> Counter:
    """Count occurrences of each regex category in a piece of text."""
    counts = Counter()
    counts["evidence"] = len(EVIDENCE_RE.findall(text))
    counts["planning"] = len(PLANNING_RE.findall(text))
    counts["delay"] = len(DELAY_RE.findall(text))
    counts["support"] = len(SUPPORT_RE.findall(text))
    counts["boundary"] = len(BOUNDARY_RE.findall(text))
    counts["outcome"] = len(OUTCOME_RE.findall(text))
    return counts


def shannon_entropy(counts: Counter) -> float:
    """Shannon entropy of the normalized count distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values())) / total
    return -float(np.sum(probs * np.log2(probs + 1e-12)))


# ----------------------------------------------------------------------
# Geometric product (Clifford) – simplified to scalar (inner) + bivector (outer)
# ----------------------------------------------------------------------
def geometric_product(v1: np.ndarray, v2: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute a simplified geometric product between two vectors.
    Returns:
        scalar_part = v1·v2  (inner product)
        bivector_part = outer product matrix v1 ⊗ v2 (antisymmetric part)
    """
    scalar_part = float(np.dot(v1, v2))
    outer = np.outer(v1, v2)
    bivector_part = outer - outer.T                     # antisymmetric component
    return scalar_part, bivector_part


# ----------------------------------------------------------------------
# Hybrid weight computation
# ----------------------------------------------------------------------
def hybrid_weight(feature_vec: List[float], entropy: float, ref_vec: np.ndarray) -> float:
    """
    Combine entropy with the scalar part of the geometric product.
    The reference vector is a global anchor (e.g., mean of all feature vectors).
    """
    v = np.asarray(feature_vec, dtype=float)
    scalar, _ = geometric_product(v, ref_vec)
    return scalar * entropy


# ----------------------------------------------------------------------
# Hybrid graph construction and curvature
# ----------------------------------------------------------------------
def build_weighted_graph(
    clusters: List[List[Node]],
    node_weights: Dict[Node, float],
) -> Dict[Node, Set[Tuple[Node, float]]]:
    """
    Create an undirected weighted graph.
    Nodes inside the same cluster are fully connected.
    Edge weight = (w_i + w_j) / 2 where w_i is the hybrid weight of node i.
    """
    graph: Dict[Node, Set[Tuple[Node, float]]] = defaultdict(set)
    for cluster in clusters:
        for i, u in enumerate(cluster):
            for v in cluster[i + 1 :]:
                avg_weight = (node_weights[u] + node_weights[v]) / 2.0
                graph[u].add((v, avg_weight))
                graph[v].add((u, avg_weight))
    return graph


def compute_ollivier_ricci_curvature(
    graph: Dict[Node, Set[Tuple[Node, float]]]
) -> Dict[Tuple[Node, Node], float]:
    """
    Simplified Ollivier‑Ricci curvature.
    For each edge (u,v) we use the edge weight w_uv and the degrees deg(u), deg(v):
        κ(u,v) = 1 - w_uv / (deg(u) + deg(v))
    This captures how “tight” the connection is relative to local connectivity.
    """
    curvature: Dict[Tuple[Node, Node], float] = {}
    for u, neighbors in graph.items():
        deg_u = len(neighbors)
        for v, w_uv in neighbors:
            if (v, u) in curvature:          # avoid double computation
                continue
            deg_v = len(graph[v])
            if deg_u + deg_v == 0:
                cur = 0.0
            else:
                cur = 1.0 - w_uv / (deg_u + deg_v)
            curvature[(u, v)] = cur
    return curvature


# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_pipeline(
    node_features: Dict[Node, List[float]],
    node_texts: Dict[Node, str],
) -> Tuple[List[List[Node]], Dict[Tuple[Node, Node], float]]:
    """
    1. Extract regex‑based feature counts and compute Shannon entropy per node.
    2. Build a global reference vector (mean of all numeric feature vectors).
    3. Compute a hybrid weight w_i = (v_i·r) * H_i for each node.
    4. Scale the numeric feature vector by w_i and compute a perceptual hash.
    5. Cluster nodes via hash proximity.
    6. Build a weighted graph from clusters and compute Ollivier‑Ricci curvature.
    Returns the list of clusters and the curvature dictionary.
    """
    # Step 1 – entropy per node
    entropies: Dict[Node, float] = {}
    for n, txt in node_texts.items():
        cnts = extract_feature_counts(txt)
        entropies[n] = shannon_entropy(cnts)

    # Step 2 – reference vector (mean)
    all_vectors = np.stack([np.asarray(v, dtype=float) for v in node_features.values()], axis=0)
    ref_vec = np.mean(all_vectors, axis=0)

    # Step 3 – hybrid weights
    hybrid_weights: Dict[Node, float] = {}
    for n, vec in node_features.items():
        hybrid_weights[n] = hybrid_weight(vec, entropies.get(n, 0.0), ref_vec)

    # Step 4 – weighted perceptual hashes
    hashes: Dict[Node, int] = {}
    for n, vec in node_features.items():
        scaled = [x * hybrid_weights[n] for x in vec]
        hashes[n] = compute_phash(scaled)

    # Step 5 – clustering
    clusters = cluster_by_phash(hashes, max_distance=4)

    # Step 6 – graph + curvature
    weighted_graph = build_weighted_graph(clusters, hybrid_weights)
    curvature = compute_ollivier_ricci_curvature(weighted_graph)

    return clusters, curvature


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic dataset
    random.seed(42)
    np.random.seed(42)

    nodes = [f"node_{i}" for i in range(10)]

    # Random numeric features (3‑dimensional)
    node_features: Dict[Node, List[float]] = {
        n: list(np.random.rand(3) * 10) for n in nodes
    }

    # Random textual snippets containing some of the regex keywords
    snippets = [
        "Evidence was verified and the source is documented.",
        "We need a plan and a checklist for the next phase.",
        "Please wait, we will pause the rollout tomorrow.",
        "Ask the doctor for support and advice.",
        "Set clear boundaries and protect privacy.",
        "The task is done and shipped successfully.",
        "Evidence and proof are required before audit.",
        "Planning and scheduling are critical.",
        "Delay the deployment, hold off for safety.",
        "Support team will handle the outcome."
    ]
    node_texts: Dict[Node, str] = {n: snippets[i % len(snippets)] for i, n in enumerate(nodes)}

    clusters_out, curvature_out = hybrid_pipeline(node_features, node_texts)

    print("Clusters (by node list):")
    for idx, cl in enumerate(clusters_out):
        print(f"  C{idx}: {cl}")

    print("\nSample curvature values (first 5 edges):")
    for i, ((u, v), cur) in enumerate(curvature_out.items()):
        if i >= 5:
            break
        print(f"  Edge ({u}, {v}): κ = {cur:.4f}")

    sys.exit(0)