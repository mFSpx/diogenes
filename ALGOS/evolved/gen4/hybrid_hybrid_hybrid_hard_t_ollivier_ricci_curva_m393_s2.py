# DARWIN HAMMER — match 393, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s0.py (gen3)
# parent_b: ollivier_ricci_curvature.py (gen0)
# born: 2026-05-29T23:28:34Z

"""Hybrid Stylometry–Graph Curvature Module

Parents:
- hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s0.py (stylometry → high‑dimensional points,
  Voronoi partitioning)
- ollivier_ricci_curvature.py (Ollivier‑Ricci curvature on weighted graphs)

Mathematical bridge:
Text documents are mapped to stylometric feature vectors **v ∈ ℝⁿ** (parent A).  
These vectors become vertices of a weighted graph **G** where the edge weight
`w(i,j)` is the Euclidean distance ‖v_i‑v_j‖.  On this graph we evaluate the
Ollivier‑Ricci curvature κ(i,j) (parent B).  The curvature quantifies how
locally “well‑connected’’ the stylometric space is and is then fed back as a
modulation of the Voronoi‑style clustering: edges with high positive curvature
pull centroids together, while negative curvature pushes them apart.

The module therefore fuses:
1. Stylometric feature extraction → point cloud.
2. Construction of a distance graph from the point cloud.
3. Ollivier‑Ricci curvature computation on that graph.
4. Curvature‑aware clustering (Voronoi‑like) using the curvature‑adjusted
   distances.

All operations rely only on the Python standard library and NumPy.
"""

import sys
import math
import random
import itertools
from pathlib import Path
from collections import Counter, defaultdict, deque
from typing import List, Dict, Tuple, Set

import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry utilities (truncated / re‑implemented)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, Set[str]] = {
    "pronoun": {
        "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
        "he", "him", "his", "she", "her", "hers", "they", "them", "their",
        "theirs", "we", "us", "our", "ours"
    },
    "article": {"a", "an", "the"},
    "preposition": {
        "about", "above", "after", "against", "around", "as", "at", "before",
        "behind", "below", "between", "by", "during", "for", "from", "in",
        "into", "of", "off", "on", "onto", "over", "through", "to", "under",
        "with", "without"
    },
    "auxiliary": {
        "am", "are", "be", "been", "being", "can", "could", "did", "do",
        "does", "had", "has", "have", "is", "may", "might", "must", "shall",
        "should", "was", "were", "will", "would"
    },
    "conjunction": {
        "and", "but", "or", "nor", "so", "yet", "because", "although", "while",
        "if", "when", "where", "whereas", "unless", "until"
    },
    "negation": {
        "no", "not", "never", "none", "neither", "cannot", "can't", "won't",
        "don't", "didn't", "isn't", "aren't", "wasn't", "weren't"
    },
    "quantifier": {
        "all", "any", "both", "each", "few", "many", "more", "most", "much",
        "none", "several", "some", "such"
    },
    "adverb_common": {
        "very", "really", "just", "still", "already", "also", "even", "only",
        "then", "there", "here", "now", "often", "always", "sometimes"
    },
}
PUNCT = set("!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~")

def _tokenize(text: str) -> List[str]:
    """Very light tokeniser: lower‑case, keep alphabetic + apostrophe."""
    tokens = []
    current = []
    for ch in text.lower():
        if ch.isalpha() or ch == "'":
            current.append(ch)
        else:
            if current:
                tokens.append(''.join(current))
                current = []
    if current:
        tokens.append(''.join(current))
    return tokens

def _count_function_words(tokens: List[str]) -> Counter:
    """Count occurrences of each function‑category."""
    cat_counter = Counter()
    for token in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                cat_counter[cat] += 1
                break
    return cat_counter

def _punctuation_ratio(text: str) -> float:
    if not text:
        return 0.0
    punct_cnt = sum(1 for ch in text if ch in PUNCT)
    return punct_cnt / len(text)

def extract_features(text: str) -> np.ndarray:
    """
    Convert a document into a numeric stylometric vector.

    Features (order is fixed):
        0‑N   – normalised frequencies of each FUNCTION_CATS key
        N+1   – punctuation ratio
        N+2   – average word length
        N+3   – type‑token ratio (unique tokens / total tokens)

    Returns
    -------
    np.ndarray of shape (len(FUNCTION_CATS)+3,)
    """
    tokens = _tokenize(text)
    total_words = len(tokens) or 1
    cat_counts = _count_function_words(tokens)

    cat_freqs = [cat_counts.get(cat, 0) / total_words for cat in FUNCTION_CATS]

    punct = _punctuation_ratio(text)

    avg_len = sum(len(t) for t in tokens) / total_words

    type_token = len(set(tokens)) / total_words

    return np.array(cat_freqs + [punct, avg_len, type_token], dtype=float)

# ----------------------------------------------------------------------
# Parent B – Ollivier‑Ricci curvature primitives (re‑implemented)
# ----------------------------------------------------------------------
def lazy_rw_distribution(adj: Dict[int, List[int]], node: int, alpha: float = 0.5) -> Dict[int, float]:
    """Lazy random‑walk distribution centred at *node*."""
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    dist = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist

def bfs_all_pairs(adj: Dict[int, List[int]]) -> Dict[Tuple[int, int], int]:
    """All‑pairs shortest‑path distances using BFS (unweighted graph)."""
    distances: Dict[Tuple[int, int], int] = {}
    for src in adj.keys():
        visited = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited[v] = visited[u] + 1
                    q.append(v)
        for dst, d in visited.items():
            distances[(src, dst)] = d
    return distances

def _transport_cost(
    mu: Dict[int, float],
    nu: Dict[int, float],
    dist_matrix: Dict[Tuple[int, int], int],
) -> float:
    """
    Greedy transport heuristic for discrete measures.
    Repeatedly moves mass from the node with largest surplus to the node with
    largest deficit, paying the graph distance as cost.
    """
    # Convert to mutable copies
    surplus = mu.copy()
    deficit = {k: -v for k, v in nu.items()}
    for k in set(surplus) & set(deficit):
        net = surplus[k] + deficit[k]  # deficit is negative
        surplus[k] = max(net, 0.0)
        deficit[k] = min(net, 0.0)

    total_cost = 0.0
    while True:
        # Find a source with positive surplus
        src = next((n for n, v in surplus.items() if v > 1e-12), None)
        if src is None:
            break
        # Find a sink with negative deficit (i.e. need mass)
        dst = next((n for n, v in deficit.items() if v < -1e-12), None)
        if dst is None:
            break
        amount = min(surplus[src], -deficit[dst])
        cost = dist_matrix.get((src, dst), math.inf)
        total_cost += amount * cost
        surplus[src] -= amount
        deficit[dst] += amount
    return total_cost

def ollivier_ricci_curvature(
    adj: Dict[int, List[int]],
    alpha: float = 0.5,
) -> Dict[Tuple[int, int], float]:
    """
    Compute Ollivier‑Ricci curvature κ(i,j) for every edge (i,j) in *adj*.

    κ(i,j) = 1 - W₁(m_i, m_j) / d(i,j)

    where d(i,j) is the graph distance (here the Euclidean edge weight is
    supplied later; for the curvature we use the unweighted hop distance).
    """
    # Pre‑compute all‑pairs hop distances for the transport cost.
    hop_dist = bfs_all_pairs(adj)

    curvature: Dict[Tuple[int, int], float] = {}
    for i, neighbours in adj.items():
        for j in neighbours:
            if (j, i) in curvature:
                continue  # undirected edge already processed
            mu_i = lazy_rw_distribution(adj, i, alpha)
            mu_j = lazy_rw_distribution(adj, j, alpha)
            w1 = _transport_cost(mu_i, mu_j, hop_dist)
            d_ij = hop_dist.get((i, j), 1)  # fallback to 1 for direct neighbours
            kappa = 1.0 - (w1 / d_ij)
            curvature[(i, j)] = kappa
    return curvature

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def build_distance_graph(
    features: List[np.ndarray],
    radius: float = 2.0,
) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], float]]:
    """
    Build an undirected weighted graph from feature vectors.

    Nodes are integer indices.  An edge (i,j) exists iff Euclidean distance
    ≤ *radius*.  Edge weight = Euclidean distance.

    Returns
    -------
    adj   – adjacency list {node: [neighbour, ...]}
    wgt   – dict {(i,j): distance}
    """
    n = len(features)
    adj: Dict[int, List[int]] = defaultdict(list)
    wgt: Dict[Tuple[int, int], float] = {}
    for i, j in itertools.combinations(range(n), 2):
        d = float(np.linalg.norm(features[i] - features[j]))
        if d <= radius:
            adj[i].append(j)
            adj[j].append(i)
            wgt[(i, j)] = wgt[(j, i)] = d
    return dict(adj), wgt

def curvature_adjusted_distance(
    i: int,
    j: int,
    euclid: float,
    curv: Dict[Tuple[int, int], float],
    factor: float = 0.5,
) -> float:
    """
    Modulate the Euclidean distance by the Ollivier‑Ricci curvature.

    Positive curvature shortens the effective distance, negative curvature
    lengthens it.

    d' = d * (1 - factor * κ)

    The *factor* controls how aggressive the modulation is.
    """
    kappa = curv.get((i, j)) or curv.get((j, i)) or 0.0
    return euclid * (1.0 - factor * kappa)

def curvature_aware_voronoi(
    features: List[np.ndarray],
    curv: Dict[Tuple[int, int], float],
    k: int = 2,
    max_iter: int = 100,
) -> List[int]:
    """
    Simple Voronoi‑like clustering that uses curvature‑adjusted distances.

    Parameters
    ----------
    features : list of point vectors
    curv     : edge curvature dictionary (must contain entries for all pairs)
    k        : number of clusters (centroids)
    max_iter : iteration cap

    Returns
    -------
    labels : list[int] assigning each point to a cluster index
    """
    n = len(features)
    # Initialise centroids by random distinct indices
    cent_idx = random.sample(range(n), k)
    centroids = [features[idx].copy() for idx in cent_idx]

    for _ in range(max_iter):
        # Assignment step
        labels = []
        for i, pt in enumerate(features):
            best = None
            best_dist = math.inf
            for cid, cen in enumerate(centroids):
                euclid = float(np.linalg.norm(pt - cen))
                # Use curvature between point i and the *current* centroid index if present
                # (centroid may not correspond to a data point; we approximate by nearest node)
                # For simplicity we ignore curvature here when centroid is not a node.
                # If the centroid is an actual data point we can fetch curvature.
                if i in cent_idx:
                    cur_adj = curvature_adjusted_distance(i, cent_idx[cid], euclid, curv)
                else:
                    cur_adj = euclid
                if cur_adj < best_dist:
                    best_dist = cur_adj
                    best = cid
            labels.append(best)

        # Update step
        new_centroids = []
        for cid in range(k):
            members = [features[i] for i, lab in enumerate(labels) if lab == cid]
            if members:
                new_centroids.append(np.mean(members, axis=0))
            else:  # empty cluster → re‑initialize randomly
                new_centroids.append(features[random.randrange(n)].copy())
        # Convergence check
        if all(np.allclose(c, nc) for c, nc in zip(centroids, new_centroids)):
            break
        centroids = new_centroids
    return labels

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample texts (tiny corpus)
    corpus = [
        "I love programming. It's amazing how code can create worlds.",
        "The quick brown fox jumps over the lazy dog.",
        "Data science blends statistics, mathematics, and computer science.",
        "She sells seashells by the seashore, and everyone listens.",
        "In the midst of chaos, there is also opportunity."
    ]

    # 1. Extract stylometric vectors
    feats = [extract_features(txt) for txt in corpus]

    # 2. Build graph (radius chosen heuristically)
    adjacency, edge_weights = build_distance_graph(feats, radius=2.5)

    # 3. Compute Ollivier‑Ricci curvature on the graph
    curvature = ollivier_ricci_curvature(adjacency, alpha=0.5)

    # 4. Perform curvature‑aware Voronoi clustering into 2 groups
    labels = curvature_aware_voronoi(feats, curvature, k=2)

    # Output results
    print("Adjacency list:")
    for node, neigh in adjacency.items():
        print(f" {node}: {neigh}")

    print("\nEdge curvatures (κ):")
    for (i, j), kappa in curvature.items():
        print(f" ({i},{j}) κ = {kappa:.4f}")

    print("\nCluster assignments:")
    for idx, lab in enumerate(labels):
        print(f" Document {idx} → Cluster {lab}")

    sys.exit(0)