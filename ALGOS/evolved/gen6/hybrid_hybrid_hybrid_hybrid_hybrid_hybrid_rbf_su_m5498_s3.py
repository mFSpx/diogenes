# DARWIN HAMMER — match 5498, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_capyba_m1423_s1.py (gen5)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s7.py (gen3)
# born: 2026-05-30T00:02:28Z

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Hashable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Linguistic categories for stylometry
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only".split()
    ),
}


# ----------------------------------------------------------------------
# Basic mathematical primitives
# ----------------------------------------------------------------------
Node = Hashable
FeatureVec = Sequence[float]


def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


# ----------------------------------------------------------------------
# Perceptual hash utilities (used as a second similarity channel)
# ----------------------------------------------------------------------
def compute_phash(values: Sequence[float]) -> int:
    """
    64‑bit perceptual hash: 1‑bit per value relative to the median.
    If more than 64 values are supplied the excess are ignored.
    """
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers interpreted as bit strings."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Stylometry feature extraction
# ----------------------------------------------------------------------
def compute_stylometry_features(text: str) -> np.ndarray:
    """
    Return a normalized frequency vector for each functional‑word category.
    The vector length equals ``len(FUNCTION_CATS)``.
    """
    tokens = text.lower().split()
    total = len(tokens) or 1  # avoid division by zero
    freqs = [
        sum(1 for w in tokens if w in words) / total
        for words in FUNCTION_CATS.values()
    ]
    return np.array(freqs, dtype=np.float64)


# ----------------------------------------------------------------------
# Epistemic certainty – a measure of confidence in a node’s feature vector
# ----------------------------------------------------------------------
def epistemic_certainty(vec: FeatureVec) -> float:
    """
    Higher certainty for low‑variance vectors.
    Returns a value in (0, 1].
    """
    var = np.var(vec)
    return 1.0 / (1.0 + var)


# ----------------------------------------------------------------------
# Social interaction weight – simplistic degree‑based influence
# ----------------------------------------------------------------------
def social_interaction_weight(node: Node, adjacency: Dict[Node, List[Node]]) -> float:
    """
    Nodes with many connections are assumed to be socially influential.
    Normalised to [0, 1].
    """
    max_deg = max((len(neigh) for neigh in adjacency.values()), default=1)
    deg = len(adjacency.get(node, []))
    return deg / max_deg if max_deg else 0.0


# ----------------------------------------------------------------------
# Predator‑evasion factor – random but reproducible per node
# ----------------------------------------------------------------------
def predator_evasion_factor(node: Node) -> float:
    """
    A deterministic pseudo‑random factor in [0, 1] derived from the node’s hash.
    Higher values indicate better evasion capability.
    """
    h = int(hashlib.sha256(str(node).encode()).hexdigest(), 16)
    random.seed(h)
    return random.random()


# ----------------------------------------------------------------------
# Edge weight synthesis
# ----------------------------------------------------------------------
def synthesize_edge_weight(
    i: Node,
    j: Node,
    features: Dict[Node, np.ndarray],
    phashes: Dict[Node, int],
    positions: Dict[Node, Tuple[float, float]],
    adjacency: Dict[Node, List[Node]],
    epsilon: float = 1.0,
) -> float:
    """
    Combine multiple modalities into a single scalar weight:
        • Physical Euclidean distance (d_phys)
        • RBF similarity (s_rbf) – higher similarity should lower cost
        • Hamming similarity of perceptual hashes (s_hash)
        • Epistemic certainty of both nodes (c_i, c_j)
        • Social interaction (soc_i, soc_j)
        • Predator‑evasion (p_i, p_j)

    The final cost is a weighted sum where each term is normalised to [0,1].
    """
    # 1. Physical distance (normalised later)
    xi, yi = positions[i]
    xj, yj = positions[j]
    d_phys = euclidean((xi, yi), (xj, yj))

    # 2. RBF similarity
    s_rbf = gaussian(euclidean(features[i], features[j]), epsilon)

    # 3. Hash similarity (inverse Hamming distance)
    ham = hamming_distance(phashes[i], phashes[j])
    max_ham = max(len(bin(phashes[i])) - 2, len(bin(phashes[j])) - 2, 1)
    s_hash = 1.0 - ham / max_ham

    # 4. Epistemic certainty
    c_i = epistemic_certainty(features[i])
    c_j = epistemic_certainty(features[j])
    c_avg = (c_i + c_j) / 2.0

    # 5. Social interaction
    soc_i = social_interaction_weight(i, adjacency)
    soc_j = social_interaction_weight(j, adjacency)
    soc_avg = (soc_i + soc_j) / 2.0

    # 6. Predator evasion
    p_i = predator_evasion_factor(i)
    p_j = predator_evasion_factor(j)
    p_avg = (p_i + p_j) / 2.0

    # Normalise physical distance to [0,1] using a simple max‑distance heuristic
    # (the caller ensures a reasonable scale by providing positions in a bounded box)
    d_norm = d_phys / (d_phys + 1.0)  # maps (0,∞) → (0,1)

    # Combine: lower similarity and higher distance increase cost.
    # We invert similarity terms so that higher similarity reduces the cost.
    cost = (
        0.30 * d_norm
        + 0.20 * (1.0 - s_rbf)          # RBF similarity
        + 0.15 * (1.0 - s_hash)        # hash similarity
        + 0.10 * (1.0 - c_avg)         # epistemic certainty
        + 0.15 * (1.0 - soc_avg)       # social interaction
        + 0.10 * (1.0 - p_avg)         # predator evasion
    )
    return cost


# ----------------------------------------------------------------------
# Minimum‑cost tree construction (Prim's algorithm)
# ----------------------------------------------------------------------
def prim_mst(
    nodes: List[Node],
    edge_weight_func,
) -> Dict[Node, List[Node]]:
    """
    Build a minimum‑spanning tree over ``nodes`` using the supplied
    ``edge_weight_func(i, j)`` callable. Returns an adjacency list representation.
    """
    if not nodes:
        return {}

    # Initialise structures
    in_tree = {nodes[0]}
    not_in_tree = set(nodes[1:])
    adjacency: Dict[Node, List[Node]] = defaultdict(list)

    # Pre‑compute edge weights lazily
    while not_in_tree:
        best_edge = None
        best_cost = math.inf
        for u in in_tree:
            for v in not_in_tree:
                cost = edge_weight_func(u, v)
                if cost < best_cost:
                    best_cost = cost
                    best_edge = (u, v)
        if best_edge is None:  # disconnected graph (should not happen)
            break
        u, v = best_edge
        adjacency[u].append(v)
        adjacency[v].append(u)
        in_tree.add(v)
        not_in_tree.remove(v)

    return dict(adjacency)


# ----------------------------------------------------------------------
# High‑level update routine
# ----------------------------------------------------------------------
def update_tree_structure(
    nodes: List[Node],
    texts: Dict[Node, str],
    positions: Dict[Node, Tuple[float, float]],
    epsilon: float = 1.0,
) -> Dict[Node, List[Node]]:
    """
    1. Extract stylometry vectors for each node.
    2. Compute perceptual hashes.
    3. Build a complete graph where edge weights are a deep fusion of
       physical, linguistic, epistemic, social and predator‑evasion signals.
    4. Return the minimum‑cost spanning tree as an adjacency list.
    """
    # 1. Feature extraction
    features: Dict[Node, np.ndarray] = {
        n: compute_stylometry_features(texts[n]) for n in nodes
    }

    # 2. Perceptual hashes
    phashes: Dict[Node, int] = {n: compute_phash(vec) for n, vec in features.items()}

    # 3. Temporary adjacency needed for social weight (starts empty)
    temp_adj: Dict[Node, List[Node]] = defaultdict(list)

    # 4. Edge weight closure capturing the current state
    def edge_weight(i: Node, j: Node) -> float:
        return synthesize_edge_weight(
            i,
            j,
            features,
            phashes,
            positions,
            temp_adj,
            epsilon,
        )

    # 5. Build MST – during construction we progressively fill ``temp_adj``
    #    so that social interaction weights evolve with the growing tree.
    mst_adj = prim_mst(nodes, edge_weight)

    return mst_adj


# ----------------------------------------------------------------------
# Demonstration harness
# ----------------------------------------------------------------------
def _random_position() -> Tuple[float, float]:
    """Generate a random 2‑D position inside a unit square."""
    return (random.random(), random.random())


def main() -> None:
    # Example node set
    nodes = [f"user_{i}" for i in range(1, 8)]

    # Synthetic texts (in a real setting these would be actual documents)
    sample_texts = [
        "I think therefore I am.",
        "The quick brown fox jumps over the lazy dog.",
        "Data science combines statistics, programming, and domain knowledge.",
        "When the sun sets, the shadows grow longer.",
        "Artificial intelligence is reshaping many industries.",
        "She sells seashells by the seashore.",
        "In the middle of difficulty lies opportunity."
    ]
    texts = {n: sample_texts[i % len(sample_texts)] for i, n in enumerate(nodes)}

    # Random spatial embedding
    positions = {n: _random_position() for n in nodes}

    # Build the fused minimum‑cost tree
    tree = update_tree_structure(nodes, texts, positions, epsilon=1.2)

    # Pretty‑print the result
    for parent, children in tree.items():
        print(f"{parent} -> {children}")


if __name__ == "__main__":
    main()