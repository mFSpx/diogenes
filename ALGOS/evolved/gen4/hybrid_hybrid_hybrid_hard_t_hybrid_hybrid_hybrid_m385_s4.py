# DARWIN HAMMER — match 385, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_hard_truth_ma_kan_m27_s4.py (gen2)
# parent_b: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s4.py (gen3)
# born: 2026-05-29T23:28:39Z

import hashlib
import math
import random
import re
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# ----------------------------------------------------------------------
# Text processing utilities
# ----------------------------------------------------------------------
WORD_RE = re.compile(r"\b[a-zA-Z]+\b")

def words(text: Optional[str]) -> List[str]:
    """Return a list of lower‑cased alphabetic tokens."""
    if not text:
        return []
    return WORD_RE.findall(text.lower())

# ----------------------------------------------------------------------
# Function‑word categories (stylometry)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
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
        "no not never none neither cannot cant wont dont didnt isnt arent wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}

PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

# ----------------------------------------------------------------------
# Stylometry feature extraction
# ----------------------------------------------------------------------
def lsm_vector(text: str) -> Dict[str, float]:
    """Return normalized frequencies of function‑word categories."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)

    result: Dict[str, float] = {}
    for cat, vocab in FUNCTION_CATS.items():
        # sum of occurrences of words belonging to the category
        cat_count = sum(cnt[w] for w in vocab)
        result[cat] = cat_count / total
    return result


def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Utility to make an array exactly `target_len` long."""
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] > target_len:
        return arr[:target_len]
    # pad with zeros
    pad_width = target_len - arr.shape[0]
    return np.concatenate([arr, np.zeros(pad_width, dtype=arr.dtype)])


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """
    Produce a fixed‑size numeric representation of a text.

    The vector consists of:
      * 7 handcrafted scalar features (length, avg‑word‑len, line density, punctuation ratios, uppercase ratio)
      * LSM category frequencies (len(FUNCTION_CATS) = 8)
      * Zero padding to reach `dim` dimensions.

    Parameters
    ----------
    text: str
        Input text.
    dim: int, optional
        Desired dimensionality (default 96). Must be >= 15.

    Returns
    -------
    np.ndarray
        1‑D array of length `dim`.
    """
    if dim < 15:
        raise ValueError("dim must be at least 15 to hold all base features")

    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))

    # handcrafted scalars
    handcrafted = np.array(
        [
            total_words / 500.0,
            sum(len(w) for w in ws) / total_words / 12.0,
            (text.count("\n") + 1) / 200.0,
            sum(text.count(p) for p in "!?") / total_chars,
            sum(text.count(p) for p in ";:") / total_chars,
            sum(text.count(p) for p in "-—") / total_chars,
            sum(1 for ch in text if ch.isupper()) / total_chars,
        ],
        dtype=float,
    )

    # LSM frequencies (ordered by FUNCTION_CATS keys)
    lsm_vals = np.array([lsm_vector(text)[cat] for cat in sorted(FUNCTION_CATS)], dtype=float)

    base_vec = np.concatenate([handcrafted, lsm_vals])
    return _pad_or_truncate(base_vec, dim)


# ----------------------------------------------------------------------
# Tree structures and Bayesian VRAM estimation
# ----------------------------------------------------------------------
@dataclass
class TreeNode:
    name: str
    size: int                                 # VRAM footprint in MB
    prior_probability: float                 # a priori belief that this node will be needed
    style_profile: np.ndarray = field(default_factory=lambda: np.zeros(96))  # 96‑dim stylometry prototype


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], b[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adjacency : dict
        adjacency list of the undirected tree
    edge_lengths : dict
        Euclidean length of each directed edge
    node_distances : dict
        distance from the root to each node
    """
    adjacency: Dict[str, List[str]] = {node: [] for node in nodes}
    edge_lengths: Dict[Tuple[str, str], float] = {}
    node_distances: Dict[str, float] = {root: 0.0}

    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
        d = length(nodes[u], nodes[v])
        edge_lengths[(u, v)] = d
        edge_lengths[(v, u)] = d

    # BFS to compute root‑to‑node distances
    q = deque([root])
    visited = {root}
    while q:
        cur = q.popleft()
        for nb in adjacency[cur]:
            if nb not in visited:
                node_distances[nb] = node_distances[cur] + edge_lengths[(cur, nb)]
                visited.add(nb)
                q.append(nb)

    return adjacency, edge_lengths, node_distances


def bayesian_update(prior: float, likelihood: float, eps: float = 1e-12) -> float:
    """
    Perform a numerically stable Bayesian update.

    posterior = (prior * likelihood) / (prior * likelihood + (1‑prior)*(1‑likelihood))

    eps prevents division by zero.
    """
    num = prior * likelihood
    den = num + (1.0 - prior) * (1.0 - likelihood)
    if den < eps:
        return 0.0
    return num / den


def likelihood_from_stylometry(node: TreeNode, text_vec: np.ndarray) -> float:
    """
    Derive a likelihood that the given text will trigger `node`.

    We treat the node's `style_profile` as a prototype vector and compute
    cosine similarity with the observed `text_vec`. The similarity is then
    squashed into [0,1] with a sigmoid.
    """
    if node.style_profile.shape != text_vec.shape:
        raise ValueError("style_profile and text_vec must have the same shape")

    # cosine similarity
    dot = float(np.dot(node.style_profile, text_vec))
    norm_prod = float(np.linalg.norm(node.style_profile) * np.linalg.norm(text_vec))
    cos_sim = dot / (norm_prod + 1e-12)

    # sigmoid mapping (steeper than 1 to accentuate differences)
    return 1.0 / (1.0 + math.exp(-5.0 * cos_sim))


def expected_vram(
    nodes: List[TreeNode],
    text_vec: np.ndarray,
    node_distances: Optional[Dict[str, float]] = None,
    distance_weight: float = 0.01,
) -> float:
    """
    Compute the expected VRAM consumption.

    For each node we:
      1. Compute a likelihood from stylometry.
      2. Perform a Bayesian update of the prior.
      3. Multiply by the node size.
      4. Optionally down‑weight by its distance from the root
         (farther nodes are less likely to be needed early).

    Parameters
    ----------
    nodes : list[TreeNode]
        All candidate nodes.
    text_vec : np.ndarray
        Stylometry vector of the current workload.
    node_distances : dict | None
        Mapping name → root distance (if None, distance weighting is omitted).
    distance_weight : float
        Linear factor applied to distances (default 0.01).

    Returns
    -------
    float
        Expected VRAM usage in the same units as `node.size`.
    """
    total = 0.0
    for node in nodes:
        lik = likelihood_from_stylometry(node, text_vec)
        post = bayesian_update(node.prior_probability, lik)
        weight = 1.0
        if node_distances is not None and node.name in node_distances:
            weight = max(0.0, 1.0 - distance_weight * node_distances[node.name])
        total += node.size * post * weight
    return total


# ----------------------------------------------------------------------
# Example usage (self‑test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample text and its stylometry vector
    sample_text = "This is a sample text for testing the stylometry features."
    vec = stylometry_features(sample_text, dim=96)

    # Create a tiny tree
    nodes_info = {
        "root": (0.0, 0.0),
        "A": (1.0, 2.0),
        "B": (3.0, 1.0),
        "C": (4.0, 4.0),
    }
    edges = [("root", "A"), ("root", "B"), ("A", "C")]
    adjacency, edge_lens, distances = tree_metrics(nodes_info, edges, root="root")

    # Initialise TreeNode objects with random style prototypes
    rng = np.random.default_rng(42)
    node_objs = [
        TreeNode(name="root", size=150, prior_probability=0.6, style_profile=rng.normal(size=96)),
        TreeNode(name="A", size=200, prior_probability=0.4, style_profile=rng.normal(size=96)),
        TreeNode(name="B", size=120, prior_probability=0.3, style_profile=rng.normal(size=96)),
        TreeNode(name="C", size=80, prior_probability=0.2, style_profile=rng.normal(size=96)),
    ]

    ev = expected_vram(node_objs, vec, node_distances=distances, distance_weight=0.02)
    print(f"Expected VRAM consumption: {ev:.2f} MB")