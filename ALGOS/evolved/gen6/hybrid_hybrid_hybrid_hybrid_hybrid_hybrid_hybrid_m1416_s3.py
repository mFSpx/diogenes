# DARWIN HAMMER — match 1416, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_krampu_m1196_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m482_s0.py (gen5)
# born: 2026-05-29T23:36:11Z

"""
Hybrid Algorithm combining:
- Parent A: Text feature extraction, pheromone decay, Ollivier‑Ricci curvature and sheaf aggregation.
- Parent B: Morphology‑weighted Gini impurity and Hoeffding‑bound driven decision tree splitting.

Mathematical Bridge
-------------------
The bridge is built by interpreting each feature dimension (or graph node) produced by Parent A as a
*Morphology* object.  The curvature computed from the master‑vector graph supplies a
`length`, the normalized pheromone signal supplies a `mass`, the entropy supplies a `height`,
and the normalized feature magnitude supplies a `width`.  These four scalars are then used
as the weighting factors in the Gini impurity formula of Parent B.  The resulting
*health‑informed* Gini gain is evaluated with the Hoeffding bound to decide whether a node
in the decision tree should be split.  Thus the sheaf‑based aggregation of pheromones feeds
directly into a statistically‑controlled tree growth, fusing the two topologies into a
single, time‑aware, uncertainty‑quantified classifier.

The module implements:
1. Text → master vector → pheromone entries.
2. Ollivier‑Ricci curvature on the vector graph.
3. Morphology construction from curvature, pheromone, entropy and magnitude.
4. Morphology‑weighted Gini impurity and Hoeffding‑bound split decision.
5. A minimal decision‑tree learner that operates on the hybrid representation.
"""

import sys
import math
import random
import pathlib
import numpy as np
from collections import Counter
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# ----------------------------------------------------------------------
# Parent A – Feature extraction, pheromone handling, curvature
# ----------------------------------------------------------------------

WIDTH = 64          # dimension of master vector
HALF_LIFE_BASE = 10.0   # base half‑life in seconds


class PheromoneEntry:
    def __init__(self, feature: str, value: float, half_life: float):
        self.feature = feature
        self.value = value
        self.half_life = half_life
        self.signal = value  # current signal strength


def _master_vector(text: str, width: int = WIDTH) -> np.ndarray:
    """Hash each character and spread bits across a fixed‑size vector."""
    vec = np.zeros(width, dtype=float)
    for i, ch in enumerate(text):
        h = hash(ch) & 0xffffffff
        idx = (i * 7) % width
        vec[idx] ^= (h & 0xff) / 255.0
        vec[(idx + 1) % width] ^= ((h >> 8) & 0xff) / 255.0
    # normalise to unit length
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _entropy(tokens: List[str]) -> float:
    """Shannon entropy of token distribution."""
    total = len(tokens)
    if total == 0:
        return 0.0
    counts = Counter(tokens)
    ent = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return ent


def extract_features(text: str) -> Dict[str, Any]:
    """Return a dictionary with token list, entropy, master vector and magnitude."""
    tokens = text.lower().split()
    ent = _entropy(tokens)
    vec = _master_vector(text)
    mag = np.linalg.norm(vec)
    return {
        "tokens": tokens,
        "entropy": ent,
        "vector": vec,
        "magnitude": mag,
    }


def _half_life_from_entropy(entropy: float) -> float:
    """Higher entropy → slower decay (longer half‑life)."""
    return HALF_LIFE_BASE * (1.0 + entropy)


def initialise_pheromones(features: Dict[str, Any]) -> List[PheromoneEntry]:
    """Create one pheromone entry per distinct token."""
    token_counts = Counter(features["tokens"])
    total = sum(token_counts.values())
    entries = []
    for token, cnt in token_counts.items():
        value = cnt / total                     # relative frequency
        hl = _half_life_from_entropy(features["entropy"])
        entries.append(PheromoneEntry(token, value, hl))
    return entries


def _ollivier_ricci_curvature(vectors: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature for each dimension.
    For each dimension i we compute the average cosine similarity between the
    vector slice centred at i and its k nearest neighbours (circular indexing).
    The curvature is taken as 1 - average_distance, yielding values in [-1,1].
    """
    dim = vectors.shape[0]
    curv = np.zeros(dim, dtype=float)
    for i in range(dim):
        neighbours = [(i + offset) % dim for offset in range(-k, k + 1) if offset != 0]
        sims = [np.dot(vectors[i], vectors[j]) for j in neighbours]
        avg_sim = np.mean(sims) if sims else 0.0
        curv[i] = 1.0 - avg_sim
    return curv


def compute_curvature(features: Dict[str, Any]) -> np.ndarray:
    """Wrapper that extracts the master vector and returns curvature per dimension."""
    vec = features["vector"]
    return _ollivier_ricci_curvature(vec)


# ----------------------------------------------------------------------
# Parent B – Morphology, weighted Gini and Hoeffding split
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float   # curvature‑derived
    width: float    # normalized feature magnitude
    height: float   # entropy
    mass: float     # pheromone signal strength


def morphology_from_dim(
    dim_idx: int,
    curvature: np.ndarray,
    features: Dict[str, Any],
    pheromone_map: Dict[str, float],
) -> Morphology:
    """
    Build a Morphology for a single dimension (node) of the master vector.
    - length  : curvature value (already in [-1,1], shifted to [0,2] for positivity)
    - width   : normalized magnitude of the master vector component
    - height  : global entropy (same for all dimensions)
    - mass    : aggregated pheromone signal of tokens whose hash lands on this dimension
    """
    length = curvature[dim_idx] + 1.0          # shift to [0,2]
    width = abs(features["vector"][dim_idx])  # magnitude of component (>=0)
    height = features["entropy"]
    # accumulate pheromone mass from tokens that contributed to this index
    mass = 0.0
    for token, entry in pheromone_map.items():
        # reproduce the index mapping used in _master_vector
        idx = (hash(token) & 0xffffffff) % WIDTH
        if idx == dim_idx:
            mass += entry.signal
    return Morphology(length=length, width=width, height=height, mass=mass)


def weighted_gini(groups: List[List[int]], morphologies: List[Morphology]) -> float:
    """
    Compute Gini impurity weighted by the `mass` of each morphology.
    `groups` is a list of class label lists (e.g. [[0,0,1],[1,1,0]]).
    The weight of an element i is the mass of its corresponding morphology.
    """
    # flatten groups while keeping track of which morphology index each element belongs to
    total_weight = 0.0
    class_weight: Dict[int, float] = {}
    for grp_idx, grp in enumerate(groups):
        for label in grp:
            w = morphologies[grp_idx].mass
            total_weight += w
            class_weight[label] = class_weight.get(label, 0.0) + w

    if total_weight == 0.0:
        return 0.0
    impurity = 1.0 - sum((w / total_weight) ** 2 for w in class_weight.values())
    return impurity


def hoeffding_bound(R: float, delta: float, n: int) -> float:
    """Hoeffding bound ε = sqrt(R² * ln(1/δ) / (2n)). R is range of the random variable."""
    if n == 0:
        return float('inf')
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2 * n))


# ----------------------------------------------------------------------
# Hybrid Decision Tree using the combined representation
# ----------------------------------------------------------------------


class HybridNode:
    """A node in the hybrid decision tree."""
    def __init__(self, depth: int = 0, max_depth: int = 5):
        self.depth = depth
        self.max_depth = max_depth
        self.split_dim: int | None = None
        self.threshold: float | None = None
        self.left: "HybridNode" | None = None
        self.right: "HybridNode" | None = None
        self.is_leaf: bool = False
        self.prediction: int | None = None

    def fit(
        self,
        dim_indices: List[int],
        labels: List[int],
        morphologies: List[Morphology],
        delta: float = 0.05,
    ):
        """Recursively grow the tree using Hoeffding‑bound controlled splits."""
        # base cases
        if self.depth >= self.max_depth or len(set(labels)) == 1:
            self.is_leaf = True
            self.prediction = int(round(np.mean(labels)))  # majority vote
            return

        # try every dimension as a candidate split
        best_gain = -float('inf')
        best_dim = None
        best_thresh = None
        best_groups = None

        for dim in dim_indices:
            # use the width (absolute component) as the split attribute
            values = [abs(m.width) for m in morphologies]
            thresh = np.median([values[i] for i in range(len(values)) if dim == dim_indices[i]])
            left_idx = [i for i, v in enumerate(values) if v <= thresh]
            right_idx = [i for i, v in enumerate(values) if v > thresh]

            if not left_idx or not right_idx:
                continue

            left_labels = [labels[i] for i in left_idx]
            right_labels = [labels[i] for i in right_idx]

            # weighted Gini before split
            parent_gini = weighted_gini([list(range(len(labels)))], morphologies)

            # weighted Gini after split
            left_morph = [morphologies[i] for i in left_idx]
            right_morph = [morphologies[i] for i in right_idx]
            left_gini = weighted_gini([list(range(len(left_labels)))], left_morph)
            right_gini = weighted_gini([list(range(len(right_labels)))], right_morph)

            n = len(labels)
            gain = parent_gini - (len(left_labels) / n) * left_gini - (len(right_labels) / n) * right_gini

            if gain > best_gain:
                best_gain = gain
                best_dim = dim
                best_thresh = thresh
                best_groups = (left_idx, right_idx)

        if best_dim is None:
            # cannot find a useful split
            self.is_leaf = True
            self.prediction = int(round(np.mean(labels)))
            return

        # Hoeffding bound test
        epsilon = hoeffding_bound(R=1.0, delta=delta, n=len(labels))
        if best_gain <= epsilon:
            # not enough statistical evidence to split
            self.is_leaf = True
            self.prediction = int(round(np.mean(labels)))
            return

        # Accept split
        self.split_dim = best_dim
        self.threshold = best_thresh
        left_idx, right_idx = best_groups

        self.left = HybridNode(depth=self.depth + 1, max_depth=self.max_depth)
        self.right = HybridNode(depth=self.depth + 1, max_depth=self.max_depth)

        left_morph = [morphologies[i] for i in left_idx]
        right_morph = [morphologies[i] for i in right_idx]
        left_labels = [labels[i] for i in left_idx]
        right_labels = [labels[i] for i in right_idx]

        self.left.fit([dim_indices[i] for i in left_idx], left_labels, left_morph, delta)
        self.right.fit([dim_indices[i] for i in right_idx], right_labels, right_morph, delta)

    def predict_one(self, morph: Morphology) -> int:
        if self.is_leaf:
            return self.prediction if self.prediction is not None else 0
        attr = abs(morph.width)
        if attr <= self.threshold:
            return self.left.predict_one(morph) if self.left else 0
        else:
            return self.right.predict_one(morph) if self.right else 0

    def predict(self, morphologies: List[Morphology]) -> List[int]:
        return [self.predict_one(m) for m in morphologies]


# ----------------------------------------------------------------------
# Public API – three demonstration functions
# ----------------------------------------------------------------------


def hybrid_feature_pipeline(text: str) -> Tuple[Dict[str, Any], List[PheromoneEntry], np.ndarray]:
    """
    Executes Parent A steps:
    1. Extract features.
    2. Initialise pheromones.
    3. Compute curvature.
    Returns the three intermediate objects.
    """
    feats = extract_features(text)
    pher = initialise_pheromones(feats)
    curv = compute_curvature(feats)
    return feats, pher, curv


def build_morphologies(
    features: Dict[str, Any],
    curvature: np.ndarray,
    pheromones: List[PheromoneEntry],
) -> List[Morphology]:
    """
    Constructs a Morphology per dimension of the master vector by fusing
    curvature, pheromone mass and entropy (Parent A → Parent B bridge).
    """
    # map token → pheromone entry for fast lookup
    pher_map = {p.feature: p for p in pheromones}
    morphs = [
        morphology_from_dim(i, curvature, features, pher_map)
        for i in range(WIDTH)
    ]
    return morphs


def train_hybrid_tree(texts: List[str], labels: List[int]) -> HybridNode:
    """
    Trains a hybrid decision tree on a list of texts and binary labels.
    The tree uses morphology‑weighted Gini impurity and Hoeffding‑bound splits.
    """
    # aggregate morphologies for each text
    all_morphs: List[Morphology] = []
    for txt in texts:
        feats, pher, curv = hybrid_feature_pipeline(txt)
        morphs = build_morphologies(feats, curv, pher)
        # for simplicity we collapse the WIDTH‑dimensional morph list into a single
        # representative morphology by averaging its fields.
        avg = Morphology(
            length=np.mean([m.length for m in morphs]),
            width=np.mean([m.width for m in morphs]),
            height=np.mean([m.height for m in morphs]),
            mass=np.mean([m.mass for m in morphs]),
        )
        all_morphs.append(avg)

    # indices are dummy because we treat each averaged morphology as a single dimension
    dim_indices = list(range(len(all_morphs)))
    root = HybridNode(depth=0, max_depth=4)
    root