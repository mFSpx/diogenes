# DARWIN HAMMER — match 4087, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1995_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_shap_a_hybrid_hybrid_hybrid_m2345_s2.py (gen6)
# born: 2026-05-29T23:53:27Z

"""Hybrid Algorithm Fusion of DARWIN HAMMER Parents A and B.

Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1995_s0.py) contributes:
- Morphological indices (sphericity, flatness) and MinHash signature generation.
- ProceduralSlot data structure.

Parent B (hybrid_hybrid_hybrid_shap_a_hybrid_hybrid_hybrid_m2345_s2.py) contributes:
- Graph construction from master vectors, Ollivier‑Ricci curvature approximation,
  and SHAP‑like attribution of curvature to node features.

Mathematical Bridge:
1. Morphological indices are assembled into a feature vector that becomes a
   *master vector* for each entity.
2. Master vectors define a similarity graph; edge weights are Euclidean distances.
3. Curvature on the graph is approximated by the inverse of the average neighbor
   distance, mirroring Ollivier‑Ricci curvature.
4. SHAP‑like attributions are computed by linearly distributing a node's curvature
   over its feature components (sphericity, flatness, mass). These attributions
   are turned into textual tokens.
5. Tokens are fed to the MinHash routine from Parent A, producing a signature that
   encodes both morphological shape and curvature attribution.
6. The signature determines a ternary offset for the ProceduralSlot, completing
   the fusion of both parents into a single unified system.
"""

import sys
import math
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Set

import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Core mathematical primitives (Parent A)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        hashlib.blake2b(data, digest_size=8).digest(), "big"
    )


def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    """MinHash signature of a token list (Parent A)."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2 ** 63 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


# ----------------------------------------------------------------------
# Graph utilities (Parent B)
# ----------------------------------------------------------------------
Node = int
Graph = Dict[Node, Set[Node]]


def build_graph(master_vectors: List[np.ndarray], eps: float = 1e-6) -> Graph:
    """Construct an undirected graph where an edge exists if vectors are
    closer than the median pairwise distance."""
    n = len(master_vectors)
    if n == 0:
        return {}
    # Compute all pairwise distances
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(master_vectors[i] - master_vectors[j])
            dists.append(d)
    median_dist = np.median(dists) if dists else eps

    graph: Graph = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(master_vectors[i] - master_vectors[j])
            if d <= median_dist + eps:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def approximate_ollivier_ricci_curvature(
    graph: Graph, master_vectors: List[np.ndarray], eps: float = 1e-12
) -> Dict[Node, float]:
    """A lightweight Ollivier‑Ricci curvature approximation:
    curvature(node) = 1 - (average neighbor distance / max_distance).
    """
    n = len(master_vectors)
    if n == 0:
        return {}

    # Global max distance for normalization
    max_dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(master_vectors[i] - master_vectors[j])
            if d > max_dist:
                max_dist = d
    max_dist = max(max_dist, eps)

    curvature: Dict[Node, float] = {}
    for node, neighbors in graph.items():
        if not neighbors:
            curvature[node] = 0.0
            continue
        avg_dist = np.mean(
            [np.linalg.norm(master_vectors[node] - master_vectors[nbr]) for nbr in neighbors]
        )
        curvature[node] = 1.0 - (avg_dist / max_dist)
    return curvature


# ----------------------------------------------------------------------
# SHAP‑like attribution (bridge between Parent B and Parent A)
# ----------------------------------------------------------------------
def shap_like_attribution(
    features: np.ndarray, curvature: float, eps: float = 1e-12
) -> np.ndarray:
    """Distribute curvature proportionally to feature magnitudes.
    Returns an attribution vector of the same shape as *features*."""
    if features.size == 0:
        return np.array([])
    total = np.sum(np.abs(features)) + eps
    contributions = (np.abs(features) / total) * curvature
    return contributions


# ----------------------------------------------------------------------
# Fusion pipeline (three demonstrative functions)
# ----------------------------------------------------------------------
def compute_master_vectors(morphologies: List[Morphology]) -> List[np.ndarray]:
    """Transform each Morphology into a master vector consisting of
    [sphericity, flatness, mass]."""
    vectors: List[np.ndarray] = []
    for m in morphologies:
        sph = sphericity_index(m.length, m.width, m.height)
        flt = flatness_index(m.length, m.width, m.height)
        vec = np.array([sph, flt, m.mass], dtype=float)
        vectors.append(vec)
    return vectors


def evaluate_node_curvature(
    master_vectors: List[np.ndarray],
) -> Tuple[Graph, Dict[Node, float]]:
    """Build the similarity graph and compute curvature for each node."""
    graph = build_graph(master_vectors)
    curvature = approximate_ollivier_ricci_curvature(graph, master_vectors)
    return graph, curvature


def generate_procedural_slots(
    morphologies: List[Morphology],
    base_name: str = "Entity",
) -> List[ProceduralSlot]:
    """
    Full hybrid generation:
    1. Compute master vectors from morphology.
    2. Build graph and obtain curvature per node.
    3. Compute SHAP‑like attributions and turn them into tokens.
    4. Produce a MinHash signature from the tokens.
    5. Derive a ternary offset from the signature and create ProceduralSlot objects.
    """
    master_vectors = compute_master_vectors(morphologies)
    _, curvature_map = evaluate_node_curvature(master_vectors)

    slots: List[ProceduralSlot] = []
    for idx, (morph, vec) in enumerate(zip(morphologies, master_vectors)):
        curvature = curvature_map.get(idx, 0.0)

        # SHAP‑like attribution
        attrib = shap_like_attribution(vec, curvature)

        # Token creation (feature name + attribution value)
        tokens = [
            f"sph:{attrib[0]:.6f}",
            f"flt:{attrib[1]:.6f}",
            f"mass:{attrib[2]:.6f}",
            f"curv:{curvature:.6f}",
        ]

        # MinHash signature
        signature = minhash_signature(tokens, k=64)

        # Derive ternary offset from signature (e.g., sum of first 8 hashes)
        offset = int(sum(signature[:8])) % 3

        # UUID deterministic from index and current UTC timestamp
        ts = datetime.now(timezone.utc).isoformat()
        raw_uuid = f"{idx}-{ts}"
        uuid = hashlib.blake2b(raw_uuid.encode(), digest_size=16).hexdigest()

        slot = ProceduralSlot(
            slot_index=idx,
            name=f"{base_name}_{idx}",
            alias=f"{base_name.lower()}_{idx}",
            persona="HybridEntity",
            uuid=uuid,
            ternary_offset=offset,
        )
        slots.append(slot)

    return slots


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a modest set of random morphologies
    random.seed(42)
    morphs = [
        Morphology(
            length=random.uniform(0.5, 5.0),
            width=random.uniform(0.5, 5.0),
            height=random.uniform(0.5, 5.0),
            mass=random.uniform(1.0, 10.0),
        )
        for _ in range(7)
    ]

    slots = generate_procedural_slots(morphs, base_name="HybridTest")
    for slot in slots:
        print(slot.as_dict())
    print("Fusion test completed successfully.")