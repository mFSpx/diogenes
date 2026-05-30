# DARWIN HAMMER — match 2938, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m587_s2.py (gen5)
# parent_b: hybrid_hybrid_hdc_serpentin_hybrid_hybrid_decisi_m162_s1.py (gen3)
# born: 2026-05-29T23:46:52Z

"""Hybrid algorithm combining DARWIN HAMMER — match 224, survivor 3
(PARENT A) and DARWIN HAMMER — match 162, survivor 1 (PARENT B).

Mathematical bridge:
- PARENT A supplies a Fisher information I_F(θ) that quantifies the sensitivity of a
  Gaussian beam intensity to the angular parameter θ.  This quantity is naturally a
  *precision* (inverse variance) in a Bayesian sense.
- PARENT B supplies morphology‑derived dynamical indices (righting‑time, sphericity,
  flatness) and a high‑dimensional morphology vector v(m).  The dot‑product
  similarity between two such vectors can be interpreted as a covariance between
  edge attributes.

The hybrid therefore defines an edge precision

    Π(e) = I_F(θ_e) * R(m_e)               (1)

where I_F is the Fisher precision from A and R(m) is the right‑ing‑time index from B.
Edge weight is taken as the inverse precision modulated by morphological similarity

    w(e) = (1 / Π(e)) * (1 - sim(v_i , v_j))   (2)

with sim the cosine similarity of the two endpoint morphology vectors.
These definitions allow the Fisher information to drive the statistical confidence of
graph edges while the morphology vectors encode geometric/physical compatibility.
The resulting weighted graph can be processed with a Prim‑style MST to obtain a
minimum‑cost routing tree that respects both information‑theoretic and physical
constraints.

The module implements three core hybrid operations:
1. `edge_precision` – computes Π(e) from beam parameters and morphology.
2. `hybrid_sketch` – a count‑min sketch that aggregates Fisher‑precision contributions
   of many packets together with morphology hashes.
3. `prim_mst_hybrid` – builds a minimum‑spanning tree using the hybrid edge weights
   defined above.
"""

import math
import random
import hashlib
import sys
import pathlib
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# PARENT A building blocks
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ (precision of the intensity)."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# PARENT B building blocks
# ----------------------------------------------------------------------
Vector = List[float]


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def random_vector(dim: int = 1024, seed: str | int | None = None) -> Vector:
    """Deterministic pseudo‑random vector of given dimension."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def morphology_vector(m: Morphology, dim: int = 1024) -> Vector:
    """Hash‑seeded vector whose scale is set by the morphology parameters."""
    seed_bytes = f"{m.length}{m.width}{m.height}{m.mass}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
    base = np.array(random_vector(dim, seed))
    scaling = np.array([m.length, m.width, m.height, m.mass])
    # pad scaling to match dim
    if dim > 4:
        scaling = np.pad(scaling, (0, dim - 4), constant_values=1.0)
    return (base * scaling).tolist()


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length+width)/(2*height)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Physical right‑ing time index derived from morphology."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ----------------------------------------------------------------------
# HYBRID OPERATIONS
# ----------------------------------------------------------------------
def edge_precision(theta: float, center: float, width: float,
                   morphology: Morphology) -> float:
    """
    Compute hybrid edge precision Π(e) = I_F(θ) * R(m).

    I_F   – Fisher information (precision) from the Gaussian beam.
    R(m)  – Right‑ing‑time index from morphology.
    """
    fisher = fisher_score(theta, center, width)
    righting = righting_time_index(morphology)
    return fisher * righting


def hybrid_sketch(
    packets: Iterable[Tuple[float, float, float, Morphology]],
    width: int = 64,
    depth: int = 4,
    dim: int = 1024,
) -> List[List[int]]:
    """
    Count‑min sketch that aggregates hybrid edge precisions.

    Each packet is a tuple (θ, center, width, Morphology).  For each packet we:
        1. Compute Π(e) via `edge_precision`.
        2. Hash the packet (using its numeric values) together with the sketch depth
           to obtain an index in each row.
        3. Increment the corresponding cell by int(Π(e) * 1e6) to keep integer counters.

    The sketch can be queried later for an approximate sum of precisions for a
    given (θ,center,width) pattern.
    """
    table = [[0] * width for _ in range(depth)]

    for theta, center, w, morph in packets:
        precision = edge_precision(theta, center, w, morph)
        # Scale to integer for the sketch
        increment = int(precision * 1e6)

        # Deterministic hashing per depth
        for d in range(depth):
            hasher = hashlib.sha256()
            hasher.update(f"{theta},{center},{w},{d}".encode("utf-8"))
            idx = int(hasher.hexdigest(), 16) % width
            table[d][idx] += increment

    return table


def prim_mst_hybrid(
    nodes: List[int],
    edges: List[Tuple[int, int, float, float, Morphology, Morphology]],
) -> List[Tuple[int, int, float]]:
    """
    Build a minimum‑spanning tree using hybrid edge weights.

    `edges` entries are:
        (u, v, theta, width, morph_u, morph_v)

    The weight w_uv is defined by equation (2):
        w_uv = (1 / Π_uv) * (1 - sim(v_u, v_v))

    where Π_uv = edge_precision(theta, theta, width, morph_u) (using the same
    theta for both endpoints for simplicity) and sim is the cosine similarity of
    the two morphology vectors.
    """
    # Pre‑compute morphology vectors for all nodes
    morph_vectors: Dict[int, Vector] = {}
    for node in nodes:
        # Placeholder morphology for isolated nodes (unit cube, mass=1)
        default_morph = Morphology(1.0, 1.0, 1.0, 1.0)
        morph_vectors[node] = morphology_vector(default_morph)

    # Update with provided morphologies (the first occurrence wins)
    for u, v, theta, width, mu, mv in edges:
        if u not in morph_vectors or morph_vectors[u] == morphology_vector(Morphology(1.0,1.0,1.0,1.0)):
            morph_vectors[u] = morphology_vector(mu)
        if v not in morph_vectors or morph_vectors[v] == morphology_vector(Morphology(1.0,1.0,1.0,1.0)):
            morph_vectors[v] = morphology_vector(mv)

    # Build adjacency list with hybrid weights
    adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for u, v, theta, width, mu, mv in edges:
        # Edge precision using mu (could also average mu,mv; we pick mu)
        precision = edge_precision(theta, theta, width, mu)
        if precision == 0:
            continue
        sim = cosine_similarity(morph_vectors[u], morph_vectors[v])
        weight = (1.0 / precision) * (1.0 - sim)
        adj[u].append((v, weight))
        adj[v].append((u, weight))

    # Prim's algorithm
    start = nodes[0]
    visited = {start}
    edges_mst: List[Tuple[int, int, float]] = []
    import heapq
    heap: List[Tuple[float, int, int]] = []
    for v, w in adj[start]:
        heapq.heappush(heap, (w, start, v))

    while heap and len(visited) < len(nodes):
        w, u, v = heapq.heappop(heap)
        if v in visited:
            continue
        visited.add(v)
        edges_mst.append((u, v, w))
        for nxt, w2 in adj[v]:
            if nxt not in visited:
                heapq.heappush(heap, (w2, v, nxt))

    return edges_mst


# ----------------------------------------------------------------------
# SMOKE TEST
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic packets
    packets = []
    for i in range(10):
        theta = random.uniform(-math.pi, math.pi)
        center = random.uniform(-1.0, 1.0)
        width = random.uniform(0.1, 0.5)
        morph = Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(0.1, 5.0),
        )
        packets.append((theta, center, width, morph))

    sketch = hybrid_sketch(packets)
    print("Sketch dimensions:", len(sketch), "x", len(sketch[0]))

    # Build a tiny graph for MST
    node_ids = list(range(5))
    edges = []
    for i in range(4):
        u = i
        v = i + 1
        theta = random.uniform(-math.pi, math.pi)
        width = random.uniform(0.1, 0.5)
        mu = Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(0.1, 5.0),
        )
        mv = Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(0.1, 5.0),
        )
        edges.append((u, v, theta, width, mu, mv))

    mst = prim_mst_hybrid(node_ids, edges)
    print("MST edges (u, v, weight):")
    for e in mst:
        print(e)