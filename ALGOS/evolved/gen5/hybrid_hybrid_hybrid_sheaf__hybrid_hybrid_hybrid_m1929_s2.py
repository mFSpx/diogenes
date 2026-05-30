# DARWIN HAMMER — match 1929, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_sheaf_cohomol_hybrid_shannon_entro_m6_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m995_s1.py (gen4)
# born: 2026-05-29T23:39:58Z

import numpy as np
import hashlib
import json
import random
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Set

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ProceduralSlot:
    """Immutable container for a procedural slot."""
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Sheaf:
    """
    Minimal sheaf representation with node dimensions, edge restrictions,
    sections, entropy, and pheromone levels.
    """

    def __init__(self, node_dims: Dict[int, int], edges: List[Tuple[int, int]]):
        self.node_dims: Dict[int, int] = dict(node_dims)
        self.edges: List[Tuple[int, int]] = list(edges)

        # internal storage
        self._restrictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[int, np.ndarray] = {}
        self._entropy: Dict[int, float] = {}
        self.pheromones: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def set_restriction(self, edge: Tuple[int, int], src_map: Sequence[float], dst_map: Sequence[float]) -> None:
        """Associate linear restriction maps with an oriented edge."""
        u, v = edge
        src = np.asarray(src_map, dtype=float)
        dst = np.asarray(dst_map, dtype=float)
        if src.shape != dst.shape:
            raise ValueError("src_map and dst_map must have the same shape")
        self._restrictions[(u, v)] = (src, dst)

    def set_section(self, node: int, value: Sequence[float]) -> None:
        """Assign a section (vector) to a node."""
        self._sections[node] = np.asarray(value, dtype=float)

    def set_entropy(self, node: int, entropy: float) -> None:
        """Store pre‑computed Shannon entropy for a node."""
        self._entropy[node] = float(entropy)

    def set_pheromone(self, node: int, pheromone: float) -> None:
        """Store the pheromone intensity for a node."""
        self.pheromones[node] = float(pheromone)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _edge_dim(self, u: int, v: int) -> int:
        """Return the dimension of the restriction map attached to edge (u,v)."""
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        raise KeyError(f"No restriction map for edge ({u}, {v})")

    def _c0_layout(self) -> Tuple[List[int], Dict[int, int], int]:
        """Layout of C⁰ – nodes concatenated in a single vector."""
        nodes = list(self.node_dims.keys())
        offsets: Dict[int, int] = {}
        pos = 0
        for n in nodes:
            offsets[n] = pos
            pos += self.node_dims[n]
        return nodes, offsets, pos

    def _c1_layout(self) -> Tuple[Dict[Tuple[int, int], int], int]:
        """Layout of C¹ – edges concatenated in a single vector."""
        offsets: Dict[Tuple[int, int], int] = {}
        pos = 0
        for e in self.edges:
            offsets[e] = pos
            pos += self._edge_dim(*e)
        return offsets, pos

    # ------------------------------------------------------------------
    # Cohomology‑like utilities
    # ------------------------------------------------------------------

    def coboundary_matrix(self) -> np.ndarray:
        """
        Build a sparse‑ish coboundary matrix d¹ : C⁰ → C¹.
        For each edge (u,v) we place the restriction maps as blocks.
        """
        _, node_offsets, total_node_dim = self._c0_layout()
        edge_offsets, total_edge_dim = self._c1_layout()
        d = np.zeros((total_edge_dim, total_node_dim), dtype=float)

        for (u, v), (src_map, dst_map) in self._restrictions.items():
            e_off = edge_offsets[(u, v)]
            dim = src_map.shape[0]

            # source block (negative orientation)
            n_off_u = node_offsets[u]
            d[e_off:e_off + dim, n_off_u:n_off_u + dim] = -src_map.reshape(dim, dim)

            # target block (positive orientation)
            n_off_v = node_offsets[v]
            d[e_off:e_off + dim, n_off_v:n_off_v + dim] = dst_map.reshape(dim, dim)

        return d

    # ------------------------------------------------------------------
    # Entropy utilities
    # ------------------------------------------------------------------

    @staticmethod
    def shannon_entropy(values: Sequence[float]) -> float:
        """Compute Shannon entropy of a discrete distribution given by values."""
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return 0.0
        prob = arr / arr.sum()
        prob = prob[prob > 0]  # avoid log(0)
        return -float(np.sum(prob * np.log2(prob)))


# ----------------------------------------------------------------------
# Hash utilities (used for graph construction)
# ----------------------------------------------------------------------


def compute_dhash(values: List[float]) -> int:
    """
    Directional hash: compare each adjacent pair and set a bit if the
    left value is larger than the right one.
    """
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def compute_phash(values: List[float]) -> int:
    """
    Perceptual hash: threshold against the mean of the first 64 values.
    If fewer than 64 values are present, use all of them.
    """
    if not values:
        return 0
    arr = np.asarray(values[:64], dtype=float)
    avg = float(arr.mean())
    bits = 0
    for v in arr:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Return the Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Graph construction based on perceptual similarity
# ----------------------------------------------------------------------


def build_similarity_graph(elements: List[List[float]], max_hamming: int = 4) -> Dict[str, Set[str]]:
    """
    Build an undirected graph where each node corresponds to an element.
    Two nodes are connected if their perceptual hashes differ by at most
    `max_hamming` bits.
    """
    hashes: Dict[str, int] = {str(i): compute_phash(el) for i, el in enumerate(elements)}
    graph: Dict[str, Set[str]] = {key: set() for key in hashes}
    keys = list(hashes.keys())
    for i, ki in enumerate(keys):
        for kj in keys[i + 1:]:
            if hamming_distance(hashes[ki], hashes[kj]) <= max_hamming:
                graph[ki].add(kj)
                graph[kj].add(ki)
    return graph


# ----------------------------------------------------------------------
# Probabilistic utilities
# ----------------------------------------------------------------------


def broadcast_probability(phase: int, step: int) -> float:
    """
    Return a probability that decays exponentially with the distance
    between `phase` and `step`.  The function is monotone decreasing
    and always lies in (0,1].
    """
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive integers")
    exponent = max(0, phase - step)
    return 1.0 / (2 ** exponent)


# ----------------------------------------------------------------------
# Risk and Differential Privacy calculations
# ----------------------------------------------------------------------


def calculate_reconstruction_risk(sheaf: Sheaf, graph: Dict[str, Set[str]]) -> float:
    """
    Reconstruction risk combines three ingredients:

    1. Pheromone intensity at each node.
    2. Shannon entropy of the node's section (if available).
    3. Structural similarity of the node in the similarity graph.

    The risk for a node i is:
        r_i = pheromone_i * (1 - entropy_i / max_entropy) * (1 - avg_sim_i)

    where `avg_sim_i` is the average Jaccard similarity of i's neighbourhood.
    The total risk is the sum over all nodes.
    """
    if not sheaf.node_dims:
        return 0.0

    # pre‑compute maximum possible entropy for normalisation
    max_entropy = max(
        Sheaf.shannon_entropy([0] * dim) if dim > 0 else 0.0
        for dim in sheaf.node_dims.values()
    )
    max_entropy = max(max_entropy, 1e-12)  # avoid division by zero

    total_risk = 0.0
    for node_str, neigh in graph.items():
        node = int(node_str)
        pher = sheaf.pheromones.get(node, 0.0)

        # entropy term
        ent = sheaf._entropy.get(node)
        if ent is None:
            # compute on‑the‑fly from the stored section if possible
            sec = sheaf._sections.get(node)
            ent = Sheaf.shannon_entropy(sec) if sec is not None else 0.0
            sheaf._entropy[node] = ent
        norm_ent = ent / max_entropy

        # neighbourhood similarity term (Jaccard)
        if neigh:
            intersect_sizes = [
                len(neigh.intersection(graph[other]))
                for other in neigh
            ]
            union_sizes = [
                len(neigh.union(graph[other]))
                for other in neigh
            ]
            jaccards = [
                (i / u) if u > 0 else 0.0
                for i, u in zip(intersect_sizes, union_sizes)
            ]
            avg_sim = float(np.mean(jaccards))
        else:
            avg_sim = 0.0

        node_risk = pher * (1.0 - norm_ent) * (1.0 - avg_sim)
        total_risk += node_risk

    return total_risk


def laplace_mechanism(value: float, epsilon: float, sensitivity: float = 1.0) -> float:
    """Add Laplace noise calibrated to `epsilon` and `sensitivity`."""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale)
    return float(value + noise)


def calculate_differentially_private_aggregation(
    sheaf: Sheaf,
    graph: Dict[str, Set[str]],
    epsilon: float = 1.0,
) -> float:
    """
    Compute a DP‑protected aggregation of pheromone levels.
    The raw aggregation is a weighted sum where each node's weight is
    proportional to its degree in the similarity graph (higher degree → more
    influence).  Laplace noise is added to guarantee ε‑differential privacy.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    total_weight = 0.0
    weighted_sum = 0.0
    for node_str, neigh in graph.items():
        node = int(node_str)
        pher = sheaf.pheromones.get(node, 0.0)
        weight = len(neigh) + 1  # +1 to give isolated nodes a minimal weight
        total_weight += weight
        weighted_sum += weight * pher

    raw_agg = weighted_sum / total_weight if total_weight > 0 else 0.0
    dp_agg = laplace_mechanism(raw_agg, epsilon, sensitivity=1.0)
    return dp_agg


# ----------------------------------------------------------------------
# Example usage (executed only when run as script)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a tiny sheaf
    sheaf = Sheaf(node_dims={0: 5, 1: 3}, edges=[(0, 1)])

    # Restriction maps (identity for simplicity)
    sheaf.set_restriction((0, 1), np.identity(5), np.identity(5)[:3, :3])

    # Sections (random vectors)
    rng = np.random.default_rng(42)
    sheaf.set_section(0, rng.random(5))
    sheaf.set_section(1, rng.random(3))

    # Compute and store entropies
    for n in sheaf.node_dims:
        sheaf.set_entropy(n, Sheaf.shannon_entropy(sheaf._sections[n]))

    # Pheromone intensities
    sheaf.set_pheromone(0, 0.7)
    sheaf.set_pheromone(1, 0.4)

    # Build similarity graph from the sections
    elements = [list(sheaf._sections[n]) for n in sorted(sheaf.node_dims)]
    sim_graph = build_similarity_graph(elements)

    # Risk and DP aggregation
    risk = calculate_reconstruction_risk(sheaf, sim_graph)
    dp_agg = calculate_differentially_private_aggregation(sheaf, sim_graph, epsilon=0.8)

    print(f"Reconstruction risk: {risk:.4f}")
    print(f"Differentially private aggregation: {dp_agg:.4f}")