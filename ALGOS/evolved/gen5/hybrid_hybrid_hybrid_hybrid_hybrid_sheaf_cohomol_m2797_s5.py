# DARWIN HAMMER — match 2797, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1035_s0.py (gen4)
# parent_b: hybrid_sheaf_cohomology_percyphon_m2_s1.py (gen1)
# born: 2026-05-29T23:45:56Z

"""Hybrid algorithm merging:
- hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1035_s0 (Algorithm A)
- hybrid_sheaf_cohomology_percyphon_m2_s1 (Algorithm B)

Mathematical bridge:
Algorithm A produces an expected feature‑count vector **f** (posterior edge
beliefs) for each node.  Algorithm B manipulates vector spaces attached to the
nodes of a graph via a cellular sheaf.  The fusion treats **f** as the
stalk‑section of the sheaf: the dimension of the stalk at a node equals the
length of **f**, and the restriction maps on an edge are diagonal matrices
whose entries are the feature values of the source/target node.  Consequently,
the coboundary (consistency) of the sheaf is weighted by the posterior beliefs
from A.  A recovery‑priority scalar derived from A (using the ternary offset of
a procedural slot) is added to the global inconsistency norm, yielding a
single hybrid cost that simultaneously respects probabilistic evidence and
sheaf‑cohomology consistency.
"""

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import re
from dataclasses import asdict, dataclass

# ----------------------------------------------------------------------
# Types and simple utilities (Algorithm A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|"
    r"receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|"
    r"check|checked|audit)\b",
    re.IGNORECASE,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|"
    r"prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|"
    r"test|smoke)\b",
    re.IGNORECASE,
)


def hybrid_feature_vector(text: str) -> np.ndarray:
    """
    Compute a 2‑dimensional expected feature‑count vector from raw text.
    The first component counts evidence‑related tokens, the second counts
    planning‑related tokens.  The vector is normalized to sum to 1 (if any
    token is found) to serve as a posterior belief distribution.
    """
    evidence_hits = len(EVIDENCE_RE.findall(text))
    planning_hits = len(PLANNING_RE.findall(text))
    raw = np.array([evidence_hits, planning_hits], dtype=float)
    total = raw.sum()
    if total == 0.0:
        return np.array([0.5, 0.5])  # uniform prior when nothing is found
    return raw / total


def hybrid_recovery_priority(feature_vec: np.ndarray, ternary_offset: int) -> float:
    """
    Map a feature vector to a scalar recovery priority.
    A simple linear model is used: priority = w·f + offset,
    where w = [0.7, 0.3] favours evidence over planning.
    """
    weights = np.array([0.7, 0.3])
    return float(weights @ feature_vec) + ternary_offset


# ----------------------------------------------------------------------
# Sheaf machinery (Algorithm B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Sheaf:
    """
    Cellular sheaf over a graph.

    node_dims : dict[node_id -> int]   dimension of the stalk at each node
    edges     : list[(u, v)]           oriented edges (u = tail, v = head)
    """

    def __init__(self, node_dims: Dict[str, int], edge_list: List[Tuple[str, str]]):
        self.node_dims = dict(node_dims)
        self.edges = list(edge_list)
        self._restrictions: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Restriction maps
    # ------------------------------------------------------------------
    def set_restriction(self, edge: Tuple[str, str], src_map: np.ndarray, dst_map: np.ndarray):
        """
        Assign restriction maps for an oriented edge.
        src_map : (edge_dim, dim_u)  linear map from stalk_u to edge space
        dst_map : (edge_dim, dim_v)  linear map from stalk_v to edge space
        For our hybrid we choose edge_dim = dim_u = dim_v and use diagonal matrices.
        """
        u, v = edge
        src_map = np.array(src_map, dtype=float)
        dst_map = np.array(dst_map, dtype=float)
        if src_map.shape[1] != self.node_dims[u] or dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("Restriction matrix column size must match node dimension.")
        self._restrictions[(u, v)] = (src_map, dst_map)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------
    def set_section(self, node: str, value: np.ndarray):
        """Assign a local section (vector) to a node."""
        value = np.asarray(value, dtype=float)
        if value.shape != (self.node_dims[node],):
            raise ValueError(f"Section shape {value.shape} incompatible with node dimension {self.node_dims[node]}.")
        self._sections[node] = value

    # ------------------------------------------------------------------
    # Consistency (coboundary) measure
    # ------------------------------------------------------------------
    def total_inconsistency(self) -> float:
        """
        Compute the sum over edges of ||src_map·s_u - dst_map·s_v||².
        This is the L2‑norm squared of the coboundary of the global section.
        """
        total = 0.0
        for (u, v), (src_map, dst_map) in self._restrictions.items():
            s_u = self._sections.get(u)
            s_v = self._sections.get(v)
            if s_u is None or s_v is None:
                continue  # missing section; treat as zero contribution
            diff = src_map @ s_u - dst_map @ s_v
            total += float(np.linalg.norm(diff) ** 2)
        return total


# ----------------------------------------------------------------------
# Hybrid construction utilities
# ----------------------------------------------------------------------
def build_hybrid_sheaf(
    node_texts: Dict[str, str],
    edges: List[Tuple[str, str]],
) -> Tuple[Sheaf, Dict[str, np.ndarray]]:
    """
    Create a Sheaf where each node's stalk dimension equals the length of the
    feature vector produced by `hybrid_feature_vector`.  The restriction maps
    on an edge (u, v) are diagonal matrices diag(f_u) and diag(f_v) respectively,
    thereby weighting the coboundary by the posterior beliefs of the incident
    nodes.
    Returns the sheaf and the dictionary of feature vectors for later use.
    """
    feature_vectors: Dict[str, np.ndarray] = {
        node: hybrid_feature_vector(txt) for node, txt in node_texts.items()
    }

    # All feature vectors have the same length (2) in this design.
    node_dims = {node: vec.shape[0] for node, vec in feature_vectors.items()}
    sheaf = Sheaf(node_dims, edges)

    # Set diagonal restriction maps for each edge.
    for u, v in edges:
        f_u = feature_vectors[u]
        f_v = feature_vectors[v]
        # Diagonal matrices with the feature values on the diagonal.
        src_map = np.diag(f_u)
        dst_map = np.diag(f_v)
        sheaf.set_restriction((u, v), src_map, dst_map)

    # Initialise sections with the raw feature vectors themselves.
    for node, vec in feature_vectors.items():
        sheaf.set_section(node, vec)

    return sheaf, feature_vectors


def hybrid_total_cost(
    sheaf: Sheaf,
    feature_vectors: Dict[str, np.ndarray],
    slots: List[ProceduralSlot],
) -> float:
    """
    Hybrid cost = sheaf inconsistency + Σ recovery_priority.
    The recovery priority for a slot uses the feature vector of the slot's
    associated node (identified by its `name` field) and the slot's ternary_offset.
    """
    inconsistency = sheaf.total_inconsistency()

    priority_sum = 0.0
    for slot in slots:
        # Assume slot.name matches a node identifier; fall back to uniform vector.
        f_vec = feature_vectors.get(slot.name, np.array([0.5, 0.5]))
        priority = hybrid_recovery_priority(f_vec, slot.ternary_offset)
        priority_sum += priority

    return inconsistency + priority_sum


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample node texts mimicking evidence / planning language.
    node_texts = {
        "A": "Evidence shows the system was verified. The plan includes three steps.",
        "B": "No concrete evidence, but the checklist is ready for execution.",
        "C": "Audit log contains hash and screenshot proof. Scheduling is pending.",
    }

    # Simple directed edges forming a triangle.
    edges = [("A", "B"), ("B", "C"), ("C", "A")]

    # Build the hybrid sheaf.
    sheaf, feats = build_hybrid_sheaf(node_texts, edges)

    # Create a few procedural slots.
    slots = [
        ProceduralSlot(
            slot_index=0,
            name="A",
            alias="alpha",
            persona="engineer",
            uuid="1111-2222",
            ternary_offset=2,
        ),
        ProceduralSlot(
            slot_index=1,
            name="B",
            alias="beta",
            persona="manager",
            uuid="3333-4444",
            ternary_offset=1,
        ),
        ProceduralSlot(
            slot_index=2,
            name="C",
            alias="gamma",
            persona="auditor",
            uuid="5555-6666",
            ternary_offset=3,
        ),
    ]

    cost = hybrid_total_cost(sheaf, feats, slots)
    print(f"Hybrid total cost: {cost:.4f}")
    # Ensure the code runs without raising exceptions.