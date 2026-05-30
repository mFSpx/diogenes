# DARWIN HAMMER — match 2956, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_krampus_brain_hybrid_hybrid_ternar_m55_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_decisi_ternary_lens_audit_m733_s1.py (gen3)
# born: 2026-05-29T23:46:51Z

"""Hybrid Algorithm: Fusion of Krampus Brain‑Map Sheaf Cohomology (Parent A) and
Decision‑Hygiene Log‑Count/Ternary Lens Audit (Parent B).

Mathematical Bridge
-------------------
Parent A provides a sheaf over a graph: each node carries a feature vector (a
section) and edges have restriction maps.  Ollivier‑Ricci curvature κ(e) can be
computed from the node sections as a transport‑cost based similarity.

Parent B supplies a frequency vector updated by a Count‑Min sketch and a
classification of items (the ternary lens audit).  The frequencies are
log‑scaled, yielding a log‑likelihood term L.

The hybrid combines these by using κ(e) as a weighting factor for the
log‑frequency updates on the sketch: each edge contributes
 Δc = κ(e)·log(f_i+1) to the sketch cells belonging to its incident nodes.
The total “Hybrid Free Energy” is defined as

    F = Σ_e (1‑κ(e))²   –   Σ_i log(count_i + 1)

where the first sum penalises low curvature (inconsistent sheaf sections) and
the second sum rewards high log‑counts from the decision‑hygiene statistics.

The code below implements this unified system with three core functions:
 * build_sheaf – creates a sheaf from raw texts,
 * update_counts_and_sketch – merges classifications, curvature and a
   Count‑Min sketch,
 * compute_hybrid_free_energy – evaluates the combined objective.
"""

import math
import random
import sys
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (Sheaf, curvature)
# ----------------------------------------------------------------------
class Sheaf:
    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)          # node -> dimension of its section
        self.edges = list(edge_list)              # list of (u, v)
        self._restrictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[int, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[int, int], src_map: List[float], dst_map: List[float]) -> None:
        u, v = edge
        src_map = np.array(src_map, dtype=float)
        dst_map = np.array(dst_map, dtype=float)
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node: int, value: List[float]) -> None:
        self._sections[node] = np.array(value, dtype=float)

    def get_section(self, node: int) -> np.ndarray:
        return self._sections[node]

    def edge_dim(self, u: int, v: int) -> int:
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        return 0

def ollivier_ricci_curvature(sheaf: Sheaf, edge: Tuple[int, int]) -> float:
    """Compute a simple Ollivier‑Ricci curvature proxy for an edge.

    κ = 1 - (‖s_u - s_v‖_1) / (max_norm + ε)
    where s_u, s_v are node sections and max_norm is the maximal L1 distance
    observed among all edges (computed lazily on first call).
    """
    u, v = edge
    su = sheaf.get_section(u)
    sv = sheaf.get_section(v)
    dist = np.linalg.norm(su - sv, ord=1)
    # Lazy global max distance cache
    if not hasattr(ollivier_ricci_curvature, "_max_dist"):
        ollivier_ricci_curvature._max_dist = max(
            np.linalg.norm(
                sheaf.get_section(a) - sheaf.get_section(b), ord=1
            )
            for a, b in sheaf.edges
        )
    max_dist = ollivier_ricci_curvature._max_dist or 1e-9
    return 1.0 - (dist / (max_dist + 1e-12))

# ----------------------------------------------------------------------
# Parent B components (regex classification, Count‑Min sketch)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|"
    r"receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|"
    r"check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|schedule|timetable|agenda|program|"
    r"procedure|protocol|policy|provision|arrangement)\b",
    re.I,
)
CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}

def simple_hash_factory(seed: int, width: int) -> Callable[[int], int]:
    """Return a deterministic hash function for Count‑Min sketch."""
    def _hash(x: int) -> int:
        random.seed((x, seed))
        return random.randint(0, width - 1)
    return _hash

def init_count_min_sketch(depth: int, width: int) -> np.ndarray:
    """Create a zero‑initialized Count‑Min sketch."""
    return np.zeros((depth, width), dtype=np.int64)

def update_sketch(
    sketch: np.ndarray,
    hash_fns: List[Callable[[int], int]],
    item_id: int,
    increment: int = 1,
) -> None:
    for i, h in enumerate(hash_fns):
        idx = h(item_id)
        sketch[i, idx] += increment

def estimate_sketch(
    sketch: np.ndarray,
    hash_fns: List[Callable[[int], int]],
    item_id: int,
) -> int:
    return min(sketch[i, h(item_id)] for i, h in enumerate(hash_fns))

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def extract_dummy_features(text: str) -> List[float]:
    """Create a low‑dimensional deterministic feature vector from a text.

    Features:
        0 – normalized length (chars / 200)
        1 – vowel ratio
        2 – presence of evidence keywords (binary)
        3 – presence of planning keywords (binary)
    """
    length = len(text)
    norm_len = min(length / 200.0, 1.0)
    vowels = sum(c.lower() in "aeiou" for c in text)
    vowel_ratio = vowels / max(length, 1)
    ev = 1.0 if EVIDENCE_RE.search(text) else 0.0
    pl = 1.0 if PLANNING_RE.search(text) else 0.0
    return [norm_len, vowel_ratio, ev, pl]

def build_sheaf(texts: List[str]) -> Sheaf:
    """Construct a Sheaf where each text becomes a node with a feature section.

    Edges are added between consecutive texts to form a simple path graph.
    """
    node_dims = {i: 4 for i in range(len(texts))}
    edges = [(i, i + 1) for i in range(len(texts) - 1)]
    sheaf = Sheaf(node_dims, edges)

    for i, txt in enumerate(texts):
        sheaf.set_section(i, extract_dummy_features(txt))

    # Restriction maps are identity (no transformation) for simplicity.
    for u, v in edges:
        dim = node_dims[u]
        sheaf.set_restriction((u, v), np.eye(dim).flatten().tolist(),
                              np.eye(dim).flatten().tolist())
    return sheaf

def update_counts_and_sketch(
    sheaf: Sheaf,
    manifest: dict,
    sketch: np.ndarray,
    hash_fns: List[Callable[[int], int]],
) -> None:
    """Merge ternary‑lens classifications with sheaf curvature into the sketch.

    For each candidate in the manifest:
        * Derive a node id from its hash (modulo number of nodes).
        * Use its classification to decide a weight w:
            usable_now            -> 1.0
            research_only         -> 0.8
            needs_conversion      -> 0.5
            unsafe_for_fastpath   -> 0.3
            unsupported           -> 0.0 (skip)
        * For every incident edge, compute κ(e) and add
            increment = int(round(w * κ(e) * 10))
          to the sketch cell corresponding to the node id.
    """
    classification_weight = {
        "usable_now": 1.0,
        "research_only": 0.8,
        "needs_conversion": 0.5,
        "unsafe_for_fastpath": 0.3,
        "unsupported": 0.0,
    }

    num_nodes = len(sheaf.node_dims)
    for candidate in manifest.get("vendors", []):
        cls = candidate.get("classification")
        if cls not in CLASSIFICATIONS or classification_weight[cls] == 0.0:
            continue

        # Derive a deterministic integer id from candidate_key
        key = candidate.get("candidate_key", "")
        node_id = abs(hash(key)) % num_nodes
        weight = classification_weight[cls]

        # Find edges incident to node_id
        incident_edges = [e for e in sheaf.edges if node_id in e]
        for edge in incident_edges:
            κ = ollivier_ricci_curvature(sheaf, edge)
            inc = max(1, int(round(weight * κ * 10)))
            update_sketch(sketch, hash_fns, node_id, inc)

def compute_hybrid_free_energy(sheaf: Sheaf, sketch: np.ndarray, hash_fns: List[Callable[[int], int]]) -> float:
    """Calculate the Hybrid Free Energy F = Σ_e (1‑κ(e))² – Σ_i log(c_i+1).

    c_i is the estimated count for node i from the Count‑Min sketch.
    """
    curvature_term = sum(
        (1.0 - ollivier_ricci_curvature(sheaf, e)) ** 2 for e in sheaf.edges
    )
    # Estimate counts for each node
    counts = [
        estimate_sketch(sketch, hash_fns, node_id)
        for node_id in range(len(sheaf.node_dims))
    ]
    log_term = sum(math.log(c + 1) for c in counts)
    return curvature_term - log_term

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample texts
    sample_texts = [
        "The system recorded evidence of a breach at 12:00 UTC.",
        "Plan the rollout steps for the new module.",
        "Verify the checksum sha256 of the binary before deployment.",
        "Research only: this feature is experimental.",
    ]

    # Build sheaf from texts
    my_sheaf = build_sheaf(sample_texts)

    # Mock manifest resembling the output of load_manifest
    mock_manifest = {
        "vendors": [
            {
                "candidate_key": "mod_alpha",
                "classification": "usable_now",
                "family": "standard_lora",
                "notes": "",
            },
            {
                "candidate_key": "mod_beta",
                "classification": "needs_conversion",
                "family": "custom",
                "notes": "",
            },
            {
                "candidate_key": "mod_gamma",
                "classification": "unsupported",
                "family": "legacy",
                "notes": "",
            },
        ]
    }

    # Initialize Count‑Min sketch
    DEPTH, WIDTH = 5, 128
    cm_sketch = init_count_min_sketch(DEPTH, WIDTH)
    hash_functions = [simple_hash_factory(seed, WIDTH) for seed in range(DEPTH)]

    # Update sketch with manifest data
    update_counts_and_sketch(my_sheaf, mock_manifest, cm_sketch, hash_functions)

    # Compute hybrid free energy
    energy = compute_hybrid_free_energy(my_sheaf, cm_sketch, hash_functions)
    print(f"Hybrid Free Energy: {energy:.4f}")

    # Simple sanity checks
    assert isinstance(energy, float)
    print("Smoke test completed successfully.")