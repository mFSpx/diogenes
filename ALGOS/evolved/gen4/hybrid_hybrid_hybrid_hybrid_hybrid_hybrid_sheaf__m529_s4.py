# DARWIN HAMMER — match 529, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hoeffding_tre_m16_s1.py (gen3)
# parent_b: hybrid_hybrid_sheaf_cohomol_hybrid_shannon_entro_m6_s1.py (gen2)
# born: 2026-05-29T23:29:36Z

"""
Hybrid Sheaf‑Sketch‑Hoeffding‑Tropical Module
================================================

Parents
-------
* **hybrid_hybrid_sketch_hybrid_hoeffding_tre_m16_s1.py** – provides
  Count‑Min sketch, MinHash LSH, and the Hoeffding bound for online
  decision‑tree splitting.
* **hybrid_hybrid_sheaf_cohomol_hybrid_shannon_entro_m6_s1.py** – defines a
  cellular sheaf structure together with Shannon entropy on node sections.

Mathematical Bridge
-------------------
The bridge is the **information‑theoretic view of sheaf sections**.  
A sheaf stores a vector (section) on each node; the Shannon entropy of this
vector quantifies its uncertainty.  In a streaming setting the Hoeffding bound
decides whether the observed reduction of entropy (gain) is statistically
significant enough to split a decision node.  The restriction maps of the sheaf
are linear; by interpreting addition as the tropical “max‑plus” operation we
obtain a *tropical matrix product* that propagates sketch‑derived updates
through the sheaf while preserving the max‑plus algebra used for the Hoeffding
tree’s decision boundaries.

Thus the hybrid algorithm:
1. Sketches the incoming items (Count‑Min + MinHash) → frequency matrix.
2. Computes Shannon entropy of the sketch frequencies (information content).
3. Uses the Hoeffding bound on the entropy gain to decide a split.
4. Propagates the sketch‑derived update through the sheaf using tropical
   max‑plus matrix multiplication.

The code below implements this pipeline with three core functions that
demonstrate the fused behaviour.
"""

import hashlib
import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def count_min_sketch(items: List[Any], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Return a Count‑Min sketch table for *items*."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][h] += 1
    return table

def minhash_lsh_index(docs: Dict[int, List[str]]) -> Dict[str, List[int]]:
    """Very simple MinHash LSH bucket index."""
    buckets = defaultdict(list)
    for doc_id, shingles in docs.items():
        # pick the smallest 6‑hex‑digit hash as the bucket key
        key = min(
            (hashlib.sha1(s.encode()).hexdigest()[:6] for s in shingles),
            default="empty",
        )
        buckets[key].append(doc_id)
    return dict(buckets)

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound used for split decisions."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

# ----------------------------------------------------------------------
# Sheaf definition from Parent B (extended with tropical operations)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ProceduralSlot:
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
    Cellular sheaf storing a vector (section) on each node and linear restriction
    maps on edges.  The restriction maps are ordinary NumPy arrays; tropical
    operations are performed on‑the‑fly when needed.
    """

    def __init__(self, node_dims: Dict[Any, int], edge_list: List[Tuple[Any, Any]]):
        self.node_dims: Dict[Any, int] = dict(node_dims)
        self.edges: List[Tuple[Any, Any]] = list(edge_list)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}
        self._entropy: Dict[Any, float] = {}

    # ------------------------------------------------------------------
    # Core sheaf API
    # ------------------------------------------------------------------
    def set_restriction(self, edge: Tuple[Any, Any], src_map: List[List[float]], dst_map: List[List[float]]) -> None:
        u, v = edge
        src_arr = np.array(src_map, dtype=float)
        dst_arr = np.array(dst_map, dtype=float)
        self._restrictions[(u, v)] = (src_arr, dst_arr)

    def set_section(self, node: Any, value: List[float]) -> None:
        self._sections[node] = np.array(value, dtype=float)

    def get_section(self, node: Any) -> np.ndarray:
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    # ------------------------------------------------------------------
    # Entropy handling (Parent B)
    # ------------------------------------------------------------------
    @staticmethod
    def _shannon_entropy(vec: np.ndarray) -> float:
        """Shannon entropy of a non‑negative vector (treated as a probability distribution)."""
        if vec.size == 0:
            return 0.0
        total = vec.sum()
        if total == 0:
            return 0.0
        p = vec / total
        # avoid log(0) by masking zeros
        mask = p > 0
        return -float(np.sum(p[mask] * np.log(p[mask])))

    def compute_node_entropy(self, node: Any) -> float:
        """Compute and store entropy for *node* based on its current section."""
        entropy = self._shannon_entropy(self.get_section(node))
        self._entropy[node] = entropy
        return entropy

    # ------------------------------------------------------------------
    # Tropical max‑plus propagation (mathematical bridge)
    # ------------------------------------------------------------------
    @staticmethod
    def _tropical_max_plus(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Tropical (max‑plus) matrix product C = A ⊗ B where
        C[i, j] = max_k (A[i, k] + B[k, j]).
        """
        # Expand dimensions to broadcast addition, then take max over k
        # A: (m, k), B: (k, n) -> result (m, n)
        return np.max(A[:, :, None] + B[None, :, :], axis=1)

    def tropical_update_via_edge(self, edge: Tuple[Any, Any]) -> None:
        """
        Propagate the section from source to destination using the tropical
        max‑plus product of the source section with the restriction map.
        """
        u, v = edge
        if (u, v) not in self._restrictions:
            raise KeyError(f"No restriction defined for edge {edge}")

        src_map, dst_map = self._restrictions[(u, v)]

        src_section = self.get_section(u).reshape(-1, 1)          # column vector
        # Tropical product src_section ⊗ src_map (src_map maps from u‑dim → edge‑dim)
        edge_vec = self._tropical_max_plus(src_section.T, src_map)  # shape (1, edge_dim)

        # Now map edge_vec to destination using dst_map (edge‑dim → v‑dim)
        dst_section = self._tropical_max_plus(edge_vec, dst_map)    # shape (1, dim_v)

        # Store the updated destination section
        self.set_section(v, dst_section.ravel())

    def tropical_propagate_all(self) -> None:
        """Apply tropical_update_via_edge to every edge in the sheaf."""
        for edge in self.edges:
            self.tropical_update_via_edge(edge)

# ----------------------------------------------------------------------
# Hybrid Functions (demonstrate fused behaviour)
# ----------------------------------------------------------------------
def sketch_entropy(items: List[Any]) -> float:
    """
    Compute the Shannon entropy of the frequency distribution obtained from a
    Count‑Min sketch of *items*.  The sketch collapses the multiset into a
    low‑dimensional histogram; entropy on that histogram measures information
    content after dimensionality reduction.
    """
    sketch = count_min_sketch(items)
    # Flatten the sketch matrix into a single vector of counts
    flat_counts = np.array([c for row in sketch for c in row], dtype=float)
    # Use the Sheaf static method for entropy (re‑use implementation)
    return Sheaf._shannon_entropy(flat_counts)


def hybrid_split_decision(
    sheaf: Sheaf,
    node: Any,
    sketch_gain: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> Tuple[bool, float]:
    """
    Decide whether to split *node* in the sheaf based on an entropy gain
    derived from a sketch.  The decision follows the Hoeffding bound logic
    from Parent A, but the gain is the reduction of Shannon entropy
    (Parent B).
    Returns (should_split, epsilon_used).
    """
    current_entropy = sheaf.compute_node_entropy(node)
    # hypothetical new entropy after applying the sketch gain (lower is better)
    new_entropy = max(0.0, current_entropy - sketch_gain)

    best_gain = current_entropy - new_entropy
    second_best_gain = 0.0  # assume no alternative split

    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain

    should = (gap > eps) or (abs(gap) < tie_threshold and best_gain > 0)
    return should, eps


def hybrid_process(
    items: List[Any],
    sheaf: Sheaf,
    target_node: Any,
    r: float = 1.0,
    delta: float = 0.05,
    n: int = 100,
) -> None:
    """
    Full hybrid pipeline:
    1. Sketch the incoming *items* and compute entropy gain.
    2. Use Hoeffding bound to decide on a split for *target_node*.
    3. If a split is warranted, perform a tropical max‑plus propagation
       to update the sheaf sections.
    """
    # Step 1 – sketch → entropy
    entropy_before = sheaf.compute_node_entropy(target_node)
    sketch_gain = sketch_entropy(items)  # information extracted from the stream

    # Step 2 – split decision
    split, eps = hybrid_split_decision(sheaf, target_node, sketch_gain, r, delta, n)

    print(f"[Hybrid] Node '{target_node}' entropy before: {entropy_before:.4f}")
    print(f"[Hybrid] Sketch‑derived entropy gain: {sketch_gain:.4f}")
    print(f"[Hybrid] Hoeffding epsilon: {eps:.4f} → split decision: {split}")

    if split:
        # For demonstration, we simply treat the split as an update of the node's
        # section using the sketch frequencies (tropical propagation will spread it).
        # Create a dummy vector from the sketch (flattened counts)
        sketch_vec = np.array([c for row in count_min_sketch(items) for c in row], dtype=float)
        # Pad / truncate to match node dimension
        dim = sheaf.node_dims[target_node]
        if sketch_vec.size < dim:
            padded = np.pad(sketch_vec, (0, dim - sketch_vec.size), constant_values=0)
        else:
            padded = sketch_vec[:dim]
        sheaf.set_section(target_node, padded)

        # Propagate through the sheaf using tropical max‑plus algebra
        sheaf.tropical_propagate_all()
        print(f"[Hybrid] Tropical propagation performed after split.")
    else:
        print(f"[Hybrid] No split performed; sheaf unchanged.")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Construct a tiny sheaf with two nodes and a single directed edge.
    node_dimensions = {"A": 8, "B": 8}
    edges = [("A", "B")]
    sheaf = Sheaf(node_dimensions, edges)

    # Initialise sections with random positive values.
    rng = np.random.default_rng(seed=42)
    sheaf.set_section("A", rng.random(node_dimensions["A"]))
    sheaf.set_section("B", rng.random(node_dimensions["B"]))

    # Define a simple restriction: identity maps (tropical max‑plus will act as
    # max of elementwise sums, effectively a shift).
    identity = np.eye(node_dimensions["A"])
    sheaf.set_restriction(("A", "B"), src_map=identity.tolist(), dst_map=identity.tolist())

    # Simulated streaming items (simple integers)
    stream_items = [random.randint(0, 100) for _ in range(250)]

    # Run the hybrid pipeline
    hybrid_process(
        items=stream_items,
        sheaf=sheaf,
        target_node="A",
        r=1.0,
        delta=0.05,
        n=len(stream_items),
    )

    # Show final sections and entropies
    for node in node_dimensions:
        sec = sheaf.get_section(node)
        ent = sheaf.compute_node_entropy(node)
        print(f"Node '{node}' final section (first 4 values): {sec[:4]}")
        print(f"Node '{node}' final entropy: {ent:.4f}")