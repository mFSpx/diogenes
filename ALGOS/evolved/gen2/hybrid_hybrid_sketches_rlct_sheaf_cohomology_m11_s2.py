# DARWIN HAMMER — match 11, survivor 2
# gen: 2
# parent_a: hybrid_sketches_rlct_grokking_m5_s0.py (gen1)
# parent_b: sheaf_cohomology.py (gen0)
# born: 2026-05-29T23:22:48Z

"""Hybrid Sketch–Sheaf Cohomology Module.

Parents:
- hybrid_sketches_rlct_grokking_m5_s0.py (Algorithm A)
- sheaf_cohomology.py (Algorithm B)

Mathematical Bridge:
Algorithm A reduces data dimensionality via Count‑Min sketches (or MinHash LSH) and
estimates information loss with a Real Log Canonical Threshold (RLCT) derived from
the sequence of sketch counts.  
Algorithm B treats data attached to graph vertices as a 0‑cochain of a cellular
sheaf and measures inconsistency through the coboundary operator δ and the
sheaf Laplacian L = δᵀδ.

The hybrid algorithm interprets the sketch table as a *section* of a sheaf:
each hash bucket (width) is a vertex, each sketch depth is a separate vector‑space
dimension at that vertex, and the restriction maps between vertices are induced
by the hash functions themselves (identity maps up to a scaling factor).  The
coboundary residual δ(s) therefore quantifies how much the sketch‑based reduction
distorts the original data.  By feeding the magnitudes of these residuals into the
same log‑log regression used in Algorithm A we obtain an RLCT that now reflects
both combinatorial sketch loss and sheaf‑theoretic inconsistency.

The module provides three high‑level hybrid operations:
1. `count_min_sheaf` – builds a sheaf from a Count‑Min sketch.
2. `hybrid_rlct_via_sheaf` – computes an RLCT from the sheaf coboundary residuals.
3. `hybrid_info_loss` – returns a normalized information‑loss measure that blends
   the RLCT estimate with the sheaf Laplacian energy.

All functions are pure NumPy/Python and require only the standard library.
"""

import hashlib
import math
import random
import sys
import pathlib
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Sheaf class (adapted from parent B)
# ---------------------------------------------------------------------------

class Sheaf:
    """Cellular sheaf over a graph.

    Nodes carry vector spaces of prescribed dimensions; edges carry restriction
    maps from the incident node spaces to a common edge space.
    """

    def __init__(self, node_dims, edge_list):
        self.node_dims = dict(node_dims)          # node_id -> int
        self.edges = list(edge_list)              # list of (u, v) tuples, orientation matters
        self._restrictions = {}                   # (u, v) -> (src_map, dst_map)
        self._sections = {}                       # node_id -> np.ndarray

    def set_restriction(self, edge, src_map, dst_map):
        u, v = edge
        self._restrictions[(u, v)] = (np.array(src_map, dtype=float),
                                      np.array(dst_map, dtype=float))

    def set_section(self, node, value):
        self._sections[node] = np.array(value, dtype=float)

    # -----------------------------------------------------------------------
    # Internal layout helpers
    # -----------------------------------------------------------------------
    def _edge_dim(self, u, v):
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        raise KeyError(f"No restriction map for edge ({u}, {v})")

    def _c0_layout(self):
        nodes = list(self.node_dims.keys())
        offsets = {}
        pos = 0
        for n in nodes:
            offsets[n] = pos
            pos += self.node_dims[n]
        return nodes, offsets, pos  # pos = total dimension of C^0

    def _c1_layout(self):
        offsets = {}
        pos = 0
        for e in self.edges:
            u, v = e
            d = self._edge_dim(u, v)
            offsets[e] = (pos, d)
            pos += d
        return offsets, pos  # pos = total dimension of C^1

    # -----------------------------------------------------------------------
    # Linear operators
    # -----------------------------------------------------------------------
    def coboundary_operator(self):
        """Construct the coboundary matrix δ : C⁰ → C¹."""
        nodes, c0_off, c0_dim = self._c0_layout()
        c1_off, c1_dim = self._c1_layout()
        delta = np.zeros((c1_dim, c0_dim), dtype=float)

        for u, v in self.edges:
            row_start, d_e = c1_off[(u, v)]

            if (u, v) in self._restrictions:
                src_map, dst_map = self._restrictions[(u, v)]
            else:
                # reversed orientation stored
                dst_map, src_map = self._restrictions[(v, u)]

            col_u = c0_off[u]
            col_v = c0_off[v]
            dim_u = self.node_dims[u]
            dim_v = self.node_dims[v]

            delta[row_start:row_start + d_e, col_u:col_u + dim_u] -= src_map
            delta[row_start:row_start + d_e, col_v:col_v + dim_v] += dst_map

        return delta

    def consistency_residual(self):
        """δ(s) for the current node sections s."""
        nodes, c0_off, c0_dim = self._c0_layout()
        s = np.zeros(c0_dim, dtype=float)
        for n in nodes:
            if n in self._sections:
                off = c0_off[n]
                dim = self.node_dims[n]
                s[off:off + dim] = self._sections[n]
        delta = self.coboundary_operator()
        return delta @ s

    def global_inconsistency(self):
        """||δ(s)||²."""
        r = self.consistency_residual()
        return float(np.dot(r, r))

    def laplacian(self):
        """Sheaf Laplacian L = δᵀδ."""
        delta = self.coboundary_operator()
        return delta.T @ delta

# ---------------------------------------------------------------------------
# Hybrid construction utilities
# ---------------------------------------------------------------------------

def count_min_sketch(items, width=64, depth=4):
    """Standard Count‑Min sketch (parent A). Returns a depth×width integer table."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table

def _hash_to_edge(u, v, depth):
    """Deterministic edge identifier derived from hash of the ordered pair."""
    h = hashlib.sha256(f'{u}-{v}-{depth}'.encode()).hexdigest()
    # Map to a small integer to serve as edge dimension (1 or 2)
    return 1 + (int(h, 16) % 2)   # edge space dimension 1 or 2

def count_min_sheaf(items, width=64, depth=4):
    """
    Build a Sheaf where each (depth, bucket) pair is a vertex.
    Edges connect consecutive depths for the same bucket, and the restriction maps
    are identity matrices (or scaled by a small random factor to avoid singularities).
    The node section at (d, b) is the count stored in the sketch table.
    """
    sketch = count_min_sketch(items, width, depth)

    # Define node identifiers as tuples (d, b)
    node_dims = {}
    for d in range(depth):
        for b in range(width):
            node_dims[(d, b)] = 1  # scalar count per bucket

    # Build edges: connect (d, b) -> (d+1, b) for d = 0..depth-2
    edges = []
    for d in range(depth - 1):
        for b in range(width):
            edges.append(((d, b), (d + 1, b)))  # oriented from lower depth to higher

    sheaf = Sheaf(node_dims, edges)

    # Set restriction maps (identity scaled)
    for (u, v) in edges:
        edge_dim = _hash_to_edge(u, v, depth)
        src_map = np.eye(edge_dim, node_dims[u])  # shape (edge_dim, 1)
        dst_map = np.eye(edge_dim, node_dims[v])  # shape (edge_dim, 1)
        # Slight random scaling to keep matrices full rank
        scale = 1.0 + 0.01 * random.random()
        src_map *= scale
        dst_map *= scale
        sheaf.set_restriction((u, v), src_map, dst_map)

    # Populate sections with sketch counts
    for d in range(depth):
        for b in range(width):
            count = sketch[d][b]
            sheaf.set_section((d, b), [count])

    return sheaf

def _log_log_regression(y_vals):
    """
    Perform the same regression as parent A's `estimate_rlct_from_losses`,
    but on an arbitrary positive sequence y_vals.
    Returns the slope (RLCT estimate).
    """
    losses = np.asarray(y_vals, dtype=np.float64)
    n = np.arange(1, len(losses) + 1, dtype=np.float64)

    if np.any(n <= np.e):
        raise ValueError("All n must be > e for log(log(n)) to be positive")
    if len(losses) != len(n):
        raise ValueError("Length mismatch between values and indices")

    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(n))

    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)

def hybrid_rlct_via_sheaf(sheaf):
    """
    Compute an RLCT estimate from the sheaf coboundary residuals.
    The residual vector is split into blocks per edge; the ℓ₁ norm of each block
    forms a loss sequence analogous to the sketch count losses.
    """
    residual = sheaf.consistency_residual()
    # Edge layout to split residual into per‑edge blocks
    c1_off, _ = sheaf._c1_layout()
    edge_losses = []
    for edge in sheaf.edges:
        row_start, d_e = c1_off[edge]
        block = residual[row_start:row_start + d_e]
        edge_losses.append(np.linalg.norm(block, 1))  # ℓ₁ norm as loss magnitude

    if len(edge_losses) < 2:
        return 0.0
    return _log_log_regression(edge_losses)

def hybrid_info_loss(items, width=64, depth=4):
    """
    End‑to‑end hybrid information‑loss metric.
    1. Build a sheaf from a Count‑Min sketch of `items`.
    2. Compute RLCT via sheaf coboundary.
    3. Compute sheaf Laplacian energy (global inconsistency).
    Returns a normalized scalar in [0, 1] where larger values indicate more loss.
    """
    sheaf = count_min_sheaf(items, width, depth)
    rlct = hybrid_rlct_via_sheaf(sheaf)
    inconsistency = sheaf.global_inconsistency()

    # Normalization: depth acts as a scale for RLCT (as in parent A)
    rlct_norm = rlct / max(depth, 1.0)
    # Inconsistency is scaled by the total number of edges to keep it comparable
    edge_factor = max(len(sheaf.edges), 1)
    inc_norm = inconsistency / (edge_factor * np.max([1.0, inconsistency]))

    # Combine multiplicatively to emphasise simultaneous high RLCT and high inconsistency
    info_loss = 1.0 - (1.0 - rlct_norm) * (1.0 - inc_norm)
    return float(min(max(info_loss, 0.0), 1.0))

# ---------------------------------------------------------------------------
# Optional LSH‑based sheaf construction (demonstrates second hybrid function)
# ---------------------------------------------------------------------------

def minhash_lsh_index(docs):
    """Parent A helper: simple MinHash LSH bucket index."""
    buckets = defaultdict(list)
    for doc_id, shingles in docs.items():
        key = min((hashlib.sha1(s.encode()).hexdigest()[:6] for s in shingles), default='empty')
        buckets[key].append(doc_id)
    return dict(buckets)

def lsh_sheaf(docs):
    """
    Build a sheaf where each LSH bucket is a node.
    Edges connect buckets that share at least one document (co‑occurrence).
    Node sections count the number of documents in the bucket.
    """
    index = minhash_lsh_index(docs)
    bucket_ids = list(index.keys())
    node_dims = {b: 1 for b in bucket_ids}
    edges = []

    # Build edges for every unordered pair of buckets that share a doc
    doc_to_buckets = defaultdict(set)
    for b, docs_in_bucket in index.items():
        for d in docs_in_bucket:
            doc_to_buckets[d].add(b)

    seen = set()
    for d, buckets in doc_to_buckets.items():
        buckets = list(buckets)
        for i in range(len(buckets)):
            for j in range(i + 1, len(buckets)):
                u, v = buckets[i], buckets[j]
                if (u, v) not in seen and (v, u) not in seen:
                    edges.append((u, v))
                    seen.add((u, v))

    sheaf = Sheaf(node_dims, edges)

    # Identity restriction maps (edge_dim = 1)
    for (u, v) in edges:
        src_map = np.array([[1.0]])  # shape (1,1)
        dst_map = np.array([[1.0]])
        sheaf.set_restriction((u, v), src_map, dst_map)

    # Sections: count of docs per bucket
    for b, docs_in_bucket in index.items():
        sheaf.set_section(b, [len(docs_in_bucket)])

    return sheaf

def hybrid_lsh_info_loss(docs):
    """
    Hybrid loss for a collection of documents using MinHash LSH → sheaf pipeline.
    """
    sheaf = lsh_sheaf(docs)
    rlct = hybrid_rlct_via_sheaf(sheaf)
    inc = sheaf.global_inconsistency()
    # Simple normalization similar to `hybrid_info_loss`
    rlct_norm = rlct / max(len(sheaf.edges), 1)
    inc_norm = inc / (len(sheaf.edges) + 1e-9)
    return float(min(max(1.0 - (1.0 - rlct_norm) * (1.0 - inc_norm), 0.0), 1.0))

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test 1: simple numeric data
    data = [str(i) for i in range(1000)]
    print("Hybrid RLCT (sketch → sheaf):", hybrid_rlct_via_sheaf(count_min_sheaf(data)))
    print("Hybrid info loss (items):", hybrid_info_loss(data))

    # Test 2: small document collection for LSH path
    docs = {
        "doc1": {"hello", "world", "foo"},
        "doc2": {"foo", "bar"},
        "doc3": {"world", "baz"},
        "doc4": {"hello", "baz"},
    }
    print("Hybrid RLCT (LSH → sheaf):", hybrid_rlct_via_sheaf(lsh_sheaf(docs)))
    print("Hybrid info loss (docs):", hybrid_lsh_info_loss(docs))