# DARWIN HAMMER — match 5163, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m1050_s1.py (gen5)
# parent_b: hybrid_hybrid_sheaf_cohomol_hybrid_ternary_lens__m23_s0.py (gen2)
# born: 2026-05-30T00:00:11Z

"""Hybrid Sheaf-MinHash Cohomology
Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m1050_s1.py
Parent B: hybrid_hybrid_sheaf_cohomol_hybrid_ternary_lens__m23_s0.py

Mathematical Bridge:
- Parent A provides a MinHash signature for a text, which is transformed by a
  fractional‑power binding into a complex‑valued vector.
- Parent B defines a sheaf over a graph with restriction maps and a coboundary
  operator.
- The bridge consists in treating the bound MinHash vector as a *section* of the
  sheaf at each node, then applying a *pruning probability* (from Parent B) to
  filter sections before evaluating the sheaf cohomology.  The similarity of
  adjacent sections is measured with the structural‑similarity index (SSIM)
  defined in Parent A.
"""

import numpy as np
import math
import random
import sys
import pathlib

# ---------- Parent A components ----------
def minhash_for_text(text: str, k: int = 64) -> list[int]:
    """Generate a MinHash signature of length k for the given text."""
    text = text.replace(" ", "").strip().lower()
    shingles = [text[i:i + 5] for i in range(len(text) - 4)]
    signature = np.random.randint(0, 1_000_000, size=k)
    for s in shingles:
        h = hash(s) % k
        signature[h] = min(signature[h], hash(s) % 1_000_000)
    return signature.tolist()


def fractional_power_binding(minhash: list[int], power: float) -> np.ndarray:
    """Bind a MinHash signature with a fractional power, yielding a complex vector."""
    vec = np.array(minhash, dtype=float)
    # magnitude raised to `power`, phase preserved (here phase is 0 because vec is real)
    return np.power(np.abs(vec), power) * np.exp(1j * np.angle(vec))


def ssim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Structural similarity (cosine similarity) between two vectors."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


# ---------- Parent B components ----------
class Sheaf:
    """Sheaf over a finite directed graph with vector‑valued sections."""

    def __init__(self, node_dims: dict, edge_list: list[tuple[int, int]]):
        self.node_dims = dict(node_dims)          # node -> dimension
        self.edges = list(edge_list)              # list of (u, v)
        self._restrictions = {}                   # (u, v) -> (src_map, dst_map)
        self._sections = {}                       # node -> np.ndarray

    # ----- restriction handling -----
    def set_restriction(self, edge: tuple[int, int], src_map, dst_map):
        u, v = edge
        self._restrictions[(u, v)] = (
            np.array(src_map, dtype=float),
            np.array(dst_map, dtype=float)
        )

    # ----- section handling -----
    def set_section(self, node: int, value):
        self._sections[node] = np.array(value, dtype=complex)

    # ----- internal helpers -----
    def _edge_dim(self, u: int, v: int) -> int:
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        raise KeyError(f"No restriction map for edge ({u}, {v})")

    def _c0_layout(self):
        """Layout for C⁰ (node space). Returns (nodes, offsets, total_dim)."""
        nodes = list(self.node_dims.keys())
        offsets = {}
        pos = 0
        for n in nodes:
            offsets[n] = pos
            pos += self.node_dims[n]
        return nodes, offsets, pos

    def _c1_layout(self):
        """Layout for C¹ (edge space). Returns (edges, offsets, total_dim)."""
        offsets = {}
        pos = 0
        for e in self.edges:
            u, v = e
            d = self._edge_dim(u, v)
            offsets[e] = (pos, d)
            pos += d
        return self.edges, offsets, pos

    # ----- coboundary operator -----
    def coboundary(self) -> np.ndarray:
        """
        Assemble the global coboundary matrix d : C⁰ → C¹.
        For each edge (u, v) we place the restriction matrices:
            d_e = [ -R_uv   R_vu ]
        where R_uv maps sections at u to the edge space, and R_vu maps sections at v.
        """
        nodes, node_off, node_dim = self._c0_layout()
        edges, edge_off, edge_dim = self._c1_layout()
        D = np.zeros((edge_dim, node_dim), dtype=complex)

        for (u, v) in edges:
            off_e, dim_e = edge_off[(u, v)]
            src_map, dst_map = self._restrictions[(u, v)]
            # source contribution (negative)
            off_u = node_off[u]
            D[off_e:off_e + dim_e, off_u:off_u + src_map.shape[1]] -= src_map
            # target contribution (positive)
            off_v = node_off[v]
            D[off_e:off_e + dim_e, off_v:off_v + dst_map.shape[1]] += dst_map
        return D

    # ----- cohomology evaluation -----
    def evaluate_cohomology(self) -> np.ndarray:
        """
        Compute d·s where s is the stacked section vector.
        Returns the edge‑wise discrepancy (a measure of inconsistency).
        """
        D = self.coboundary()
        # stack sections in node order
        nodes, node_off, node_dim = self._c0_layout()
        s = np.zeros(node_dim, dtype=complex)
        for n in nodes:
            off = node_off[n]
            sec = self._sections.get(n, np.zeros(self.node_dims[n], dtype=complex))
            s[off:off + sec.shape[0]] = sec
        return D @ s

# ---------- Hybrid Functions ----------
def prune_section(section: np.ndarray, prob: float) -> np.ndarray | None:
    """Randomly keep a section with probability `prob`; otherwise discard (return None)."""
    return section if random.random() < prob else None


def compute_ssim_between_nodes(sheaf: Sheaf, u: int, v: int) -> float:
    """Compute SSIM between sections of nodes u and v after applying restriction maps."""
    if (u, v) not in sheaf._restrictions:
        raise KeyError(f"No restriction for edge ({u}, {v})")
    src_map, dst_map = sheaf._restrictions[(u, v)]
    sec_u = sheaf._sections.get(u)
    sec_v = sheaf._sections.get(v)
    if sec_u is None or sec_v is None:
        return 0.0
    # project onto edge space
    proj_u = src_map @ sec_u
    proj_v = dst_map @ sec_v
    return ssim(proj_u, proj_v)


def hybrid_process(
    text: str,
    node_dims: dict,
    edge_list: list[tuple[int, int]],
    power: float = 0.5,
    prune_prob: float = 0.7
) -> dict:
    """
    Full hybrid pipeline:
    1. Produce a MinHash signature from `text`.
    2. Bind it with a fractional power → complex vector.
    3. Install the bound vector as a section on every node.
    4. Prune sections according to `prune_prob`.
    5. Evaluate sheaf cohomology (inconsistency) and pairwise SSIM.
    Returns a dictionary with keys:
        'inconsistency' : np.ndarray (edge‑wise discrepancy)
        'ssim_matrix'   : dict of ((u,v): float)
    """
    # Step 1‑2
    mh = minhash_for_text(text)
    bound_vec = fractional_power_binding(mh, power)

    # Initialise sheaf
    sheaf = Sheaf(node_dims, edge_list)

    # Identity restriction maps (each edge just copies the node vector)
    for (u, v) in edge_list:
        dim = node_dims[u]
        src_map = np.eye(dim, dtype=complex)
        dst_map = np.eye(dim, dtype=complex)
        sheaf.set_restriction((u, v), src_map, dst_map)

    # Step 3‑4: assign (possibly pruned) sections to each node
    for node in node_dims:
        sec = prune_section(bound_vec[: node_dims[node]], prune_prob)
        if sec is not None:
            sheaf.set_section(node, sec)

    # Step 5: evaluate
    inconsistency = sheaf.evaluate_cohomology()
    ssim_matrix = {}
    for (u, v) in edge_list:
        try:
            ssim_matrix[(u, v)] = compute_ssim_between_nodes(sheaf, u, v)
        except Exception:
            ssim_matrix[(u, v)] = 0.0

    return {
        "inconsistency": inconsistency,
        "ssim_matrix": ssim_matrix,
    }

# ---------- Smoke Test ----------
if __name__ == "__main__":
    # Simple graph: two nodes connected bidirectionally
    nodes = {0: 8, 1: 8}
    edges = [(0, 1), (1, 0)]

    result = hybrid_process(
        text="The quick brown fox jumps over the lazy dog",
        node_dims=nodes,
        edge_list=edges,
        power=0.6,
        prune_prob=0.8
    )
    print("Inconsistency vector (edge space):", result["inconsistency"])
    print("SSIM per edge:", result["ssim_matrix"])