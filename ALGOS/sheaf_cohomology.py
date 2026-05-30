#!/usr/bin/env python3
"""Cellular sheaf cohomology over undirected graphs.

A cellular sheaf F assigns:
  - A vector space F(v) = R^{dim_v} to each node v.
  - A vector space F(e) = R^{dim_e} to each edge e = (u, v).
  - A restriction map F(u->e) : F(u) -> F(e) for each node-edge incidence.

The coboundary operator delta : C^0 -> C^1 measures local disagreement:
  delta(s)[e=(u,v)] = F(v->e) s_v - F(u->e) s_u

  C^0 = direct sum of all node spaces (global sections space).
  C^1 = direct sum of all edge spaces (cochains space).

If delta(s) = 0 everywhere the section s is globally consistent (a sheaf
global section).  ||delta(s)||^2 is the Laplacian energy / Dirichlet energy.

Sheaf Laplacian: L = delta^T delta (block-structured, positive semi-definite).
Its nullspace is the space of globally consistent sections.

References:
  Hansen & Ghrist, "Toward a Spectral Theory of Cellular Sheaves", 2019.
  Curry, "Sheaves, Cosheaves and Applications", 2014.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Sheaf class
# ---------------------------------------------------------------------------

class Sheaf:
    """Cellular sheaf over a graph.

    Parameters
    ----------
    node_dims : dict
        Mapping node_id -> dimension of the stalk (vector space) at that node.
    edge_list : list of (u, v) tuples
        Undirected edges; orientation is fixed as given (u = tail, v = head)
        for sign convention in the coboundary.
    """

    def __init__(self, node_dims, edge_list):
        self.node_dims = dict(node_dims)
        self.edges = list(edge_list)
        # restriction maps: (u, v) -> (src_map: R^{dim_u}->R^{d_e}, dst_map: R^{dim_v}->R^{d_e})
        self._restrictions = {}
        # local sections: node_id -> numpy array of shape (dim,)
        self._sections = {}

    def set_restriction(self, edge, src_map, dst_map):
        """Assign restriction maps for an oriented edge.

        Parameters
        ----------
        edge : (u, v)
            Must match an entry in edge_list with the same orientation.
        src_map : numpy array, shape (edge_dim, dim_u)
            Linear map F(u->e): stalk at u -> stalk at e.
        dst_map : numpy array, shape (edge_dim, dim_v)
            Linear map F(v->e): stalk at v -> stalk at e.
        """
        u, v = edge
        src_map = np.array(src_map, dtype=float)
        dst_map = np.array(dst_map, dtype=float)
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node, value):
        """Assign a local section to a node.

        Parameters
        ----------
        node : node_id
        value : array-like of shape (dim_node,)
        """
        self._sections[node] = np.array(value, dtype=float)

    def _edge_dim(self, u, v):
        """Return the stalk dimension at edge (u,v)."""
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        raise KeyError(f"No restriction map for edge ({u}, {v})")

    def _c0_layout(self):
        """Return ordered node list and cumulative offsets in C^0."""
        nodes = list(self.node_dims.keys())
        offsets = {}
        pos = 0
        for n in nodes:
            offsets[n] = pos
            pos += self.node_dims[n]
        return nodes, offsets, pos  # pos = total dim of C^0

    def _c1_layout(self):
        """Return ordered edge list and cumulative offsets in C^1."""
        offsets = {}
        pos = 0
        for e in self.edges:
            u, v = e
            d = self._edge_dim(u, v)
            offsets[e] = (pos, d)
            pos += d
        return offsets, pos  # pos = total dim of C^1

    def coboundary_operator(self):
        """Build the full coboundary matrix delta as a numpy array.

        Shape: (dim_C1, dim_C0).

        For edge e=(u,v) at block row [row_start:row_start+d_e]:
          delta[:, col_u:col_u+dim_u] += -F(u->e)
          delta[:, col_v:col_v+dim_v] += +F(v->e)
        """
        nodes, c0_off, c0_dim = self._c0_layout()
        c1_off, c1_dim = self._c1_layout()

        delta = np.zeros((c1_dim, c0_dim), dtype=float)

        for u, v in self.edges:
            row_start, d_e = c1_off[(u, v)]

            if (u, v) in self._restrictions:
                src_map, dst_map = self._restrictions[(u, v)]
            else:
                dst_map, src_map = self._restrictions[(v, u)]

            col_u = c0_off[u]
            col_v = c0_off[v]
            dim_u = self.node_dims[u]
            dim_v = self.node_dims[v]

            delta[row_start:row_start + d_e, col_u:col_u + dim_u] -= src_map
            delta[row_start:row_start + d_e, col_v:col_v + dim_v] += dst_map

        return delta

    def consistency_residual(self):
        """Compute delta(s) using current sections.

        Returns a 1-D array of length dim_C1 with the residual for each edge
        block concatenated in edge_list order.  Missing sections default to zero.
        """
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
        """Return ||delta(s)||^2 as a float."""
        r = self.consistency_residual()
        return float(np.dot(r, r))

    def inconsistent_edges(self, tol=1e-9):
        """Return list of (u, v, residual_norm) for edges with nonzero residual.

        Sorted by residual_norm descending.
        """
        residual = self.consistency_residual()
        c1_off, _ = self._c1_layout()

        results = []
        for u, v in self.edges:
            row_start, d_e = c1_off[(u, v)]
            block = residual[row_start:row_start + d_e]
            norm = float(np.linalg.norm(block))
            if norm > tol:
                results.append((u, v, norm))

        results.sort(key=lambda x: x[2], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Factory: identity / constant-fiber sheaf
# ---------------------------------------------------------------------------

def identity_sheaf(adj, dim):
    """Build a constant-fiber sheaf with identity restriction maps.

    Each node and edge stalk has the same dimension `dim`.  Both restriction
    maps for every edge are the dim×dim identity matrix — so delta(s)[e=(u,v)]
    = s_v - s_u.  Globally consistent sections are constant functions on each
    connected component.

    Parameters
    ----------
    adj : dict
        Adjacency dict: node_id -> list of neighbor node_ids.
    dim : int
        Stalk dimension at every node and edge.

    Returns
    -------
    Sheaf
    """
    node_dims = {n: dim for n in adj}
    # Deduplicate edges (use canonical u < v ordering based on sorted keys)
    all_nodes = sorted(adj.keys())
    seen = set()
    edge_list = []
    for u in all_nodes:
        for v in adj[u]:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                edge_list.append((u, v))

    sh = Sheaf(node_dims, edge_list)
    eye = np.eye(dim, dtype=float)
    for u, v in edge_list:
        sh.set_restriction((u, v), eye, eye)
    return sh


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Sheaf Cohomology Demo ===\n")

    # 4-node graph: 0-1-2-3 in a line plus edge 0-2
    #   0 -- 1 -- 2 -- 3
    #   |_________|
    adj = {
        0: [1, 2],
        1: [0, 2],
        2: [1, 3],
        3: [2],
    }
    dim = 2

    sh = identity_sheaf(adj, dim)

    # Consistent section: all nodes share the same vector
    consistent_val = np.array([3.0, -1.5])
    for n in adj:
        sh.set_section(n, consistent_val.copy())

    residual = sh.consistency_residual()
    incon = sh.global_inconsistency()
    print(f"Consistent sections (all nodes = {consistent_val}):")
    print(f"  global_inconsistency = {incon:.2e}")
    print(f"  inconsistent_edges   = {sh.inconsistent_edges()}")
    assert incon < 1e-12, "Expected zero inconsistency"
    print("  [PASS] delta(s) = 0 everywhere\n")

    # Break node 2: assign a different section
    broken_val = np.array([9.0, 7.0])
    sh.set_section(2, broken_val)

    incon2 = sh.global_inconsistency()
    bad_edges = sh.inconsistent_edges()
    print(f"After breaking node 2 (set to {broken_val}):")
    print(f"  global_inconsistency = {incon2:.4f}")
    print("  inconsistent_edges (u, v, residual_norm):")
    for u, v, norm in bad_edges:
        print(f"    edge ({u}, {v})  residual_norm = {norm:.6f}")

    assert incon2 > 1e-6, "Expected nonzero inconsistency"
    involved = {(u, v) for u, v, _ in bad_edges}
    # edges incident to node 2
    assert any(2 in e for e in involved), "Node 2 edges should be flagged"
    print("  [PASS] inconsistent_edges correctly identifies edges incident to broken node\n")

    # Show the coboundary matrix shape
    delta = sh.coboundary_operator()
    c0 = sum(sh.node_dims.values())
    c1 = sum(d for _, d in sh._c1_layout()[0].values())
    print(f"Coboundary matrix delta shape: {delta.shape}  (C1={c1} x C0={c0})")

    # Sheaf Laplacian L = delta^T delta
    L = delta.T @ delta
    evals = np.linalg.eigvalsh(L)
    print(f"Sheaf Laplacian eigenvalues: {np.round(evals, 4)}")
    print("  (zero eigenvalues count connected components in the trivial case)")
    print("\nDone.")
