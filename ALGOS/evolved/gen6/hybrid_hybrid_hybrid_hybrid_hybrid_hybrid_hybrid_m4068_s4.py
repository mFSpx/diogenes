# DARWIN HAMMER — match 4068, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_minimu_m662_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_percep_m1570_s0.py (gen5)
# born: 2026-05-29T23:53:25Z

"""Hybrid Sketch‑Geometric‑Fisher Algorithm
========================================

This module fuses the two parent algorithms:

* **Parent A** – count‑min sketch + MinHash signatures + Bayesian epistemic
  edge cost (see *hybrid_hybrid_hybrid_sketch_hybrid_hybrid_minimu_m662_s3.py*).
* **Parent B** – geometric‑algebra multivector representation together with a
  Fisher‑information weighted similarity used inside an RBF surrogate model
  (see *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_percep_m1570_s0.py*).

**Mathematical bridge**

A node’s high‑dimensional item multiset is first compressed with a
count‑min sketch.  The sketch matrix is flattened and interpreted as the
coefficients of a grade‑1 multivector in a Clifford algebra
`Cl(N,0)`, where `N = width*depth`.  This multivector provides a geometric
embedding of the sketch.

Similarity between two nodes is evaluated in two complementary ways:

1. **Set‑based similarity** – Jaccard estimate derived from MinHash
   signatures (Parent A).
2. **Metric‑based similarity** – Fisher‑information quadratic form applied
   to the multivector vectors (Parent B).

Both similarities feed the Bayesian update of the epistemic confidence
`p`.  The final hybrid edge cost combines Euclidean distance, the
set‑based Jaccard term, the Fisher‑metric term, and the posterior
confidence:


d   = Euclidean distance between node positions
j   = Jaccard similarity from MinHash
f   = Fisher‑metric similarity  (0..1)
p   = prior epistemic confidence (0..1)

L   = j * f                     # joint likelihood from both parents
M   = bayes_marginal(p, L, fp)  # fp = false‑positive rate (hyper‑parameter)
p'  = bayes_update(p, L, M)     # posterior confidence

cost = d * (1 - L) * (1 - p')


The functions below implement the full pipeline and expose three public
operations:

* `hybrid_node_representation` – builds the count‑min sketch, MinHash
  signature and multivector for a node.
* `hybrid_edge_cost` – computes the hybrid cost between two nodes.
* `build_hybrid_minimum_spanning_tree` – builds an MST using Kruskal’s
  algorithm on the hybrid costs.

All code relies only on the Python standard library and NumPy."""

import hashlib
import math
import random
import sys
import pathlib
from collections import defaultdict
import numpy as np

# ----------------------------------------------------------------------
# Helper utilities (Parent A)
# ----------------------------------------------------------------------
def count_min_sketch(items, width: int = 64, depth: int = 4):
    """Return a depth×width count‑min sketch matrix for *items*."""
    sketch = np.zeros((depth, width), dtype=np.int64)
    # simple pairwise‑independent hash functions via salted SHA‑256
    for item in items:
        item_bytes = str(item).encode('utf-8')
        for d in range(depth):
            hasher = hashlib.sha256()
            hasher.update(item_bytes)
            hasher.update(d.to_bytes(2, 'little'))
            idx = int(hasher.hexdigest(), 16) % width
            sketch[d, idx] += 1
    return sketch


def minhash_signature(items, num_perm: int = 128):
    """Return a MinHash signature (list of ints) for *items*."""
    max_hash = (1 << 32) - 1
    signature = [max_hash] * num_perm
    for item in items:
        item_bytes = str(item).encode('utf-8')
        for i in range(num_perm):
            hasher = hashlib.sha256()
            hasher.update(item_bytes)
            hasher.update(i.to_bytes(2, 'little'))
            hv = int(hasher.hexdigest(), 16) & max_hash
            if hv < signature[i]:
                signature[i] = hv
    return signature


def jaccard_from_minhash(sig_a, sig_b):
    """Estimate Jaccard similarity from two MinHash signatures."""
    assert len(sig_a) == len(sig_b)
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def bayes_marginal(p, L, fp):
    """Marginal probability of observing likelihood L given prior p and false‑positive rate fp."""
    # Simple model: true positive = L, false positive = fp
    return p * L + (1 - p) * fp


def bayes_update(p, L, M):
    """Posterior confidence after observing likelihood L with marginal M."""
    if M == 0:
        return 0.0
    return (p * L) / M


# ----------------------------------------------------------------------
# Geometric‑Algebra utilities (Parent B)
# ----------------------------------------------------------------------
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j:j + 2]
                n -= 2
                sign *= 1
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Grade‑1 multivector (vector) in Cl(n,0) built from a coefficient dict."""

    def __init__(self, components: dict):
        # components: {int index -> float coefficient}
        self.components = {int(k): float(v) for k, v in components.items()}

    def __add__(self, other):
        res = self.components.copy()
        for k, v in other.components.items():
            res[k] = res.get(k, 0.0) + v
        return Multivector(res)

    def __sub__(self, other):
        res = self.components.copy()
        for k, v in other.components.items():
            res[k] = res.get(k, 0.0) - v
        return Multivector(res)

    def to_vector(self, dim):
        """Return a dense NumPy vector of length *dim*."""
        vec = np.zeros(dim, dtype=np.float64)
        for idx, coeff in self.components.items():
            if 0 <= idx < dim:
                vec[idx] = coeff
        return vec

    def __repr__(self):
        return f"Multivector({self.components})"


def fisher_similarity(mv_a: Multivector, mv_b: Multivector, fisher_mat: np.ndarray):
    """Fisher‑information based similarity (0..1) between two multivectors."""
    diff = mv_a.to_vector(fisher_mat.shape[0]) - mv_b.to_vector(fisher_mat.shape[0])
    mahalanobis = diff @ fisher_mat @ diff
    # Convert to a similarity in [0,1]; larger distance -> smaller similarity
    return math.exp(-0.5 * mahalanobis)


# ----------------------------------------------------------------------
# Hybrid core (integration of both parents)
# ----------------------------------------------------------------------
def hybrid_node_representation(items,
                               width: int = 64,
                               depth: int = 4,
                               num_perm: int = 128):
    """
    Build the hybrid representation of a node.

    Returns a dict with keys:
        'sketch'   – count‑min sketch (numpy array)
        'signature'– MinHash signature (list of ints)
        'multivector' – grade‑1 Multivector built from the flattened sketch
    """
    sketch = count_min_sketch(items, width, depth)
    signature = minhash_signature(items, num_perm)

    # Flatten sketch column‑major and use each entry as a coefficient.
    flat = sketch.flatten()
    components = {i: float(v) for i, v in enumerate(flat) if v != 0}
    mv = Multivector(components)

    return {
        'sketch': sketch,
        'signature': signature,
        'multivector': mv,
    }


def hybrid_edge_cost(node_a_repr,
                     node_b_repr,
                     pos_a,
                     pos_b,
                     epistemic_p: float,
                     fisher_mat: np.ndarray,
                     false_positive_rate: float = 0.01):
    """
    Compute the hybrid edge cost between two nodes.

    Parameters
    ----------
    node_*_repr : dict
        Output of ``hybrid_node_representation``.
    pos_* : array‑like of length 2 or 3
        Euclidean coordinates of the nodes.
    epistemic_p : float
        Prior epistemic confidence (0..1).
    fisher_mat : np.ndarray
        Positive‑definite Fisher information matrix (size = sketch_dim).
    false_positive_rate : float
        Hyper‑parameter for the Bayesian marginal.

    Returns
    -------
    cost : float
    """
    # 1. Euclidean distance
    d = math.dist(pos_a, pos_b)

    # 2. Jaccard similarity from MinHash
    j = jaccard_from_minhash(node_a_repr['signature'],
                             node_b_repr['signature'])

    # 3. Fisher‑metric similarity from multivectors
    f = fisher_similarity(node_a_repr['multivector'],
                          node_b_repr['multivector'],
                          fisher_mat)

    # Joint likelihood (product because we treat the two evidences as independent)
    L = j * f

    # Bayesian update
    M = bayes_marginal(epistemic_p, L, false_positive_rate)
    p_post = bayes_update(epistemic_p, L, M)

    # Hybrid cost
    cost = d * (1.0 - L) * (1.0 - p_post)
    return cost


def build_hybrid_minimum_spanning_tree(nodes,
                                       positions,
                                       items_per_node,
                                       epistemic_flags,
                                       width: int = 64,
                                       depth: int = 4,
                                       num_perm: int = 128,
                                       false_positive_rate: float = 0.01):
    """
    Construct a Minimum‑Spanning Tree (MST) over *nodes* using the hybrid edge cost.

    Parameters
    ----------
    nodes : iterable of hashable identifiers
    positions : dict mapping node -> coordinate (list/tuple of floats)
    items_per_node : dict mapping node -> iterable of items (the high‑dimensional multiset)
    epistemic_flags : dict mapping node -> prior epistemic confidence (0..1)

    Returns
    -------
    mst_edges : list of tuples (node_u, node_v, cost)
    """
    # 1. Build hybrid representations for every node
    reprs = {}
    for n in nodes:
        reprs[n] = hybrid_node_representation(
            items_per_node[n],
            width=width,
            depth=depth,
            num_perm=num_perm,
        )

    # 2. Prepare Fisher matrix (identity scaled by average sketch count + epsilon)
    sketch_dim = width * depth
    avg_count = np.mean([r['sketch'].sum() for r in reprs.values()]) + 1e-6
    fisher_mat = np.eye(sketch_dim) * (1.0 / avg_count)

    # 3. Compute all pairwise hybrid costs (dense O(N^2) for simplicity)
    node_list = list(nodes)
    edges = []
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            u = node_list[i]
            v = node_list[j]
            cost = hybrid_edge_cost(
                reprs[u],
                reprs[v],
                positions[u],
                positions[v],
                epistemic_p=epistemic_flags.get(u, 0.5),  # use u's prior
                fisher_mat=fisher_mat,
                false_positive_rate=false_positive_rate,
            )
            edges.append((u, v, cost))

    # 4. Kruskal's algorithm
    parent = {n: n for n in node_list}
    rank = {n: 0 for n in node_list}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        xr, yr = find(x), find(y)
        if xr == yr:
            return False
        if rank[xr] < rank[yr]:
            parent[xr] = yr
        elif rank[xr] > rank[yr]:
            parent[yr] = xr
        else:
            parent[yr] = xr
            rank[xr] += 1
        return True

    edges.sort(key=lambda e: e[2])  # sort by cost
    mst = []
    for u, v, cost in edges:
        if union(u, v):
            mst.append((u, v, cost))
        if len(mst) == len(node_list) - 1:
            break

    return mst


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic dataset
    random.seed(42)
    np.random.seed(42)

    nodes = ['A', 'B', 'C', 'D']
    positions = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.2),
        'C': (0.5, 0.9),
        'D': (1.2, 1.1),
    }

    # Each node holds a multiset of integer "items"
    items_per_node = {
        'A': [random.randint(0, 20) for _ in range(30)],
        'B': [random.randint(0, 20) for _ in range(30)],
        'C': [random.randint(0, 20) for _ in range(30)],
        'D': [random.randint(0, 20) for _ in range(30)],
    }

    epistemic_flags = {n: random.uniform(0.4, 0.9) for n in nodes}

    mst = build_hybrid_minimum_spanning_tree(
        nodes,
        positions,
        items_per_node,
        epistemic_flags,
        width=32,
        depth=2,
        num_perm=64,
    )

    print("Hybrid MST edges (u, v, cost):")
    for u, v, c in mst:
        print(f"{u} – {v} : {c:.4f}")

    # Verify that the MST connects all nodes
    assert len(mst) == len(nodes) - 1, "MST does not span all nodes"
    print("Smoke test passed.")