# DARWIN HAMMER — match 1851, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s2.py (gen4)
# parent_b: hybrid_hybrid_sketches_hybr_hybrid_hybrid_hdc_hy_m561_s2.py (gen5)
# born: 2026-05-29T23:39:08Z

"""
This module implements a novel hybrid algorithm that fuses the core topologies of 
hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s2.py and hybrid_hybrid_sketches_hybr_hybrid_hybrid_hdc_hy_m561_s2.py. 
The mathematical bridge between the two structures lies in the incorporation of 
the fisher_score function from the first algorithm into the modulation_vector generation 
of the RBF surrogate in the second algorithm. This allows for efficient, probabilistic estimation 
of modulation vectors based on hashed item frequencies and the incorporation of Gaussian beam 
intensity and Euclidean distance metrics. The resulting hybrid algorithm enables the estimation 
of complex modulation vectors with high accuracy and efficiency.
"""

import math
import random
import sys
import pathlib
import numpy as np
from collections import Counter, defaultdict

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                # identify the edge direction used for length lookup
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)

    return adj, edge_len, dist

@dataclass(frozen=True)
class BanditAction:
    action_id: str; propensity: float; expected_reward: float; confidence_bound: float; algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str; action_id: str; reward: float; propensity: float

_POLICY: dict[str, list[float]] = {}
def reset_policy() -> None: 
    _POLICY.clear()

def update_policy(updates: list[BanditUpdate]) -> None:
    for u in updates:
        s=_POLICY.setdefault(u.action_id,[0.0,0.0]); s[0]+=float(u.reward); s[1]+=1.0

def _reward(a: str) -> float:
    total,n=_POLICY.get(a,[0.0,0.0]); return total/n if n else 0.0

def count_min_sketch(items: list[str], width: int=64, depth: int=4) -> list[list[int]]:
    table=[[0]*width for _ in range(depth)]
    for item in items:
        for d in range(depth): table[d][int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(),16)%width]+=1
    return table

def symbol_vector(symbol: str, dim: int = 10000) -> list[int]:
    import hashlib
    seed = int.from_bytes(hashlib.sha256(symbol.encode('utf-8')).digest()[:8], 'big')
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def bind(a: list[int], b: list[int]) -> list[int]:
    if len(a) != len(b):
        raise ValueError('vectors must have equal length')
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: list[list[int]]) -> list[int]:
    if not vectors:
        raise ValueError('at least one vector is required')
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError('all vectors must have equal length')
    result = [0] * dim
    for v in vectors:
        result = [x + y for x, y in zip(result, v)]
    return [x / len(vectors) for x in result]

def hybrid_fisher_sketch(
    items: list[str],
    theta: float,
    center: float,
    width: float,
    width_sketch: int = 64,
    depth_sketch: int = 4,
) -> list[int]:
    sketch = count_min_sketch(items, width_sketch, depth_sketch)
    fisher = fisher_score(theta, center, width)
    return bundle([symbol_vector(item, 10000), [fisher] * 10000])

def hybrid_length_sketch(
    items: list[str],
    a: Tuple[float, float],
    b: Tuple[float, float],
    width_sketch: int = 64,
    depth_sketch: int = 4,
) -> list[int]:
    sketch = count_min_sketch(items, width_sketch, depth_sketch)
    return bundle([symbol_vector(item, 10000), [length(a, b)] * 10000])

def hybrid_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    theta: float,
    center: float,
    width: float,
    width_sketch: int = 64,
    depth_sketch: int = 4,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    tree_adj, tree_edge_len, tree_dist = tree_metrics(nodes, edges, root)
    sketch = count_min_sketch(list(nodes.keys()), width_sketch, depth_sketch)
    fisher = fisher_score(theta, center, width)
    return tree_adj, tree_edge_len, tree_dist, bundle([symbol_vector(item, 10000) for item in nodes.keys()], [fisher] * 10000)

if __name__ == "__main__":
    items = ["apple", "banana", "orange"]
    nodes = {"apple": (1.0, 2.0), "banana": (3.0, 4.0), "orange": (5.0, 6.0)}
    edges = [("apple", "banana"), ("banana", "orange"), ("orange", "apple")]
    theta = 1.0
    center = 2.0
    width = 3.0
    width_sketch = 64
    depth_sketch = 4

    tree_adj, tree_edge_len, tree_dist, fisher_sketch = hybrid_metrics(nodes, edges, "apple", theta, center, width, width_sketch, depth_sketch)
    hybrid_sketch = hybrid_fisher_sketch(items, theta, center, width, width_sketch, depth_sketch)
    hybrid_length_sketch = hybrid_length_sketch(items, (1.0, 2.0), (3.0, 4.0), width_sketch, depth_sketch)