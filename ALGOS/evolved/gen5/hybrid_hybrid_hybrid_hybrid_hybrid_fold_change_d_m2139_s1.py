# DARWIN HAMMER — match 2139, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s5.py (gen4)
# parent_b: hybrid_fold_change_detectio_hybrid_regret_engine_m1100_s0.py (gen3)
# born: 2026-05-29T23:41:00Z

"""Hybrid algorithm combining tree‑cost optimization (Parent A) with fold‑change
detection and regret‑style updates (Parent B).

Mathematical bridge:
- The *tree cost* computed from a geometric spanning tree is treated as a
  global signal `u`.  Changes in this cost (`Δcost`) drive the fold‑change
  detection update (`step`) that adapts the expected values of actions
  attached to each node.
- Updated expected values are used as a *risk‑aware* weight when proposing
  edge modifications.  Acceptance of a new edge set follows the Metropolis
  rule with a logarithmic cooling schedule.
- The Gini coefficient evaluates the fairness of the distribution of
  expected values after each iteration, providing a regret‑style metric.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Geometry utilities (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Point:
    x: float
    y: float


def euclidean(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a.x - b.x, a.y - b.y)


def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Tuple[str, str]],
    root: str,
    path_weight: float = 0.2,
) -> float:
    """
    Total cost = material (sum of edge lengths) + weighted path‑to‑root cost.
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        material += euclidean(nodes[u], nodes[v])

    # BFS/DFS from root to compute distance to root for every node
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + euclidean(nodes[cur], nodes[nxt])
                stack.append(nxt)

    path_cost = sum(dist.values())
    return material + path_weight * path_cost


# ----------------------------------------------------------------------
# Cooling schedule & Metropolis acceptance (Parent A)
# ----------------------------------------------------------------------
def cooling_temperature(k: int, t0: float = 1.0, beta: float = 0.01) -> float:
    """Logarithmic cooling: T_k = t0 / log(k + 2 + beta)."""
    return t0 / math.log(k + 2.0 + beta)


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance probability."""
    if delta_e <= 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


# ----------------------------------------------------------------------
# Union‑Find for cycle detection (Parent A)
# ----------------------------------------------------------------------
class UnionFind:
    def __init__(self, elements: Set[str]):
        self.parent: Dict[str, str] = {e: e for e in elements}
        self.rank: Dict[str, int] = {e: 0 for e in elements}

    def find(self, x: str) -> str:
        """Path‑compressed find."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        """Union two sets; returns False if they are already connected."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


# ----------------------------------------------------------------------
# Fold‑change detection step (Parent B)
# ----------------------------------------------------------------------
def step(
    u: float,
    x: float,
    y: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay_x: float = 1.0,
    decay_y: float = 1.0,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """Euler integration of the feed‑forward fold‑change detector."""
    if dt < 0:
        raise ValueError("dt must be non‑negative")
    ratio = u / max(abs(x), eps)
    dy = gain * ratio - decay_y * y
    dx = u - decay_x * x
    return x + dt * dx, y + dt * dy


# ----------------------------------------------------------------------
# Gini coefficient (Parent B)
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for a non‑negative distribution."""
    xs = sorted(float(v) for v in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


# ----------------------------------------------------------------------
# Data structures for actions (Parent B)
# ----------------------------------------------------------------------
@dataclass
class MathAction:
    """Action attached to a node."""
    id: str
    expected_value: float = 0.0
    cost: float = 0.0
    risk: float = 0.0
    # internal state for fold‑change detection
    _x: float = field(default=0.0, init=False, repr=False)
    _y: float = field(default=0.0, init=False, repr=False)


# ----------------------------------------------------------------------
# Core hybrid functions (new)
# ----------------------------------------------------------------------
def update_actions_with_cost_change(
    actions: Dict[str, MathAction], delta_cost: float, dt: float = 1.0
) -> None:
    """
    Treat the change in global tree cost as the input signal `u` for every
    action and apply the fold‑change detection dynamics to its expected value.
    The internal state (`_x`, `_y`) is kept per action.
    """
    for act in actions.values():
        # Use current expected value as `x` seed; keep momentum `y`
        new_x, new_y = step(
            u=delta_cost,
            x=act._x if act._x != 0 else act.expected_value,
            y=act._y,
            dt=dt,
        )
        act._x, act._y = new_x, new_y
        act.expected_value = new_x  # map the detector state back to expected value


def propose_edge_modification(
    edges: List[Tuple[str, str]],
    nodes: Set[str],
    rng: random.Random,
) -> Tuple[List[Tuple[str, str]], bool]:
    """
    Randomly propose a single edge swap:
      - remove a random existing edge,
      - add a new edge that does **not** create a cycle.
    Returns the new edge list and a flag indicating whether a valid proposal
    was generated.
    """
    if len(edges) < 2:
        return edges, False

    # Remove a random edge
    remove_idx = rng.randrange(len(edges))
    removed_edge = edges[remove_idx]
    candidate_edges = edges[:remove_idx] + edges[remove_idx + 1 :]

    # Build UnionFind on the remaining edges
    uf = UnionFind(nodes)
    for u, v in candidate_edges:
        uf.union(u, v)

    # Try to add a new edge that connects two different components
    possible = [(a, b) for a in nodes for b in nodes if a < b and uf.find(a) != uf.find(b)]
    if not possible:
        return edges, False

    new_edge = rng.choice(possible)
    candidate_edges.append(new_edge)
    return candidate_edges, True


def hybrid_iteration(
    k: int,
    nodes: Dict[str, Point],
    actions: Dict[str, MathAction],
    edges: List[Tuple[str, str]],
    root: str,
    prev_cost: float,
    rng: random.Random,
) -> Tuple[float, List[Tuple[str, str]], float]:
    """
    Perform one hybrid iteration:
      1. Compute current tree cost.
      2. Update action expected values using the cost change (fold‑change).
      3. Propose an edge modification and decide acceptance via Metropolis
         with a logarithmic cooling schedule.
      4. Return the (possibly) new cost, edge list, and Gini coefficient of
         expected values.
    """
    current_cost = tree_cost(nodes, edges, root)

    delta = current_cost - prev_cost if prev_cost is not None else 0.0
    update_actions_with_cost_change(actions, delta)

    # Propose a new edge configuration
    new_edges, ok = propose_edge_modification(edges, set(nodes.keys()), rng)
    if not ok:
        # No valid proposal – keep current state
        return current_cost, edges, gini_coefficient([a.expected_value for a in actions.values()])

    new_cost = tree_cost(nodes, new_edges, root)

    # Metropolis acceptance
    temp = cooling_temperature(k)
    accept_prob = acceptance_probability(new_cost - current_cost, temp)
    if rng.random() < accept_prob:
        # Accept the new configuration
        return new_cost, new_edges, gini_coefficient([a.expected_value for a in actions.values()])
    else:
        # Reject – keep old configuration
        return current_cost, edges, gini_coefficient([a.expected_value for a in actions.values()])


# ----------------------------------------------------------------------
# Utility to generate a random connected tree (helper)
# ----------------------------------------------------------------------
def random_spanning_tree(node_ids: List[str], rng: random.Random) -> List[Tuple[str, str]]:
    """Kruskal‑style random tree generator."""
    uf = UnionFind(set(node_ids))
    edges: List[Tuple[str, str]] = []
    possible = [(a, b) for i, a in enumerate(node_ids) for b in node_ids[i + 1 :]]
    rng.shuffle(possible)
    for u, v in possible:
        if uf.union(u, v):
            edges.append((u, v))
        if len(edges) == len(node_ids) - 1:
            break
    return edges


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = random.Random(42)

    # Create a tiny geometric graph (5 nodes)
    node_ids = [f"N{i}" for i in range(5)]
    nodes = {
        nid: Point(x=rng.uniform(0, 10), y=rng.uniform(0, 10)) for nid in node_ids
    }

    # Attach a MathAction to each node
    actions = {
        nid: MathAction(id=nid, expected_value=rng.uniform(0, 5)) for nid in node_ids
    }

    # Initial random spanning tree
    edges = random_spanning_tree(node_ids, rng)
    root = node_ids[0]

    prev_cost = None
    print("Initial tree cost:", tree_cost(nodes, edges, root))
    print("Initial Gini:", gini_coefficient([a.expected_value for a in actions.values()]))

    # Run a few hybrid iterations
    for k in range(1, 11):
        cost, edges, gini = hybrid_iteration(
            k=k,
            nodes=nodes,
            actions=actions,
            edges=edges,
            root=root,
            prev_cost=prev_cost if prev_cost is not None else cost,
            rng=rng,
        )
        prev_cost = cost
        print(
            f"Iter {k:02d} | Cost={cost:.3f} | Gini={gini:.3f} | Edges={len(edges)}"
        )
    sys.exit(0)