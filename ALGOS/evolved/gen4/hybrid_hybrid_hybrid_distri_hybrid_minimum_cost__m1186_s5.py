# DARWIN HAMMER — match 1186, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s0.py (gen2)
# parent_b: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s1.py (gen3)
# born: 2026-05-29T23:33:31Z

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set

# ----------------------------------------------------------------------
# Geometry utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Point:
    x: float
    y: float


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Tuple[str, str]],
    root: str,
    path_weight: float = 0.2,
) -> float:
    """Total cost = material + weighted path‑to‑root cost."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        material += euclidean(nodes[u], nodes[v])

    # BFS from root to compute distances
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
# Cooling schedule (logarithmic, slower decay)
# ----------------------------------------------------------------------
def cooling_temperature(k: int, t0: float = 1.0, beta: float = 0.01) -> float:
    """T_k = t0 / log(k + 2 + beta).  Guarantees T_k > 0."""
    return t0 / math.log(k + 2.0 + beta)


# ----------------------------------------------------------------------
# Metropolis acceptance
# ----------------------------------------------------------------------
def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Standard Metropolis rule."""
    if delta_e <= 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


# ----------------------------------------------------------------------
# Union‑Find with path compression for cycle detection
# ----------------------------------------------------------------------
class UnionFind:
    def __init__(self, elements: Set[str]):
        self.parent: Dict[str, str] = {e: e for e in elements}
        self.rank: Dict[str, int] = {e: 0 for e in elements}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> bool:
        """Return False if x and y are already connected (cycle)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True


# ----------------------------------------------------------------------
# Bandit policy with dynamic reward range for Hoeffding bound
# ----------------------------------------------------------------------
class BanditPolicy:
    def __init__(self):
        # action_id -> [total_reward, count, min_reward, max_reward]
        self.stats: Dict[str, List[float]] = {}

    def reset(self) -> None:
        self.stats.clear()

    def _ensure(self, aid: str) -> None:
        if aid not in self.stats:
            # start with neutral reward 0 and infinite range placeholder
            self.stats[aid] = [0.0, 0.0, float("inf"), -float("inf")]

    def update(self, updates: List[Tuple[str, float]]) -> None:
        for aid, reward in updates:
            self._ensure(aid)
            s = self.stats[aid]
            s[0] += reward
            s[1] += 1.0
            s[2] = min(s[2], reward)
            s[3] = max(s[3], reward)

    def count(self, aid: str) -> int:
        self._ensure(aid)
        return int(self.stats[aid][1])

    def mean(self, aid: str) -> float:
        self._ensure(aid)
        total, n = self.stats[aid][0], self.stats[aid][1]
        return total / n if n > 0 else 0.0

    def reward_range(self, aid: str) -> float:
        self._ensure(aid)
        mn, mx = self.stats[aid][2], self.stats[aid][3]
        if mn == float("inf") or mx == -float("inf"):
            # No observations yet – use a conservative default range
            return 1.0
        return mx - mn if mx > mn else 1.0

    def hoeffding_bound(self, aid: str, delta: float) -> float:
        n = self.count(aid)
        if n == 0:
            return float("inf")
        r = self.reward_range(aid)
        return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

    def ucb(self, aid: str, delta: float) -> float:
        n = self.count(aid)
        if n == 0:
            return float("inf")
        return self.mean(aid) + self.hoeffding_bound(aid, delta)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def edge_id(u: str, v: str) -> str:
    """Canonical unordered identifier."""
    return f"{min(u, v)}-{max(u, v)}"


def select_edge_ucb(
    policy: BanditPolicy,
    candidate_edges: List[Tuple[str, str]],
    delta: float,
) -> Tuple[str, str]:
    best_edge = None
    best_score = -float("inf")
    for u, v in candidate_edges:
        aid = edge_id(u, v)
        score = policy.ucb(aid, delta)
        if score > best_score:
            best_score = score
            best_edge = (u, v)
    if best_edge is None:
        raise RuntimeError("No candidate edges available")
    return best_edge


def annealed_accept(
    current_cost: float,
    new_cost: float,
    temperature: float,
) -> bool:
    delta = new_cost - current_cost
    prob = acceptance_probability(delta, temperature)
    return random.random() < prob


def hybrid_grow_tree(
    nodes: Dict[str, Point],
    root: str,
    delta: float = 0.05,
    t0: float = 1.0,
    beta: float = 0.01,
    max_iters: int = 200,
) -> List[Tuple[str, str]]:
    """
    Build a spanning tree using a tighter Hoeffding‑UCB and a slower
    logarithmic cooling schedule.  The confidence term is scaled by the
    current temperature to let exploration fade together with annealing.
    """
    policy = BanditPolicy()
    tree_edges: List[Tuple[str, str]] = []
    all_nodes = set(nodes.keys())
    full_edges = [(u, v) for u in all_nodes for v in all_nodes if u < v]

    # Pre‑compute adjacency for quick connectivity checks
    uf_global = UnionFind(all_nodes)

    for k in range(max_iters):
        temperature = cooling_temperature(k, t0, beta)

        # Determine edges that keep the forest acyclic
        uf = UnionFind(all_nodes)
        for u, v in tree_edges:
            uf.union(u, v)

        candidate_edges = [
            (u, v)
            for u, v in full_edges
            if (u, v) not in tree_edges and uf.union(u, v)
        ]

        if not candidate_edges:
            break  # tree is complete

        # Edge selection via temperature‑scaled UCB
        # Scale confidence term by temperature to couple both mechanisms
        scaled_delta = delta * (temperature / (t0 + 1e-9))
        chosen_u, chosen_v = select_edge_ucb(policy, candidate_edges, scaled_delta)

        # Evaluate cost if edge were added
        tentative_edges = tree_edges + [(chosen_u, chosen_v)]
        new_cost = tree_cost(nodes, tentative_edges, root)

        current_cost = tree_cost(nodes, tree_edges, root) if tree_edges else 0.0

        if annealed_accept(current_cost, new_cost, temperature):
            tree_edges.append((chosen_u, chosen_v))
            # Reward = negative incremental cost (higher reward for cost reduction)
            reward = -(new_cost - current_cost)
            policy.update([(edge_id(chosen_u, chosen_v), reward)])

        # Early stop when a spanning tree is formed
        if len(tree_edges) == len(all_nodes) - 1:
            break

    return tree_edges