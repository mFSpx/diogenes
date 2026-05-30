# DARWIN HAMMER — match 253, survivor 3
# gen: 3
# parent_a: minimum_cost_tree.py (gen0)
# parent_b: hybrid_hybrid_bandit_router_koopman_operator_m64_s0.py (gen2)
# born: 2026-05-29T23:27:57Z

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional


@dataclass(frozen=True)
class Point:
    """Immutable 2‑D point."""
    x: float
    y: float


@dataclass(frozen=True)
class BanditUpdate:
    """Observation used to update the hybrid policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float = 1.0  # not used in the current formulation but kept for compatibility


class HybridBanditTree:
    """
    A self‑contained hybrid of a minimum‑cost tree and a contextual bandit.
    The tree supplies a *distance‑modulated* confidence term for each (context, action)
    pair, making the integration deeper than a simple additive score.
    """

    def __init__(self,
                 nodes: Dict[str, Point],
                 edges: List[Tuple[str, str]],
                 root: str,
                 path_weight: float = 0.2,
                 confidence_alpha: float = 1.0):
        """
        Parameters
        ----------
        nodes
            Mapping from node identifier to its geometric location.
        edges
            Undirected edges forming a tree (or forest). Cycles are ignored.
        root
            Identifier of the root node; distances are measured from it.
        path_weight
            Weight applied to the sum of root‑to‑node distances in the tree cost.
        confidence_alpha
            Scaling factor for the confidence term; larger values increase exploration.
        """
        self._validate_graph(nodes, edges, root)
        self.nodes = nodes
        self.root = root
        self.path_weight = path_weight
        self.confidence_alpha = confidence_alpha

        # adjacency list (undirected)
        self.adj: Dict[str, List[str]] = {n: [] for n in nodes}
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)

        # pre‑compute distances from the root (single‑source shortest paths)
        self.root_distances = self._bfs_distances(root)

        # statistics: (context, action) → [total_reward, count]
        self._stats: Dict[Tuple[str, str], List[float]] = {}

    # --------------------------------------------------------------------- #
    #                     Graph utilities and validation                  #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _validate_graph(nodes: Dict[str, Point],
                        edges: List[Tuple[str, str]],
                        root: str) -> None:
        if root not in nodes:
            raise ValueError(f"Root '{root}' is not present in the node set.")
        for a, b in edges:
            if a not in nodes or b not in nodes:
                raise ValueError(f"Edge ({a}, {b}) references undefined node(s).")

    def _bfs_distances(self, start: str) -> Dict[str, float]:
        """Breadth‑first search returning Euclidean distance from *start* to every reachable node."""
        distances: Dict[str, float] = {start: 0.0}
        frontier: List[str] = [start]
        while frontier:
            cur = frontier.pop()
            for nxt in self.adj[cur]:
                if nxt not in distances:
                    distances[nxt] = distances[cur] + self._euclidean(self.nodes[cur], self.nodes[nxt])
                    frontier.append(nxt)
        return distances

    @staticmethod
    def _euclidean(a: Point, b: Point) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    # --------------------------------------------------------------------- #
    #                     Bandit statistics helpers                         #
    # --------------------------------------------------------------------- #
    def _reward(self, ctx: str, act: str) -> float:
        total, cnt = self._stats.get((ctx, act), [0.0, 0.0])
        return total / cnt if cnt > 0 else 0.0

    def _count(self, ctx: str, act: str) -> float:
        return self._stats.get((ctx, act), [0.0, 0.0])[1]

    def _total_observations(self) -> float:
        """Total number of observations across all (context, action) pairs."""
        return sum(cnt for _, cnt in self._stats.values())

    def _confidence(self, ctx: str, act: str) -> float:
        """
        Upper‑confidence bound term based on Hoeffding’s inequality.
        The distance from the root modulates the exploration bonus:
            conf = α * sqrt( 2 * log(N) / n ) * exp( -d(root, ctx) )
        where N is the total number of observations, n the count for this pair,
        and d the tree distance.
        """
        n = self._count(ctx, act)
        if n == 0:
            # maximal uncertainty for unseen pairs
            return float('inf')
        N = max(self._total_observations(), 1.0)
        base = math.sqrt(2.0 * math.log(N) / n)
        dist = self.root_distances.get(ctx, 0.0)
        distance_factor = math.exp(-dist)  # nearer contexts get a larger bonus
        return self.confidence_alpha * base * distance_factor

    # --------------------------------------------------------------------- #
    #                     Public API                                         #
    # --------------------------------------------------------------------- #
    def update(self, updates: Iterable[BanditUpdate]) -> None:
        """Incorporate a batch of observations."""
        for u in updates:
            key = (u.context_id, u.action_id)
            stats = self._stats.setdefault(key, [0.0, 0.0])
            stats[0] += float(u.reward)
            stats[1] += 1.0

    def tree_cost(self) -> float:
        """
        Minimum‑cost tree score.
        Material cost = sum of Euclidean edge lengths.
        Path cost   = path_weight * Σ_{v} dist(root, v).
        """
        material = sum(self._euclidean(self.nodes[a], self.nodes[b]) for a, b in self._edge_iter())
        path = self.path_weight * sum(self.root_distances.values())
        return material + path

    def _edge_iter(self) -> Iterable[Tuple[str, str]]:
        """Yield each undirected edge once (the first occurrence in adjacency)."""
        seen = set()
        for a, neighbours in self.adj.items():
            for b in neighbours:
                if (b, a) not in seen:
                    seen.add((a, b))
                    yield a, b

    def bandit_score(self) -> float:
        """
        Sum of mean rewards plus confidence bonuses for all observed (context, action) pairs.
        Unobserved pairs contribute only their (infinite) confidence term,
        which is capped to a large finite value to keep the score numeric.
        """
        score = 0.0
        for (ctx, act), (total, cnt) in self._stats.items():
            mean = total / cnt if cnt > 0 else 0.0
            conf = self._confidence(ctx, act)
            if math.isinf(conf):
                conf = 1e6  # pragmatic cap for unseen pairs
            score += mean + conf
        return score

    def hybrid_score(self) -> float:
        """Combined objective: tree cost + bandit score."""
        return self.tree_cost() + self.bandit_score()

    def step(self,
             updates: Optional[Iterable[BanditUpdate]] = None,
             verbose: bool = False) -> float:
        """
        Perform a single update step and return the new hybrid score.

        Parameters
        ----------
        updates
            Optional iterable of BanditUpdate objects.
        verbose
            If True, prints intermediate diagnostics.
        """
        if updates is not None:
            self.update(updates)

        t_cost = self.tree_cost()
        b_score = self.bandit_score()
        h_score = t_cost + b_score

        if verbose:
            print(f"[HybridBanditTree] Tree cost: {t_cost:.4f}")
            print(f"[HybridBanditTree] Bandit score: {b_score:.4f}")
            print(f"[HybridBanditTree] Hybrid score: {h_score:.4f}")

        return h_score


# ------------------------------------------------------------------------- #
# Example usage (executed only when run as a script)                        #
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Define a tiny tree
    nodes = {
        "A": Point(0.0, 0.0),
        "B": Point(1.0, 1.0),
        "C": Point(2.0, 2.0)
    }
    edges = [("A", "B"), ("B", "C")]
    root = "A"

    # Initialise the hybrid system
    hybrid = HybridBanditTree(nodes, edges, root, path_weight=0.2, confidence_alpha=1.0)

    # Simulate a few bandit observations
    obs = [
        BanditUpdate(context_id="A", action_id="act1", reward=1.0),
        BanditUpdate(context_id="B", action_id="act1", reward=0.5),
        BanditUpdate(context_id="C", action_id="act2", reward=0.8)
    ]

    # Perform an update step and display the score
    hybrid.step(obs, verbose=True)