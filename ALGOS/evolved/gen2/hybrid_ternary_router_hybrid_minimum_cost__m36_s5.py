# DARWIN HAMMER — match 36, survivor 5
# gen: 2
# parent_a: ternary_router.py (gen0)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s1.py (gen1)
# born: 2026-05-29T23:23:31Z

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Compatibility shim for the original FairyFuse backend.
# ----------------------------------------------------------------------
try:
    from services.fairyfuse.fairyfuse_backend import resident_engine_from_env, route_command  # type: ignore
except Exception:  # pragma: no cover
    class _DummyEngine:
        def status(self) -> Dict[str, Any]:
            return {"state": "dummy", "uptime_s": 0}

    def resident_engine_from_env() -> _DummyEngine:  # noqa: D401
        """Return a dummy engine with a ``status`` method."""
        return _DummyEngine()

    def route_command(text: str, intent: str, context: Dict[str, Any]) -> Any:
        """Return a simple namespace mimicking the real route result."""
        class _Result:
            def to_dict(self) -> Dict[str, Any]:
                return {
                    "text": text[:100],
                    "intent": intent,
                    "context": context,
                    "confidence": random.random(),
                }

        return _Result()


# ----------------------------------------------------------------------
# Core utilities shared by both parent algorithms.
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNTIME_DIR = ROOT / "04_RUNTIME" / "fairyfuse"
DEFAULT_HEARTBEAT = RUNTIME_DIR / "ternary_router_heartbeat.jsonl"


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def emit_json(obj: Any) -> None:
    """Print a JSON object in a deterministic order."""
    print(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))


# ----------------------------------------------------------------------
# Geometry helpers.
# ----------------------------------------------------------------------
Point = Tuple[float, float]
EdgeKey = FrozenSet[str]  # unordered pair representing an edge


def _euclidean_length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Union‑Find data structure for Kruskal's MST algorithm.
# ----------------------------------------------------------------------
class _UnionFind:
    def __init__(self, elements: List[str]) -> None:
        self.parent: Dict[str, str] = {e: e for e in elements}
        self.rank: Dict[str, int] = {e: 0 for e in elements}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> bool:
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True


# ----------------------------------------------------------------------
# Graph construction.
# ----------------------------------------------------------------------
ENGINE_CHANNELS = [
    "cpu_fairyfuse_ternary",
    "gpu_fairyfuse_binary",
    "tpu_fairyfuse_quantized",
]


def _build_graph() -> Tuple[Dict[str, Point], List[Tuple[str, str]], Dict[EdgeKey, float]]:
    """
    Returns:
        nodes: mapping channel → (x, y) placed on a unit circle.
        edges: list of ordered pairs (a, b) for every undirected connection.
        priors: mapping unordered edge key → prior probability (initially 0.5).
    """
    n = len(ENGINE_CHANNELS)
    angle_step = 2 * math.pi / n
    nodes: Dict[str, Point] = {}
    for i, ch in enumerate(ENGINE_CHANNELS):
        theta = i * angle_step
        nodes[ch] = (math.cos(theta), math.sin(theta))

    edges: List[Tuple[str, str]] = []
    priors: Dict[EdgeKey, float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ENGINE_CHANNELS[i], ENGINE_CHANNELS[j]
            edges.append((a, b))
            priors[frozenset({a, b})] = 0.5
    return nodes, edges, priors


# ----------------------------------------------------------------------
# Bayesian update utilities.
# ----------------------------------------------------------------------
def bayes_update_edge_priors(
    priors: Dict[EdgeKey, float],
    evidence: Dict[EdgeKey, float],
    generic_likelihood: float = 0.8,
    false_positive: float = 0.1,
) -> Dict[EdgeKey, float]:
    """
    Perform a Bayesian update for each edge prior.

    Args:
        priors: current prior probabilities.
        evidence: per‑edge observed success frequency in [0, 1].
        generic_likelihood: baseline likelihood of a true positive.
        false_positive: baseline likelihood of a false positive.

    Returns:
        Updated posterior probabilities (still in [0, 1]).
    """
    updated: Dict[EdgeKey, float] = {}
    for edge, prior in priors.items():
        obs = evidence.get(edge, 1.0)  # default to perfect evidence if missing
        eff_likelihood = generic_likelihood * obs
        marginal = eff_likelihood * prior + false_positive * (1.0 - prior)
        posterior = (prior * eff_likelihood) / marginal if marginal > 0 else 0.0
        updated[edge] = posterior
    return updated


# ----------------------------------------------------------------------
# Expected cost computation (deeper integration).
# ----------------------------------------------------------------------
def _mst_cost(
    nodes: Dict[str, Point],
    edges: List[Tuple[str, str]],
    posteriors: Dict[EdgeKey, float],
) -> float:
    """
    Compute the total expected material cost of a minimum‑spanning tree
    where each edge weight is ``length * posterior``.
    """
    weighted_edges = [
        ( _euclidean_length(nodes[a], nodes[b]) * posteriors[frozenset({a, b})], a, b )
        for a, b in edges
    ]
    weighted_edges.sort(key=lambda x: x[0])  # Kruskal needs sorted edges

    uf = _UnionFind(list(nodes.keys()))
    total = 0.0
    for w, a, b in weighted_edges:
        if uf.union(a, b):
            total += w
    return total


def _rooted_path_cost(
    nodes: Dict[str, Point],
    edges: List[Tuple[str, str]],
    posteriors: Dict[EdgeKey, float],
    root: str,
) -> float:
    """
    Compute the sum of posterior‑weighted shortest‑path distances from ``root``
    to every other node using Dijkstra's algorithm.
    """
    import heapq

    # Build adjacency with posterior‑weighted lengths.
    adj: Dict[str, List[Tuple[str, float]]] = {n: [] for n in nodes}
    for a, b in edges:
        w = _euclidean_length(nodes[a], nodes[b]) * posteriors[frozenset({a, b})]
        adj[a].append((b, w))
        adj[b].append((a, w))

    dist: Dict[str, float] = {root: 0.0}
    heap: List[Tuple[float, str]] = [(0.0, root)]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist[cur]:
            continue
        for nxt, w in adj[cur]:
            nd = d + w
            if nxt not in dist or nd < dist[nxt]:
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return sum(dist.values())


def hybrid_tree_expected_cost(
    nodes: Dict[str, Point],
    edges: List[Tuple[str, str]],
    priors: Dict[EdgeKey, float],
    root: str,
    path_weight: float = 0.2,
) -> float:
    """
    Expected total cost for a given ``root`` channel.

    The cost consists of:
        * Expected material cost of the optimal (minimum‑spanning) tree.
        * A path‑weight term proportional to the sum of posterior‑weighted
          distances from the root to all other nodes.
    """
    # Compute posterior probabilities (single‑step Bayes with generic likelihood).
    posteriors = {
        edge: (
            priors[edge] * 0.8
        ) / (priors[edge] * 0.8 + (1 - priors[edge]) * 0.1)
        for edge in priors
    }

    material = _mst_cost(nodes, edges, posteriors)
    path = _rooted_path_cost(nodes, edges, posteriors, root)
    return material + path_weight * path


# ----------------------------------------------------------------------
# Hybrid router.
# ----------------------------------------------------------------------
def hybrid_route_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route ``packet`` to the engine channel with the lowest expected hybrid cost.

    The function:
        1. Extracts ``text`` and ``intent`` (mirroring the original router).
        2. Scores each candidate channel via ``hybrid_tree_expected_cost``.
        3. Invokes ``route_command`` for the best channel.
        4. Updates edge priors using synthetic evidence derived from the
           routing outcome (here we treat a successful routing as evidence
           that edges incident to the chosen channel are reliable).
    """
    text = packet.get("text", "")
    intent = packet.get("intent", "")
    context = packet.get("context", {})

    nodes, edges, priors = _build_graph()

    # Score every channel as the root of the tree.
    scores: Dict[str, float] = {}
    for ch in ENGINE_CHANNELS:
        scores[ch] = hybrid_tree_expected_cost(nodes, edges, priors, root=ch)

    best_channel = min(scores, key=scores.get)

    # Call the underlying routing backend.
    result = route_command(text, intent, {"engine_channel": best_channel, **context})

    # ------------------------------------------------------------------
    # Evidence generation – a lightweight heuristic.
    # ------------------------------------------------------------------
    # Assume that edges directly connected to the selected channel performed
    # well (evidence = 1.0) and all others performed poorly (evidence = 0.3).
    evidence: Dict[EdgeKey, float] = {}
    for a, b in edges:
        edge_key = frozenset({a, b})
        if best_channel in (a, b):
            evidence[edge_key] = 1.0
        else:
            evidence[edge_key] = 0.3

    # Update priors for the next routing decision.
    updated_priors = bayes_update_edge_priors(priors, evidence)

    # Store updated priors back into a module‑level cache for persistence.
    # In a real system this would be written to disk or a DB; here we keep it
    # in memory for simplicity.
    hybrid_route_packet._cached_priors = updated_priors  # type: ignore

    # Assemble the enriched response.
    response = {
        "timestamp": now_z(),
        "selected_channel": best_channel,
        "routing_score": scores[best_channel],
        "all_scores": scores,
        "backend_result": result.to_dict() if hasattr(result, "to_dict") else str(result),
    }
    emit_json(response)
    return response


# Initialise cache attribute.
hybrid_route_packet._cached_priors = {}  # type: ignore


# ----------------------------------------------------------------------
# Optional CLI for manual testing.
# ----------------------------------------------------------------------
def _cli() -> None:
    parser = argparse.ArgumentParser(description="Hybrid ternary router with Bayesian tree cost.")
    parser.add_argument("--text", type=str, default="Hello world")
    parser.add_argument("--intent", type=str, default="greeting")
    parser.add_argument("--context", type=str, default="{}")
    args = parser.parse_args()

    packet = {
        "text": args.text,
        "intent": args.intent,
        "context": json.loads(args.context),
    }
    hybrid_route_packet(packet)


if __name__ == "__main__":
    _cli()