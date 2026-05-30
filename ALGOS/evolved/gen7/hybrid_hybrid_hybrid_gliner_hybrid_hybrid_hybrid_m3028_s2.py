# DARWIN HAMMER — match 3028, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_gliner_zero_s_hybrid_doomsday_cale_m49_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m1412_s2.py (gen6)
# born: 2026-05-29T23:47:19Z

"""Hybrid algorithm merging:
- hybrid_hybrid_gliner_zero_s_hybrid_doomsday_cale_m49_s1.py (geometric span embedding + Gini coefficient)
- hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m1412_s2.py (Gaussian beam, Fisher score, tree cost, multivector algebra)

Mathematical bridge:
Both parents operate on 2‑D geometric objects.
Parent A treats each extracted span as a point **p = (start, length)** and
evaluates the inequality of the length distribution via the Gini coefficient.
Parent B builds a weighted graph on 2‑D points, computes edge lengths,
evaluates a Fisher information‑like score (derivative of a Gaussian beam) and
adds a tree‑cost term that mixes total edge material with summed root‑to‑node
distances.

The hybrid therefore:
1. Maps every `Span` to a point (start, length).
2. Constructs a minimum‑spanning‑tree (MST) on those points using Euclidean
   distances (the `length` routine from Parent B).
3. Computes the tree cost (`tree_cost` from Parent B) as a spatial coherence
   metric.
4. Computes the Gini coefficient of the span lengths (Parent A).
5. Computes an aggregated Fisher information term over the span lengths
   using the Gaussian‑beam / Fisher‑score pair (Parent B).

A final hybrid score is a weighted combination of the three
quantities, rewarding compact, well‑connected span layouts (low tree cost),
diverse length distribution (high Gini) and informative length
concentration (high Fisher)."""

import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from Parent A)
# ----------------------------------------------------------------------
DEFAULT_LABELS = [
    "Operator", "Rainmaker", "Paladin / God-Mode", "Psyche / State-Collapse",
    "Forensic Shield", "Infinite Sink", "Anchor Weight", "Server Wipe",
    "API Rate Limiting", "Environment Migration", "Cruelty Protocols",
    "Master’s Eye", "Chrono-Ledger", "KRAMPUSCHEWING", "KORPUS",
    "DIOGENES", "FairyFuse", "Job Fair Allocator", "Darwinian Surfaces",
    "Command Envelope Protocol",
]

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"

    @property
    def length(self) -> int:
        return self.end - self.start

# ----------------------------------------------------------------------
# Utility functions (Parent A)
# ----------------------------------------------------------------------
def now_iso() -> str:
    """Current UTC timestamp in ISO‑8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    """SHA‑256 hash of the supplied Unicode text."""
    import hashlib
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def parse_labels(raw: str | None) -> List[str]:
    if raw is None:
        return DEFAULT_LABELS
    return [label.strip() for label in raw.split(",")]


def gini_coefficient(values: List[float]) -> float:
    """Gini coefficient of a non‑empty list of numbers."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(values)
    cumulative = 0.0
    for i, v in enumerate(sorted_vals, 1):
        cumulative += i * v
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return gini


# ----------------------------------------------------------------------
# Functions from Parent B (geometry & Fisher information)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]   # identifiers of two nodes


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def prim_mst(nodes: Dict[str, Point]) -> List[Edge]:
    """Return a list of edges forming a Minimum Spanning Tree (Prim's algorithm)."""
    if not nodes:
        return []
    visited = set()
    edges: List[Edge] = []
    # start from an arbitrary node
    start = next(iter(nodes))
    visited.add(start)
    # candidate edges (weight, a, b)
    candidates: List[Tuple[float, str, str]] = []
    for b in nodes:
        if b != start:
            candidates.append((length(nodes[start], nodes[b]), start, b))
    import heapq
    heapq.heapify(candidates)

    while len(visited) < len(nodes):
        w, a, b = heapq.heappop(candidates)
        if b in visited:
            continue
        visited.add(b)
        edges.append((a, b))
        for c in nodes:
            if c not in visited:
                heapq.heappush(candidates, (length(nodes[b], nodes[c]), b, c))
    return edges


def tree_cost(nodes: Dict[str, Point], edges: List[Edge], root: str, path_weight: float = 0.2) -> float:
    """Cost = total edge material + path_weight * sum(dist(root, node))."""
    # material = sum of edge lengths
    material = sum(length(nodes[a], nodes[b]) for a, b in edges)

    # build adjacency list
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + length(nodes[cur], nodes[nb])
                stack.append(nb)

    return material + path_weight * sum(dist.values())


# ----------------------------------------------------------------------
# Hybrid core: bridging the two parent topologies
# ----------------------------------------------------------------------
def span_to_point(span: Span) -> Point:
    """Map a Span to a 2‑D point (start, length)."""
    return (float(span.start), float(span.length))


def build_span_graph(spans: List[Span]) -> Tuple[Dict[str, Point], List[Edge]]:
    """
    Construct a fully connected graph on span points, then extract its MST.
    Returns the node dictionary and the MST edge list.
    """
    nodes = {f"n{i}": span_to_point(s) for i, s in enumerate(spans)}
    mst_edges = prim_mst(nodes)
    return nodes, mst_edges


def fisher_aggregate(spans: List[Span], center: float, width: float) -> float:
    """
    Compute the sum of Fisher scores over the span lengths.
    The `center` and `width` parameters control the Gaussian beam.
    """
    return sum(fisher_score(s.length, center, width) for s in spans)


def hybrid_metric(
    spans: List[Span],
    root_index: int = 0,
    gini_weight: float = 0.4,
    tree_weight: float = 0.3,
    fisher_weight: float = 0.3,
    fisher_center: float | None = None,
    fisher_width: float = 10.0,
) -> float:
    """
    Unified score that blends:
    * Gini coefficient of span lengths (higher = more diverse)
    * Tree cost of the MST built on span points (lower = more compact)
    * Aggregated Fisher information on lengths (higher = more informative)

    All components are normalised to [0, 1] before weighting.
    """
    if not spans:
        raise ValueError("span list must not be empty")

    # 1. Gini (already in [0,1] for non‑negative data)
    lengths = [float(s.length) for s in spans]
    gini = gini_coefficient(lengths)

    # 2. Tree cost (we invert because lower cost is better)
    nodes, edges = build_span_graph(spans)
    root_id = f"n{root_index % len(spans)}"
    raw_tree = tree_cost(nodes, edges, root=root_id)
    # Normalise by a heuristic upper bound (max possible material + path)
    max_material = sum(length(nodes[a], nodes[b]) for a in nodes for b in nodes)  # over‑estimate
    max_path = max_material  # same scale
    max_possible = max_material + 0.2 * max_path
    tree_score = 1.0 - min(raw_tree / max_possible, 1.0)  # higher is better

    # 3. Fisher aggregate (normalise by max possible if we assume width>0)
    if fisher_center is None:
        fisher_center = np.mean(lengths) if lengths else 0.0
    raw_fisher = fisher_aggregate(spans, fisher_center, fisher_width)
    # Rough normalisation: divide by number of spans * max per‑span Fisher (when theta=center)
    max_per_span = fisher_score(fisher_center, fisher_center, fisher_width)
    fisher_score_norm = raw_fisher / (len(spans) * max_per_span) if max_per_span > 0 else 0.0
    fisher_score_norm = min(max(fisher_score_norm, 0.0), 1.0)

    # Weighted combination
    hybrid = (
        gini_weight * gini
        + tree_weight * tree_score
        + fisher_weight * fisher_score_norm
    )
    return hybrid


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def demo_gini() -> None:
    """Simple demo of the Gini coefficient on synthetic spans."""
    spans = [
        Span(start=0, end=5, text="a", label="X", score=0.9),
        Span(start=10, end=20, text="b", label="Y", score=0.8),
        Span(start=30, end=31, text="c", label="Z", score=0.7),
    ]
    g = gini_coefficient([s.length for s in spans])
    print(f"Gini coefficient of lengths: {g:.4f}")


def demo_tree_cost() -> None:
    """Build an MST from spans and report its cost."""
    spans = [
        Span(start=0, end=4, text="a", label="X", score=0.9),
        Span(start=5, end=15, text="b", label="Y", score=0.8),
        Span(start=20, end=25, text="c", label="Z", score=0.7),
        Span(start=30, end=45, text="d", label="W", score=0.6),
    ]
    nodes, edges = build_span_graph(spans)
    cost = tree_cost(nodes, edges, root="n0")
    print(f"Tree cost (material + weighted path): {cost:.4f}")


def demo_hybrid_metric() -> None:
    """Run the full hybrid metric on a random collection of spans."""
    random.seed(42)
    spans = []
    for i in range(10):
        start = random.randint(0, 100)
        length = random.randint(1, 20)
        spans.append(
            Span(
                start=start,
                end=start + length,
                text=f"token{i}",
                label=random.choice(DEFAULT_LABELS),
                score=random.random(),
            )
        )
    score = hybrid_metric(spans)
    print(f"Hybrid unified score: {score:.4f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: Gini ===")
    demo_gini()
    print("\n=== Demo: Tree Cost ===")
    demo_tree_cost()
    print("\n=== Demo: Hybrid Metric ===")
    demo_hybrid_metric()