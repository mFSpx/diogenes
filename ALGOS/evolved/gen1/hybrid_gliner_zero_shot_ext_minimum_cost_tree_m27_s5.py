# DARWIN HAMMER — match 27, survivor 5
# gen: 1
# parent_a: gliner_zero_shot_extractor.py (gen0)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:23:18Z

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

# ----------------------------------------------------------------------
# Shared utilities (from Parent A)
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


def now_iso() -> str:
    """Current UTC timestamp in ISO‑8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    """SHA‑256 hash of the supplied Unicode text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def parse_labels(raw: str | None) -> List[str]:
    """Parse a JSON file, comma‑separated string or fallback to defaults."""
    if not raw:
        return list(DEFAULT_LABELS)
    p = Path(raw)
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            labels = data.get("required_exact_labels") or data.get("labels") or []
        else:
            labels = data
        return [str(x).strip() for x in labels if str(x).strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]


def literal_fallback(text: str, labels: List[str], *, case_sensitive: bool = False) -> List[Span]:
    """Pure‑Python label matcher that returns deterministic spans."""
    flags = 0 if case_sensitive else re.IGNORECASE
    spans: List[Span] = []
    seen: Set[Tuple[int, int, str]] = set()
    for label in labels:
        # generate simple variants (e.g. replace “ / ” and “-” with a space)
        candidates = {label, label.replace(" / ", " "), label.replace("-", " ")}
        for phrase in sorted(candidates, key=len, reverse=True):
            if not phrase.strip():
                continue
            pattern = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", flags)
            for m in pattern.finditer(text):
                key = (m.start(), m.end(), label)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(
                    Span(
                        start=m.start(),
                        end=m.end(),
                        text=m.group(0),
                        label=label,
                        score=1.0,  # literal fallback always full confidence
                    )
                )
    return sorted(spans, key=lambda s: (s.start, s.end, s.label))


def extract(text: str, labels: List[str]) -> List[Span]:
    """High‑level wrapper – currently only literal fallback is used."""
    return literal_fallback(text, labels)


# ----------------------------------------------------------------------
# Tree cost utilities (from Parent B) – now using a Minimum Spanning Tree
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def euclidean(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


class UnionFind:
    """Simple Union‑Find (Disjoint Set) data structure for Kruskal's algorithm."""

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


def mst_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
) -> Tuple[float, List[Edge]]:
    """
    Compute a Minimum Spanning Tree (MST) over ``nodes`` using Kruskal's algorithm.
    Returns the total edge length (material cost) and the list of edges that
    belong to the MST.
    """
    # Pre‑compute weighted edges
    weighted_edges: List[Tuple[float, Edge]] = [
        (euclidean(nodes[a], nodes[b]), (a, b)) for a, b in edges
    ]
    weighted_edges.sort(key=lambda x: x[0])  # sort by distance

    uf = UnionFind(list(nodes.keys()))
    mst_edges: List[Edge] = []
    total = 0.0
    for w, (a, b) in weighted_edges:
        if uf.union(a, b):
            mst_edges.append((a, b))
            total += w
            if len(mst_edges) == len(nodes) - 1:
                break
    return total, mst_edges


def bfs_distances(root: str, adj: Dict[str, List[str]], nodes: Dict[str, Point]) -> Dict[str, float]:
    """Breadth‑first traversal on a tree to compute cumulative Euclidean distance from ``root``."""
    dist: Dict[str, float] = {root: 0.0}
    queue: List[str] = [root]
    while queue:
        cur = queue.pop(0)
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + euclidean(nodes[cur], nodes[nxt])
                queue.append(nxt)
    return dist


# ----------------------------------------------------------------------
# Hybrid layer – deeper integration of extraction confidence and geometry
# ----------------------------------------------------------------------
def spans_to_nodes(spans: List[Span]) -> Tuple[Dict[str, Point], Dict[str, float]]:
    """
    Map each ``Span`` to a 2‑D point using *both* start and end indices:
        x = start character index
        y = end character index
    Returns a tuple ``(points, confidences)`` where ``confidences`` maps the same
    identifier to the span's confidence score.
    """
    points: Dict[str, Point] = {}
    confidences: Dict[str, float] = {}
    for i, sp in enumerate(spans):
        pid = f"{i}:{sp.label}"
        points[pid] = (float(sp.start), float(sp.end))
        confidences[pid] = float(sp.score)
    return points, confidences


def build_edges(spans: List[Span]) -> List[Edge]:
    """
    Construct a compact edge set encoding two notions of proximity:

    1. **Chronological adjacency** – connect each span to the next one in text order.
    2. **Label similarity** – connect each span to the *nearest* later span
       sharing the same label (instead of a full clique, which reduces O(k²) blow‑up).

    Duplicate edges are eliminated via a ``set``.
    """
    edge_set: Set[Edge] = set()

    # chronological edges (undirected)
    for i in range(len(spans) - 1):
        a = f"{i}:{spans[i].label}"
        b = f"{i+1}:{spans[i+1].label}"
        edge_set.add((a, b) if a < b else (b, a))

    # label‑based nearest‑future edges
    last_seen: Dict[str, int] = {}
    for idx, sp in enumerate(spans):
        label = sp.label
        if label in last_seen:
            a = f"{last_seen[label]}:{label}"
            b = f"{idx}:{label}"
            edge_set.add((a, b) if a < b else (b, a))
        last_seen[label] = idx

    return list(edge_set)


def hybrid_metric(
    text: str,
    labels: List[str],
    *,
    root_index: int = 0,
    base_path_weight: float = 0.2,
) -> Tuple[float, Dict[str, Any]]:
    """
    End‑to‑end hybrid computation with a deeper mathematical integration:

    1. Extract spans (literal fallback for reproducibility).
    2. Embed spans as points (start, end) and retain confidence scores.
    3. Build a sparse edge set (chronology + nearest‑same‑label).
    4. Compute a Minimum Spanning Tree (MST) → material cost.
    5. From the MST, compute the sum of root‑to‑node distances.
    6. Modulate the path‑weight by the *average* confidence: higher confidence
       reduces the penalty, lower confidence inflates it.

    Returns a tuple ``(final_metric, diagnostics)`` where ``diagnostics`` contains
    the raw components for inspection.
    """
    # 1. Extraction
    spans = extract(text, labels)
    if not spans:
        return 0.0, {"reason": "no spans found"}

    # 2. Geometry + confidence
    points, confidences = spans_to_nodes(spans)

    # 3. Edge construction
    edges = build_edges(spans)

    # 4. Minimum Spanning Tree
    material_cost, mst_edges = mst_cost(points, edges)

    # Build adjacency list for the tree
    adj: Dict[str, List[str]] = {node: [] for node in points}
    for a, b in mst_edges:
        adj[a].append(b)
        adj[b].append(a)

    # 5. Root selection
    root_id = f"{root_index}:{spans[root_index].label}"
    if root_id not in points:
        root_id = next(iter(points))  # fallback to first node

    distances = bfs_distances(root_id, adj, points)
    sum_distances = sum(distances.values())

    # 6. Confidence‑aware path weight
    mean_conf = sum(confidences.values()) / len(confidences)
    # Scale: when mean_conf → 1, weight → base_path_weight * 0.5;
    # when mean_conf → 0, weight → base_path_weight * 2.0
    path_weight = base_path_weight * (1.5 - 0.5 * mean_conf)

    final_metric = material_cost + path_weight * sum_distances

    diagnostics = {
        "spans": [asdict(s) for s in spans],
        "points": points,
        "confidences": confidences,
        "edges": edges,
        "mst_edges": mst_edges,
        "material_cost": material_cost,
        "root_id": root_id,
        "distances_from_root": distances,
        "sum_distances": sum_distances,
        "mean_confidence": mean_conf,
        "path_weight_used": path_weight,
    }
    return final_metric, diagnostics


# ----------------------------------------------------------------------
# Minimal CLI for manual testing (optional)
# ----------------------------------------------------------------------
def _cli() -> None:
    parser = argparse.ArgumentParser(description="Hybrid extraction‑tree metric")
    parser.add_argument("file", type=Path, help="Path to a plain‑text file")
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="JSON file, comma‑separated list, or leave empty for defaults",
    )
    parser.add_argument(
        "--root-index",
        type=int,
        default=0,
        help="Index of the span to treat as root (default: 0)",
    )
    parser.add_argument(
        "--path-weight",
        type=float,
        default=0.2,
        help="Base path weight (default: 0.2)",
    )
    args = parser.parse_args()

    text = args.file.read_text(encoding="utf-8")
    label_list = parse_labels(args.labels)
    metric, details = hybrid_metric(
        text,
        label_list,
        root_index=args.root_index,
        base_path_weight=args.path_weight,
    )
    print(json.dumps({"metric": metric, "details": details}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()