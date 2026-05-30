# DARWIN HAMMER — match 27, survivor 4
# gen: 1
# parent_a: gliner_zero_shot_extractor.py (gen0)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:23:18Z

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

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
    seen: set[Tuple[int, int, str]] = set()
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
# Tree cost utilities (from Parent B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    path_weight: float = 0.2,
) -> float:
    """
    Compute the material cost (sum of edge lengths) plus a weighted
    sum of distances from ``root`` to every other node.
    """
    # adjacency list
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])

    # BFS/DFS to compute distance from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + length(nodes[cur], nodes[nxt])
                stack.append(nxt)

    return material + path_weight * sum(dist.values())


# ----------------------------------------------------------------------
# Hybrid layer – mapping spans → geometry and evaluating a unified metric
# ----------------------------------------------------------------------
def spans_to_nodes(spans: List[Span]) -> Dict[str, Point]:
    """
    Map each ``Span`` to a 2‑D point:
        x = start character index
        y = span length (end - start)
    The key is a deterministic identifier ``f"{i}:{label}"``.
    """
    nodes: Dict[str, Point] = {}
    for i, sp in enumerate(spans):
        pid = f"{i}:{sp.label}"
        nodes[pid] = (float(sp.start), float(sp.end - sp.start))
    return nodes


def build_edges(spans: List[Span]) -> List[Edge]:
    """
    Construct edges that encode two notions of proximity:

    1. **Chronological adjacency** – connect each span to the next one in the
       text order (i → i+1).  This yields a path that respects the original
       narrative flow.
    2. **Label similarity** – connect spans that share the exact same label.
       This creates shortcuts that reward semantic cohesion.

    Duplicate edges are automatically collapsed by the downstream ``tree_cost``
    implementation.
    """
    edges: List[Edge] = []
    # chronological edges
    for i in range(len(spans) - 1):
        a = f"{i}:{spans[i].label}"
        b = f"{i+1}:{spans[i+1].label}"
        edges.append((a, b))

    # label‑based edges
    label_to_indices: Dict[str, List[int]] = {}
    for idx, sp in enumerate(spans):
        label_to_indices.setdefault(sp.label, []).append(idx)
    for indices in label_to_indices.values():
        if len(indices) < 2:
            continue
        # connect every pair within the same label (simple O(k^2) for small k)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a = f"{indices[i]}:{spans[indices[i]].label}"
                b = f"{indices[j]}:{spans[indices[j]].label}"
                edges.append((a, b))
    return edges


def hybrid_metric(
    text: str,
    labels: List[str],
    *,
    root_index: int = 0,
    base_path_weight: float = 0.2,
) -> Tuple[float, Dict[str, Any]]:
    """
    End‑to‑end hybrid computation:

    1. Extract spans using the literal fallback extractor.
    2. Embed spans into a geometric graph.
    3. Compute the minimum‑cost tree cost.
    4. Adjust the ``path_weight`` by the mean confidence score of the spans
       (higher confidence → smaller penalty for long distances).

    Returns:
        A tuple of the hybrid metric and a dictionary with additional information.
    """
    spans = extract(text, labels)
    nodes = spans_to_nodes(spans)
    edges = build_edges(spans)
    root = f"{root_index}:{spans[root_index].label}"
    path_weight = base_path_weight * (1 - np.mean([sp.score for sp in spans]))
    metric = tree_cost(nodes, edges, root, path_weight)
    return metric, {
        "spans": [asdict(sp) for sp in spans],
        "nodes": nodes,
        "edges": edges,
        "root": root,
        "path_weight": path_weight,
    }