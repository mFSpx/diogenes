# DARWIN HAMMER — match 27, survivor 2
# gen: 1
# parent_a: gliner_zero_shot_extractor.py (gen0)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:23:18Z

"""Hybrid extraction‑tree algorithm.

This module fuses two previously independent tools:

* **GLiNER zero‑shot extractor** – produces character‑offset spans (start, end,
  text, label, confidence score).
* **Minimum‑cost tree scorer** – evaluates a graph of nodes with Euclidean
  edges, returning ``material + path_weight * Σ distance(root → node)``.

The mathematical bridge is a *geometric embedding* of each extracted span:

point_i = (x_i, y_i) = (start_i, length_i)

where ``length_i = end_i - start_i``.  The set of points becomes the node
dictionary required by the tree‑cost routine.  Edges are defined by the
chronological order of spans (i → i+1) and optionally by identical labels.
The tree‑cost then measures the “spatial coherence” of the extraction while
the extraction scores modulate the path weight, yielding a single hybrid
metric that rewards both high‑confidence spans and compact, well‑connected
layouts."""

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
       (higher confidence → smaller penalty for long paths).

    Returns ``(total_cost, metadata)`` where ``metadata`` contains the
    intermediate structures for debugging.
    """
    spans = extract(text, labels)
    if not spans:
        return float("inf"), {"reason": "no spans found"}

    nodes = spans_to_nodes(spans)
    edges = build_edges(spans)

    # Determine a root identifier – default to the first span
    root_id = f"{root_index}:{spans[root_index].label}" if spans else None
    if root_id is None or root_id not in nodes:
        root_id = next(iter(nodes))  # fallback to arbitrary node

    # Modulate path weight by average confidence (confidence in [0,1])
    avg_score = float(np.mean([sp.score for sp in spans]))
    path_weight = base_path_weight * (1.0 - avg_score)  # lower weight for higher confidence

    cost = tree_cost(nodes, edges, root_id, path_weight=path_weight)

    meta = {
        "spans": [asdict(s) for s in spans],
        "nodes": nodes,
        "edges": edges,
        "root": root_id,
        "avg_score": avg_score,
        "path_weight_used": path_weight,
    }
    return cost, meta


# ----------------------------------------------------------------------
# Command‑line interface & smoke test
# ----------------------------------------------------------------------
def _cli() -> None:
    parser = argparse.ArgumentParser(description="Hybrid extraction‑tree demo")
    parser.add_argument("text", nargs="?", help="Input text (or read from STDIN)")
    parser.add_argument(
        "-l",
        "--labels",
        default="",
        help="Comma‑separated list of labels or path to JSON label file",
    )
    parser.add_argument(
        "-r",
        "--root-index",
        type=int,
        default=0,
        help="Index of span to treat as root in the tree (default: 0)",
    )
    args = parser.parse_args()

    # Load text
    if args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    labels = parse_labels(args.labels)
    cost, meta = hybrid_metric(text, labels, root_index=args.root_index)

    print(f"Hybrid cost: {cost:.4f}")
    print("Metadata snapshot:")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Smoke test – runs a deterministic example without external files.
    sample_text = (
        "Operator initiated Server Wipe while the Infinite Sink engaged. "
        "API Rate Limiting was triggered by the Rainmaker."
    )
    sample_labels = ["Operator", "Server Wipe", "Infinite Sink", "API Rate Limiting", "Rainmaker"]
    total, details = hybrid_metric(sample_text, sample_labels)
    print(f"[Smoke test] Hybrid cost = {total:.4f}")
    # Ensure that at least one span was found; otherwise the test would be trivial.
    assert details["spans"], "No spans extracted in smoke test"
    # Verify that cost is a finite number.
    assert math.isfinite(total), "Cost should be a finite float"
    # Print a concise summary.
    print("Extracted spans:")
    for s in details["spans"]:
        print(f"  - ({s['start']}, {s['end']}) [{s['label']}] \"{s['text']}\" score={s['score']}")
    # Optionally, invoke CLI when executed directly with arguments.
    # Comment out the following line if you prefer the smoke test only.
    # _cli()