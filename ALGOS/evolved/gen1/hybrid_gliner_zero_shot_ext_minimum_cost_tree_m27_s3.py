# DARWIN HAMMER — match 27, survivor 3
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
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

def parse_labels(raw: str | None) -> List[str]:
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
    flags = 0 if case_sensitive else re.IGNORECASE
    spans: List[Span] = []
    seen: set[Tuple[int, int, str]] = set()
    for label in labels:
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
                        score=1.0,
                    )
                )
    return sorted(spans, key=lambda s: (s.start, s.end, s.label))

def extract(text: str, labels: List[str]) -> List[Span]:
    return literal_fallback(text, labels)

Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    path_weight: float = 0.2,
) -> float:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + length(nodes[cur], nodes[nxt])
                stack.append(nxt)

    return material + path_weight * sum(dist.values())

def spans_to_nodes(spans: List[Span]) -> Dict[str, Point]:
    nodes: Dict[str, Point] = {}
    for i, sp in enumerate(spans):
        pid = f"{i}:{sp.label}"
        nodes[pid] = (float(sp.start), float(sp.end - sp.start))
    return nodes

def build_edges(spans: List[Span]) -> List[Edge]:
    edges: List[Edge] = []
    for i in range(len(spans) - 1):
        a = f"{i}:{spans[i].label}"
        b = f"{i+1}:{spans[i+1].label}"
        edges.append((a, b))

    label_to_indices: Dict[str, List[int]] = {}
    for idx, sp in enumerate(spans):
        label_to_indices.setdefault(sp.label, []).append(idx)
    for indices in label_to_indices.values():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a = f"{indices[i]}:{spans[indices[i]].label}"
                b = f"{indices[j]}:{spans[indices[j]].label}"
                edges.append((a, b))
    return edges

def confidence_modulated_path_weight(spans: List[Span], base_path_weight: float) -> float:
    scores = np.array([s.score for s in spans])
    mean_score = np.mean(scores)
    modulated_path_weight = base_path_weight * (1 - mean_score)
    return modulated_path_weight

def hybrid_metric(
    text: str,
    labels: List[str],
    *,
    root_index: int = 0,
    base_path_weight: float = 0.2,
) -> Tuple[float, Dict[str, Any]]:
    spans = extract(text, labels)
    nodes = spans_to_nodes(spans)
    edges = build_edges(spans)
    root = f"{root_index}:{spans[root_index].label}"
    modulated_path_weight = confidence_modulated_path_weight(spans, base_path_weight)
    cost = tree_cost(nodes, edges, root, modulated_path_weight)
    return cost, {"spans": [asdict(s) for s in spans], "cost": cost}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--root_index", type=int, default=0)
    parser.add_argument("--base_path_weight", type=float, default=0.2)
    args = parser.parse_args()
    labels = parse_labels(args.labels)
    cost, info = hybrid_metric(args.text, labels, root_index=args.root_index, base_path_weight=args.base_path_weight)
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()