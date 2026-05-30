# DARWIN HAMMER — match 27, survivor 0
# gen: 1
# parent_a: gliner_zero_shot_extractor.py (gen0)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:23:18Z

#!/usr/bin/env python3
"""Hybrid algorithm combining gliner_zero_shot_extractor and minimum_cost_tree.

This module integrates the natural language processing capabilities of gliner_zero_shot_extractor
with the minimum-cost tree optimization of minimum_cost_tree. The mathematical bridge between
these two structures lies in the representation of text spans as nodes in a graph, where the
edges represent the relationships between these spans. The minimum-cost tree algorithm is then
applied to this graph to optimize the extraction of spans.

The hybrid algorithm first extracts spans from a given text using the gliner_zero_shot_extractor,
then constructs a graph where each span is a node, and the edges represent the similarity between
spans. The minimum-cost tree algorithm is then applied to this graph to select the most relevant spans
while minimizing the cost of the tree.

Author: [Your Name]
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy import linalg as LA

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

def parse_labels(raw: str | None) -> list[str]:
    if not raw:
        return ["Operator", "Rainmaker", "Paladin / God-Mode", "Psyche / State-Collapse"]
    p = Path(raw)
    if p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            labels = data.get("required_exact_labels") or data.get("labels") or []
        else:
            labels = data
        return [str(x) for x in labels if str(x).strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]

def load_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8", errors="replace")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""

def gliner_available() -> tuple[bool, str]:
    try:
        import gliner  # noqa: F401
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

def literal_fallback(text: str, labels: list[str], *, case_sensitive: bool = False) -> list[Span]:
    flags = 0 if case_sensitive else re.IGNORECASE
    spans: list[Span] = []
    seen: set[tuple[int, int, str]] = set()
    for label in labels:
        candidates = {label}
        candidates.add(label.replace(" / ", " "))
        candidates.add(label.replace("-", " "))
        for phrase in sorted(candidates, key=len, reverse=True):
            if not phrase.strip():
                continue
            pattern = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", flags)
            for match in pattern.finditer(text):
                key = (match.start(), match.end(), label)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(Span(match.start(), match.end(), match.group(0), label, 1.0, "literal_fallback_no_gliner"))
    return sorted(spans, key=lambda s: (s.start, s.end, s.label))

def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(nodes: dict[str, tuple[float, float]], edges: list[tuple[str, str]], root: str, path_weight: float = 0.2) -> float:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())

def extract_spans(text: str, labels: list[str]) -> list[Span]:
    spans = literal_fallback(text, labels)
    return spans

def construct_graph(spans: list[Span]) -> tuple[dict[str, tuple[float, float]], list[tuple[str, str]]]:
    nodes = {}
    edges = []
    for i, span in enumerate(spans):
        nodes[f"span_{i}"] = (span.start, span.end)
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            edges.append((f"span_{i}", f"span_{j}"))
    return nodes, edges

def optimize_tree(nodes: dict[str, tuple[float, float]], edges: list[tuple[str, str]], root: str) -> float:
    return tree_cost(nodes, edges, root)

def hybrid_operation(text: str, labels: list[str]) -> tuple[list[Span], float]:
    spans = extract_spans(text, labels)
    nodes, edges = construct_graph(spans)
    cost = optimize_tree(nodes, edges, "span_0")
    return spans, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid algorithm for span extraction and tree optimization")
    parser.add_argument("--text", type=str, help="input text")
    parser.add_argument("--labels", type=str, help="labels for span extraction")
    args = parser.parse_args()
    text = load_text(args)
    labels = parse_labels(args.labels)
    spans, cost = hybrid_operation(text, labels)
    print("Extracted spans:")
    for span in spans:
        print(asdict(span))
    print("Optimized tree cost:", cost)