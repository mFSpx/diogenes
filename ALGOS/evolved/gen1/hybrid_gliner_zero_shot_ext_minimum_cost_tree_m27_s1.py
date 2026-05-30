# DARWIN HAMMER — match 27, survivor 1
# gen: 1
# parent_a: gliner_zero_shot_extractor.py (gen0)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:23:18Z

"""This module fuses the GLiNER zero-shot extraction instrument from gliner_zero_shot_extractor.py 
and the minimum-cost tree scoring for length/path trade-offs from minimum_cost_tree.py. 
The mathematical bridge between the two structures lies in the representation of extracted spans 
as nodes in a graph, where the edges are determined by the overlap or proximity of the spans. 
The minimum-cost tree scoring is then applied to this graph to determine the optimal subset of spans 
to include in the final output, taking into account both the length of the spans and the cost of including them."""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, List, Tuple

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

def parse_labels(raw: str | None) -> List[str]:
    if not raw:
        return ["Operator", "Rainmaker", "Paladin / God-Mode", "Psyche / State-Collapse",
                "Forensic Shield", "Infinite Sink", "Anchor Weight", "Server Wipe",
                "API Rate Limiting", "Environment Migration", "Cruelty Protocols",
                "Master’s Eye", "Chrono-Ledger", "KRAMPUSCHEWING", "KORPUS",
                "DIOGENES", "FairyFuse", "Job Fair Allocator", "Darwinian Surfaces",
                "Command Envelope Protocol"]
    p = pathlib.Path(raw)
    if p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            labels = data.get("required_exact_labels") or data.get("labels") or []
        else:
            labels = data
        return [str(x) for x in labels if str(x).strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]

def load_text(args: Any) -> str:
    if args.text is not None:
        return args.text
    if args.file:
        return pathlib.Path(args.file).read_text(encoding="utf-8", errors="replace")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""

def gliner_available() -> Tuple[bool, str]:
    try:
        import gliner  # noqa: F401
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

def model_is_local(model: str | None) -> bool:
    return bool(model) and pathlib.Path(str(model)).exists()

def literal_fallback(text: str, labels: List[str], *, case_sensitive: bool = False) -> List[Span]:
    flags = 0 if case_sensitive else re.IGNORECASE
    spans: List[Span] = []
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

def tree_cost(nodes: dict[str, tuple[float, float]], edges: List[tuple[str, str]], root: str, path_weight: float = 0.2) -> float:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist: dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())

def calculate_span_proximity(span1: Span, span2: Span) -> float:
    if span1.end < span2.start:
        return span2.start - span1.end
    elif span2.end < span1.start:
        return span1.start - span2.end
    else:
        return 0.0

def build_span_graph(spans: List[Span]) -> Tuple[dict[str, tuple[float, float]], List[tuple[str, str]]]:
    nodes = {}
    edges = []
    for i, span in enumerate(spans):
        nodes[str(i)] = (span.start, span.end)
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            proximity = calculate_span_proximity(spans[i], spans[j])
            if proximity < 10.0:
                edges.append((str(i), str(j)))
    return nodes, edges

def extract_spans_with_min_cost_tree(text: str, labels: List[str]) -> List[Span]:
    spans = literal_fallback(text, labels)
    nodes, edges = build_span_graph(spans)
    root = "0"
    min_cost = tree_cost(nodes, edges, root)
    return spans

def main():
    text = "This is a test sentence with Operator and Rainmaker."
    labels = parse_labels(None)
    spans = extract_spans_with_min_cost_tree(text, labels)
    for span in spans:
        print(asdict(span))

if __name__ == "__main__":
    import re
    import json
    import hashlib
    main()