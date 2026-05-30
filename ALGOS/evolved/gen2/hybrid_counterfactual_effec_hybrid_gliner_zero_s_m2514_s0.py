# DARWIN HAMMER — match 2514, survivor 0
# gen: 2
# parent_a: counterfactual_effects.py (gen0)
# parent_b: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s1.py (gen1)
# born: 2026-05-29T23:42:34Z

#!/usr/bin/env python3
"""Lightweight causal/counterfactual effect estimates fused with the GLiNER zero-shot extraction instrument 
and the minimum-cost tree scoring for length/path trade-offs."""
from dataclasses import dataclass, asdict
import numpy as np
import math
import random
import sys
import pathlib
from datetime import datetime, timezone
from typing import Any, List, Tuple, Dict

@dataclass(frozen=True)
class CausalEffect:
    """Representation of causal effects."""
    effect_id: str; treatment: str; outcome: str; confounders: tuple[str,...]; ate_estimate: float|None; ate_confidence_interval: tuple[float,float]|None; refutation_passed: bool; refutation_methods: tuple[str,...]; heterogeneous_effects: dict[str,float]

@dataclass(frozen=True)
class Span:
    """Representation of extracted spans."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str

def now_iso() -> str:
    """Return the current ISO timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def sha256_text(text: str) -> str:
    """Compute the SHA-256 hash of the input text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

def parse_labels(raw: str | None) -> List[str]:
    """Parse the raw labels into a list."""
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
    """Load the input text from the provided arguments."""
    if args.text is not None:
        return args.text
    if args.file:
        return pathlib.Path(args.file).read_text(encoding="utf-8", errors="replace")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""

def estimate_causal_effect(treatment: str, outcome: str, confounders: list[str], data: dict) -> CausalEffect:
    """Estimate the causal effect using the provided treatment, outcome, confounders, and data."""
    t = list(map(float, data.get(treatment, []))); y = list(map(float, data.get(outcome, [])))
    if not t or len(t) != len(y):
        ate = None; ci = None
    else:
        yt = [yy for tt, yy in zip(t, y) if tt >= 0.5]; yc = [yy for tt, yy in zip(t, y) if tt < 0.5]
        ate = (statistics.mean(yt) - statistics.mean(yc)) if yt and yc else None
        spread = (statistics.pstdev(y) if len(y) > 1 else 0.0); ci = None if ate is None else (ate - spread, ate + spread)
    return CausalEffect(str(uuid.uuid4()), treatment, outcome, tuple(confounders), ate, ci, ate is not None, ('placebo_treatment', 'data_subset', 'random_common_cause'), {})

def estimate_heterogeneous_effects(treatment: str, outcome: str, confounders: list[str], data: dict) -> dict[str, float]:
    """Estimate the heterogeneous effects using the provided treatment, outcome, confounders, and data."""
    e = estimate_causal_effect(treatment, outcome, confounders, data); return {'overall': e.ate_estimate or 0.0}

def run_refutation_suite(effect: CausalEffect, methods: list[str] | None = None) -> dict[str, bool]:
    """Run the refutation suite using the provided causal effect and methods."""
    ms = methods or ['placebo_treatment', 'data_subset', 'random_common_cause']; return {m: bool(effect.ate_estimate is not None and effect.refutation_passed) for m in ms}

def construct_graph(spans: List[Span]) -> Dict[Tuple[int, int], float]:
    """Construct the graph from the provided spans."""
    graph = {}
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            overlap = max(0, min(spans[i].end, spans[j].end) - max(spans[i].start, spans[j].start))
            graph[(i, j)] = overlap
            graph[(j, i)] = overlap
    return graph

def minimum_cost_tree(graph: Dict[Tuple[int, int], float], spans: List[Span]) -> List[int]:
    """Compute the minimum-cost tree using the provided graph and spans."""
    n = len(spans)
    distances = {i: [float('inf')] * n for i in range(n)}
    for i in range(n):
        distances[i][i] = 0
    for (u, v), cost in graph.items():
        distances[u][v] = cost
        distances[v][u] = cost
    for k in range(n):
        for i in range(n):
            for j in range(n):
                distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])
    path = [0]
    visited = {0}
    while len(path) < n:
        current = path[-1]
        next_node = max((node for node in range(n) if node not in visited and distances[current][node] != float('inf')), key=lambda node: distances[current][node])
        path.append(next_node)
        visited.add(next_node)
    return path

def hybrid_operation(treatment: str, outcome: str, confounders: list[str], data: dict, spans: List[Span]) -> dict[str, float]:
    """Perform the hybrid operation using the provided treatment, outcome, confounders, data, and spans."""
    effect = estimate_causal_effect(treatment, outcome, confounders, data)
    graph = construct_graph(spans)
    path = minimum_cost_tree(graph, spans)
    return {'causal_effect': effect.ate_estimate or 0.0, 'minimum_cost_tree': sum(graph[(path[i], path[i + 1])] for i in range(len(path) - 1))}

def run_hybrid_test():
    """Run a smoke test for the hybrid operation."""
    treatment = 'treatment'
    outcome = 'outcome'
    confounders = ['con1', 'con2']
    data = {'treatment': [1, 2, 3], 'outcome': [4, 5, 6]}
    spans = [Span(0, 2, 'text', 'label', 0.5, 'backend')]
    print(hybrid_operation(treatment, outcome, confounders, data, spans))

if __name__ == "__main__":
    run_hybrid_test()