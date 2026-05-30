# DARWIN HAMMER — match 2123, survivor 2
# gen: 5
# parent_a: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m376_s0.py (gen4)
# born: 2026-05-29T23:40:59Z

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import random
import hashlib
from scipy.stats import norm

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str

@dataclass
class Endpoint:
    health_score: float
    failure_rate: float
    recovery_priority: float

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

def gliner_zero_shot_extractor(text: str) -> list[Span]:
    spans = []
    for i in range(len(text)):
        for j in range(i + 1, len(text) + 1):
            span = Span(i, j, text[i:j], "label", 1.0, "backend")
            spans.append(span)
    return spans

def confidence_interval(r: float, delta: float, n: int) -> float:
    z_score = norm.ppf(1 - delta / 2)
    return z_score * math.sqrt(r * (1 - r) / n)

def hybrid_compute_health_scores(spans: list[Span]) -> list[float]:
    health_scores = []
    for span in spans:
        health_score = span.score * (1 - span.score)
        health_scores.append(health_score)
    return health_scores

def hybrid_update_endpoint(endpoints: list[Endpoint], new_request: Span) -> list[Endpoint]:
    updated_endpoints = []
    for endpoint in endpoints:
        failure_rate = endpoint.failure_rate + new_request.score * (1 - new_request.score)
        recovery_priority = endpoint.recovery_priority + new_request.score * new_request.score
        updated_endpoint = Endpoint(endpoint.health_score, failure_rate, recovery_priority)
        updated_endpoints.append(updated_endpoint)
    return updated_endpoints

def hybrid_maybe_switch(endpoints: list[Endpoint], new_request: Span) -> bool:
    health_scores = hybrid_compute_health_scores([new_request])
    for i, endpoint in enumerate(endpoints):
        if health_scores[0] > endpoint.health_score:
            return True
    return False

def hybrid_bandit_algorithm(endpoints: list[Endpoint], spans: list[Span]) -> Span:
    context_vector = np.array([endpoint.health_score for endpoint in endpoints])
    chosen_action = np.argmax(context_vector)
    return spans[chosen_action]

def hybrid_confidence_algorithm(endpoints: list[Endpoint], spans: list[Span], delta: float, n: int) -> Span:
    health_scores = hybrid_compute_health_scores(spans)
    confidence_bound = confidence_interval(1, delta, n)
    chosen_action = np.argmax(health_scores)
    for i, health_score in enumerate(health_scores):
        if health_score > health_scores[chosen_action] - confidence_bound:
            chosen_action = i
    return spans[chosen_action]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Algorithm")
    parser.add_argument("--text", help="text to process")
    parser.add_argument("--file", help="file to read text from")
    args = parser.parse_args()
    text = load_text(args)
    spans = gliner_zero_shot_extractor(text)
    endpoints = [Endpoint(1.0, 0.0, 0.0) for _ in range(len(spans))]
    updated_endpoints = hybrid_update_endpoint(endpoints, spans[0])
    chosen_span = hybrid_bandit_algorithm(updated_endpoints, spans)
    chosen_span_confidence = hybrid_confidence_algorithm(updated_endpoints, spans, 0.1, 10)
    print(f"Chosen span: {chosen_span.text}")
    print(f"Chosen span (Confidence): {chosen_span_confidence.text}")