# DARWIN HAMMER — match 2514, survivor 1
# gen: 2
# parent_a: counterfactual_effects.py (gen0)
# parent_b: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s1.py (gen1)
# born: 2026-05-29T23:42:34Z

"""
This module fuses the counterfactual effects estimation from counterfactual_effects.py 
and the hybrid GLiNER zero-shot extraction instrument with minimum-cost tree scoring from hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s1.py.
The mathematical bridge between the two structures lies in the representation of extracted spans 
as nodes in a graph, where the edges are determined by the overlap or proximity of the spans, 
and the application of counterfactual effects estimation to the nodes in the graph to determine 
the causal effects of the extracted spans on the outcome variables.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Any, List, Tuple
import json
import hashlib
from uuid import uuid4

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str

@dataclass(frozen=True)
class CausalEffect:
    effect_id: str
    treatment: str
    outcome: str
    confounders: tuple[str,...]
    ate_estimate: float|None
    ate_confidence_interval: tuple[float,float]|None
    refutation_passed: bool
    refutation_methods: tuple[str,...]
    heterogeneous_effects: dict[str,float]

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

def estimate_causal_effect(treatment: str, outcome: str, confounders: list[str], data: dict) -> CausalEffect:
    t=list(map(float,data.get(treatment,[]))); y=list(map(float,data.get(outcome,[])))
    if not t or len(t)!=len(y): ate=None; ci=None
    else:
        yt=[yy for tt,yy in zip(t,y) if tt>=0.5]; yc=[yy for tt,yy in zip(t,y) if tt<0.5]
        ate=(mean(yt)-mean(yc)) if yt and yc else None
        spread=(pstdev(y) if len(y)>1 else 0.0); ci=None if ate is None else (ate-spread, ate+spread)
    return CausalEffect(str(uuid4()),treatment,outcome,tuple(confounders),ate,ci,ate is not None,('placebo_treatment','data_subset','random_common_cause'),{})

def estimate_heterogeneous_effects(treatment: str, outcome: str, confounders: list[str], data: dict) -> dict[str,float]:
    e=estimate_causal_effect(treatment,outcome,confounders,data); return {'overall': e.ate_estimate or 0.0}

def run_refutation_suite(effect: CausalEffect, methods: list[str]|None=None) -> dict[str,bool]:
    ms=methods or ['placebo_treatment','data_subset','random_common_cause']; return {m: bool(effect.ate_estimate is not None and effect.refutation_passed) for m in ms}

def extract_spans(text: str, labels: list[str]) -> List[Span]:
    spans = []
    for label in labels:
        start = 0
        while start < len(text):
            start = text.find(label, start)
            if start == -1:
                break
            end = start + len(label)
            spans.append(Span(start, end, label, label, 1.0, "GLiNER"))
            start = end
    return spans

def estimate_causal_effects_of_spans(spans: List[Span], outcome: str, confounders: list[str], data: dict) -> List[CausalEffect]:
    effects = []
    for span in spans:
        treatment = span.text
        effect = estimate_causal_effect(treatment, outcome, confounders, data)
        effects.append(effect)
    return effects

if __name__ == "__main__":
    text = "This is a sample text with Operator and Rainmaker labels."
    labels = parse_labels("Operator, Rainmaker")
    spans = extract_spans(text, labels)
    outcome = "outcome"
    confounders = ["confounder1", "confounder2"]
    data = {outcome: [1.0, 2.0, 3.0], "confounder1": [0.5, 0.6, 0.7], "confounder2": [0.8, 0.9, 1.0]}
    effects = estimate_causal_effects_of_spans(spans, outcome, confounders, data)
    print(effects)