# DARWIN HAMMER — match 5362, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_decreasing_pr_m1229_s2.py (gen5)
# born: 2026-05-30T00:02:54Z

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import re

EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicki)\b",
    re.I,
)

def extract_text_features(text: str) -> Dict[str, int]:
    return {
        "evidence": len(EVIDENCE_RE.findall(text)),
        "planning": len(PLANNING_RE.findall(text)),
        "delay": len(DELAY_RE.findall(text)),
        "support": len(SUPPORT_RE.findall(text)),
        "boundary": len(BOUNDARY_RE.findall(text)),
        "outcome": len(OUTCOME_RE.findall(text)),
        "impulsive": len(IMPULSIVE_RE.findall(text)),
    }

_LANCZOS_G = 7
_LANCZOS_C = np.array(
    [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857,
    ]
)

def _gamma(z: float) -> float:
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * _gamma(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def caputo_kernel(alpha: float, t: np.ndarray) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("Fractional order alpha must be positive.")
    t = np.where(t == 0, 1e-12, t)
    return t ** (alpha - 1) / _gamma(alpha)

def euclidean_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def belief_factor(feat_i: Dict[str, int], feat_j: Dict[str, int]) -> float:
    pos_i = feat_i["evidence"] + feat_i["planning"] + feat_i["outcome"]
    neg_i = feat_i["delay"] + feat_i["impulsive"] + 1e-9
    pos_j = feat_j["evidence"] + feat_j["planning"] + feat_j["outcome"]
    neg_j = feat_j["delay"] + feat_j["impulsive"] + 1e-9
    delta = (pos_i / neg_i + pos_j / neg_j) / 2.0
    temperature = 0.5
    return 1.0 / (1.0 + math.exp(-temperature * (delta - 1.0)))

def hybrid_distance(
    a: Tuple[float, ...],
    b: Tuple[float, ...],
    alpha: float,
    feat_a: Dict[str, int],
    feat_b: Dict[str, int],
) -> float:
    raw = euclidean_distance(a, b)
    kernel_val = caputo_kernel(alpha, np.array([raw]))[0]
    belief = belief_factor(feat_a, feat_b)
    return kernel_val * belief

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
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

def minimum_cost_spanning_tree(
    positions: List[Tuple[float, ...]],
    texts: List[str],
    alpha: float = 0.8,
) -> Tuple[float, List[Tuple[int, int]]]:
    n = len(positions)
    uf = _UnionFind(n)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            feat_i = extract_text_features(texts[i])
            feat_j = extract_text_features(texts[j])
            dist = hybrid_distance(positions[i], positions[j], alpha, feat_i, feat_j)
            edges.append((dist, i, j))
    edges.sort()
    mst = []
    total_cost = 0.0
    for dist, i, j in edges:
        if uf.union(i, j):
            mst.append((i, j))
            total_cost += dist
    return total_cost, mst