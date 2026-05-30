# DARWIN HAMMER — match 3730, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m1117_s2.py (gen4)
# parent_b: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s1.py (gen2)
# born: 2026-05-29T23:51:36Z

"""
Hybrid Voronoi Fractional & Decision Hygiene with Ternary Lens Audit.

This module fuses the *Voronoi Fractional* algorithm from 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m1117_s2.py 
with the *Decision Hygiene* and *Ternary Lens Audit* framework from 
hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s1.py.

The mathematical bridge is the application of the Caputo fractional 
kernel to the feature-count vector produced by the hygiene regexes 
and the ternary lens audit report. The final hybrid score multiplies 
the hygiene score by a factor that depends on the normalized 
Voronoi fractional weights and incorporates the ternary lens audit findings.
"""

import math
import re
import numpy as np
from typing import Any, Iterable, List, Tuple
from pathlib import Path
from collections import Counter

def _gamma(z: float) -> float:
    _LANCZOS_G = 7
    _LANCZOS_C = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857
    ])
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


def nearest_point(point: Tuple[float, ...], seeds: List[Tuple[float, ...]]) -> int:
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean_distance(point, seeds[i]), i))


def voronoi_partition(seeds: List[Tuple[float, ...]], points: List[Tuple[float, ...]]) -> dict:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest_point(p, seeds)].append(p)
    return regions


def hybrid_voronoi_fractional(points: List[Tuple[float, ...]], seeds: List[Tuple[float, ...]], alpha: float, times: np.ndarray) -> dict:
    regions = voronoi_partition(seeds, points)
    kernel = caputo_kernel(alpha, times)
    weighted_regions = {}
    for seed_idx, region in regions.items():
        weighted_region = {}
        for point in region:
            weight = np.dot(kernel, [euclidean_distance(point, seeds[seed_idx])] * len(times))
            weighted_region[point] = weight
        weighted_regions[seed_idx] = weighted_region
    return weighted_regions


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
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)


def extract_features(text: str) -> Counter:
    features = Counter()
    features['evidence'] = len(EVIDENCE_RE.findall(text))
    features['planning'] = len(PLANNING_RE.findall(text))
    features['delay'] = len(DELAY_RE.findall(text))
    features['support'] = len(SUPPORT_RE.findall(text))
    features['boundary'] = len(BOUNDARY_RE.findall(text))
    features['outcome'] = len(OUTCOME_RE.findall(text))
    features['impulsive'] = len(IMPULSIVE_RE.findall(text))
    features['scarcity'] = len(SCARCITY_RE.findall(text))
    return features


def decision_hygiene_score(features: Counter) -> float:
    return sum(features.values()) / len(features)


def ternary_lens_audit(features: Counter) -> float:
    return features['evidence'] / (features['evidence'] + features['impulsive'] + features['scarcity'])


def hybrid_score(points: List[Tuple[float, ...]], seeds: List[Tuple[float, ...]], alpha: float, times: np.ndarray, text: str) -> float:
    features = extract_features(text)
    hygiene_score = decision_hygiene_score(features)
    voronoi_weights = hybrid_voronoi_fractional(points, seeds, alpha, times)
    audit_score = ternary_lens_audit(features)
    return hygiene_score * np.mean(list(voronoi_weights.values())) * audit_score


if __name__ == "__main__":
    points = [(0, 0), (1, 1), (2, 2)]
    seeds = [(0, 0), (2, 2)]
    alpha = 0.5
    times = np.array([1, 2, 3])
    text = "I have evidence and a plan, but I'm feeling impulsive."
    print(hybrid_score(points, seeds, alpha, times, text))