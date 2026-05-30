# DARWIN HAMMER — match 3730, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m1117_s2.py (gen4)
# parent_b: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s1.py (gen2)
# born: 2026-05-29T23:51:36Z

import math
import re
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

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
    return min(range(len(seeds)),
               key=lambda i: (euclidean_distance(point, seeds[i]), i))


def voronoi_partition(seeds: List[Tuple[float, ...]],
                     points: List[Tuple[float, ...]]) -> Dict[int, List[Tuple[float, ...]]]:
    regions: Dict[int, List[Tuple[float, ...]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest_point(p, seeds)].append(p)
    return regions


def hybrid_voronoi_fractional(points: List[Tuple[float, ...]],
                              seeds: List[Tuple[float, ...]],
                              alpha: float,
                              times: np.ndarray) -> Dict[int, Dict[Tuple[float, ...], float]]:
    regions = voronoi_partition(seeds, points)
    kernel = caputo_kernel(alpha, times)               
    weighted_regions: Dict[int, Dict[Tuple[float, ...], float]] = {}

    for seed_idx, region in regions.items():
        weighted_region: Dict[Tuple[float, ...], float] = {}
        for point in region:
            dist = euclidean_distance(point, seeds[seed_idx])
            weight = float(np.dot(kernel, np.full_like(kernel, dist)))
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


def hygiene_counts(texts: List[str]) -> Counter:
    counters = Counter()
    regex_map = {
        "evidence": EVIDENCE_RE,
        "planning": PLANNING_RE,
        "delay": DELAY_RE,
        "support": SUPPORT_RE,
        "boundary": BOUNDARY_RE,
        "outcome": OUTCOME_RE,
        "impulsive": IMPULSIVE_RE,
        "scarcity": SCARCITY_RE,
    }
    for txt in texts:
        for label, rex in regex_map.items():
            if rex.search(txt):
                counters[label] += 1
    return counters


def hygiene_score(counter: Counter) -> float:
    return float(sum(counter.values()))


def shannon_entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]  
    return -float(np.sum(probs * np.log(probs)))


def ternary_lens_audit(report: List[str]) -> float:
    tally = Counter()
    for line in report:
        line_low = line.lower()
        if "red" in line_low:
            tally["red"] += 1
        elif "yellow" in line_low:
            tally["yellow"] += 1
        elif "green" in line_low:
            tally["green"] += 1
    total = sum(tally.values())
    if total == 0:
        return 1.0
    green_frac = tally["green"] / total
    return 0.5 + green_frac


def improved_hybrid_score(points: List[Tuple[float, ...]],
                           seeds: List[Tuple[float, ...]],
                           alpha: float,
                           times: np.ndarray,
                           texts: List[str],
                           report: List[str]) -> float:
    weighted_regions = hybrid_voronoi_fractional(points, seeds, alpha, times)
    total_weight = sum(sum(region.values()) for region in weighted_regions.values())
    probs = np.array([sum(region.values()) / total_weight for region in weighted_regions.values()])
    entropy = shannon_entropy(probs)
    N = len(seeds)
    H_norm = entropy / np.log(N) if N > 1 else 0.0
    hygiene_cnt = hygiene_counts(texts)
    hygiene = hygiene_score(hygiene_cnt)
    ternary_factor = ternary_lens_audit(report)
    return hygiene * (1.0 + H_norm) * ternary_factor


def main():
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    seeds = [(0.0, 0.0), (2.0, 2.0)]
    alpha = 0.5
    times = np.linspace(0.0, 10.0, 100)
    texts = ["This is a test with evidence.", "This is another test."]
    report = ["Something went green.", "Something else went red."]
    score = improved_hybrid_score(points, seeds, alpha, times, texts, report)
    print(score)


if __name__ == "__main__":
    main()