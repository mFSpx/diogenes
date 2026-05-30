# DARWIN HAMMER — match 3730, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m1117_s2.py (gen4)
# parent_b: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s1.py (gen2)
# born: 2026-05-29T23:51:36Z

"""Hybrid Voronoi‑Fractional / Decision‑Hygiene Entropy Module
==========================================================

Parent A – *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m1117_s2.py*  
 Provides a Voronoi partition of a point cloud and weights each point with a
 Caputo fractional kernel (order ``alpha``) evaluated over a time vector ``t``.

Parent B – *hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s1.py*  
 Extracts keyword‑based “hygiene” counts from free‑form text, computes a
 Shannon entropy of the resulting category distribution and modulates the
 score by a ternary‑lens audit factor.

**Mathematical Bridge**  
The bridge is the *distribution of fractional weights* obtained from Parent A.
Treat each Voronoi cell (seed) as a categorical bucket; the total fractional
weight per cell forms a probability vector **p**.  The Shannon entropy of **p**
is combined with the hygiene score derived from Parent B.  The final hybrid
metric is


HybridScore = HygieneScore × (1 + H_norm) × TernaryFactor


where ``H_norm = H / log(N)`` (``N`` = number of seeds) and ``TernaryFactor`` is
a scalar extracted from a simple ternary‑lens audit report.  This fuses the
spatial‑fractional analysis of Parent A with the textual‑decision hygiene of
Parent B into a single unified system.

"""

import math
import re
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – fractional Voronoi utilities
# ----------------------------------------------------------------------
def _gamma(z: float) -> float:
    """Lanczos approximation of the Gamma function."""
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
    """Caputo fractional kernel k(t) = t^{alpha-1} / Gamma(alpha)."""
    if alpha <= 0:
        raise ValueError("Fractional order alpha must be positive.")
    t = np.where(t == 0, 1e-12, t)  # avoid singularity at t=0
    return t ** (alpha - 1) / _gamma(alpha)


def euclidean_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nearest_point(point: Tuple[float, ...], seeds: List[Tuple[float, ...]]) -> int:
    """Index of the seed nearest to *point* (Euclidean metric)."""
    return min(range(len(seeds)),
               key=lambda i: (euclidean_distance(point, seeds[i]), i))


def voronoi_partition(seeds: List[Tuple[float, ...]],
                     points: List[Tuple[float, ...]]) -> Dict[int, List[Tuple[float, ...]]]:
    """Assign each point to its closest seed."""
    regions: Dict[int, List[Tuple[float, ...]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest_point(p, seeds)].append(p)
    return regions


def hybrid_voronoi_fractional(points: List[Tuple[float, ...]],
                              seeds: List[Tuple[float, ...]],
                              alpha: float,
                              times: np.ndarray) -> Dict[int, Dict[Tuple[float, ...], float]]:
    """
    Returns a nested mapping:
        seed_index → { point → weighted_value }

    The weight of a point is the dot‑product of the Caputo kernel with a
    constant vector whose entries are the Euclidean distance from the point
    to its seed.
    """
    regions = voronoi_partition(seeds, points)
    kernel = caputo_kernel(alpha, times)               # shape (len(times),)
    weighted_regions: Dict[int, Dict[Tuple[float, ...], float]] = {}

    for seed_idx, region in regions.items():
        weighted_region: Dict[Tuple[float, ...], float] = {}
        for point in region:
            dist = euclidean_distance(point, seeds[seed_idx])
            # Replicate distance for each time sample and weight with kernel
            weight = float(np.dot(kernel, np.full_like(kernel, dist)))
            weighted_region[point] = weight
        weighted_regions[seed_idx] = weighted_region
    return weighted_regions

# ----------------------------------------------------------------------
# Parent B – decision hygiene & entropy utilities
# ----------------------------------------------------------------------
# Regexes for keyword categories (identical to the original parent)
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
    """
    Scan a list of free‑form strings and count occurrences of each keyword class.
    Returns a Counter mapping class name → count.
    """
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
    """
    Simple linear hygiene score: sum of all category counts.
    Could be replaced by a more sophisticated weighting scheme.
    """
    return float(sum(counter.values()))


def shannon_entropy(probs: np.ndarray) -> float:
    """Standard Shannon entropy (natural log) for a probability vector."""
    probs = probs[probs > 0]  # filter zero entries to avoid log(0)
    return -float(np.sum(probs * np.log(probs)))


# ----------------------------------------------------------------------
# Ternary Lens Audit – lightweight placeholder
# ----------------------------------------------------------------------
def ternary_lens_audit(report: List[str]) -> float:
    """
    Very small mock‑up of a ternary lens audit.
    The report list contains strings that include the words 'red', 'yellow', or 'green'.
    The function returns a factor in [0.5, 1.5] where more 'green' pushes the factor up.
    """
    tally = Counter()
    for line in report:
        line_low = line.lower()
        if "red" in line_low:
            tally["red"] += 1
        if "yellow" in line_low:
            tally["yellow"] += 1
        if "green" in line_low:
            tally["green"] += 1
    total = sum(tally.values()) or 1
    # map to factor: 0.5 (all red) → 1.5 (all green), linear interpolation
    red_frac = tally["red"] / total
    green_frac = tally["green"] / total
    factor = 0.5 + (green_frac - red_frac)  # range roughly [-0.5, 1.5]
    return max(0.5, min(1.5, factor))        # clamp to [0.5, 1.5]


# ----------------------------------------------------------------------
# Hybrid core – merging spatial‑fractional and textual‑decision components
# ----------------------------------------------------------------------
def hybrid_metric(points: List[Tuple[float, ...]],
                  seeds: List[Tuple[float, ...]],
                  alpha: float,
                  times: np.ndarray,
                  texts: List[str],
                  audit_report: List[str]) -> float:
    """
    Compute the unified hybrid score.

    Steps
    -----
    1. Fractional Voronoi weighting → total weight per seed.
    2. Convert the seed‑wise weights into a probability distribution.
    3. Shannon entropy of that distribution, normalized by log(N).
    4. Hygiene score from the supplied texts.
    5. Ternary lens factor from the audit report.
    6. Combine as: score = hygiene_score × (1 + H_norm) × ternary_factor.
    """
    # 1. Fractional Voronoi weighting
    weighted_regions = hybrid_voronoi_fractional(points, seeds, alpha, times)

    # 2. Aggregate total weight per seed
    total_weights = np.array([
        sum(region.values()) for region in weighted_regions.values()
    ], dtype=float)

    # Guard against all‑zero weights (unlikely but possible)
    if total_weights.sum() == 0:
        probs = np.full_like(total_weights, 1.0 / len(total_weights))
    else:
        probs = total_weights / total_weights.sum()

    # 3. Normalized Shannon entropy
    H = shannon_entropy(probs)
    H_max = math.log(len(seeds)) if len(seeds) > 1 else 1.0
    H_norm = H / H_max

    # 4. Hygiene score
    h_counts = hygiene_counts(texts)
    H_score = hygiene_score(h_counts)

    # 5. Ternary lens factor
    T_factor = ternary_lens_audit(audit_report)

    # 6. Final hybrid metric
    return H_score * (1.0 + H_norm) * T_factor


# ----------------------------------------------------------------------
# Additional helper functions to showcase the hybrid workflow
# ----------------------------------------------------------------------
def sample_points(num: int, dim: int = 2, low: float = -10.0, high: float = 10.0) -> List[Tuple[float, ...]]:
    """Generate ``num`` random points in ``dim``‑dimensional space."""
    return [tuple(random.uniform(low, high) for _ in range(dim)) for _ in range(num)]


def sample_texts(num: int) -> List[str]:
    """Create dummy sentences containing a random selection of keyword categories."""
    keywords = [
        "evidence", "plan", "wait", "support", "boundary", "done",
        "rage", "broke"
    ]
    sentences = []
    for _ in range(num):
        chosen = random.sample(keywords, k=random.randint(1, 3))
        sentences.append(" ".join(chosen) + ".")
    return sentences


def sample_audit(num: int) -> List[str]:
    """Generate a fake ternary lens audit report."""
    colors = ["red", "yellow", "green"]
    lines = []
    for _ in range(num):
        line = f"Status: {random.choice(colors).upper()}"
        lines.append(line)
    return lines


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    NUM_POINTS = 200
    NUM_SEEDS = 5
    ALPHA = 0.75
    TIMES = np.linspace(0.1, 5.0, 20)          # time vector for kernel

    # Generate synthetic data
    pts = sample_points(NUM_POINTS, dim=2)
    sds = sample_points(NUM_SEEDS, dim=2)
    txts = sample_texts(15)
    audit = sample_audit(7)

    # Compute hybrid metric
    score = hybrid_metric(
        points=pts,
        seeds=sds,
        alpha=ALPHA,
        times=TIMES,
        texts=txts,
        audit_report=audit,
    )
    print(f"Hybrid metric score: {score:.4f}")