# DARWIN HAMMER — match 1701, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_ternary_route_hybrid_workshare_all_m171_s0.py (gen2)
# parent_b: hybrid_temporal_motifs_possum_filter_m87_s1.py (gen1)
# born: 2026-05-29T23:38:19Z

"""Hybrid Allocation‑Similarity‑Motif Engine
Parents:
- hybrid_hybrid_ternary_route_hybrid_workshare_all_m171_s0.py (allocation + SSIM)
- hybrid_temporal_motifs_possum_filter_m87_s1.py (temporal motif support, z‑score, spatial possum filter)

Mathematical Bridge
-------------------
Parent A distributes a scalar resource (work units) among *groups* using a deterministic
portion and a stochastic LLM‑driven portion.  Parent B assigns a scalar *support* to
temporal motifs, normalises it with a z‑score and then multiplies by a similarity factor.
Both pipelines therefore share the same abstract operation:


weight_i = base_i · similarity_i


where `base_i` is a raw count (work units or support) and `similarity_i` is a
dimension‑less factor (SSIM or (1+z)).  The hybrid algorithm fuses them by:

1. Encoding each temporal motif as a binary feature vector `v_i`.
2. Computing SSIM(v_i, v_proto) against a prototype vector `v_proto`.
3. Computing the motif‑wise z‑score `z_i` of the support distribution.
4. Defining a joint score  
   `J_i = support_i · (1 + z_i) · (1 + SSIM_i)`.
5. Mapping each motif to a *group* and allocating a total resource `U`:
   - a deterministic fraction `α·U` is split equally,
   - the remainder `(1‑α)·U` is distributed proportionally to the sum of `J_i`
     belonging to each group.
6. Applying the possum spatial filter to remove near‑duplicate motifs, yielding the
   final independent set.

The code below implements this fused mathematics with three public functions
`encode_patterns`, `compute_joint_scores`, `allocate_resources`, plus a
`possum_filter` helper and a runnable smoke‑test."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Shared constants
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
DETERMINISTIC_PCT = 90.0  # percent of total units allocated deterministically


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TemporalMotif:
    """Discrete temporal pattern with raw support count."""
    pattern: Tuple[str, ...]
    support: int


@dataclass(frozen=True)
class HybridMotif:
    """Spatio‑temporal motif enriched with joint score and centroid."""
    pattern: Tuple[str, ...]
    support: int
    centroid_lat: float
    centroid_lon: float
    joint_score: float
    group: str


# ----------------------------------------------------------------------
# Parent A – similarity utilities
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round to six decimal places for reproducibility."""
    return round(float(value), 6)


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Structural Similarity Index for two 1‑D vectors."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return _pct(numerator / denominator)


# ----------------------------------------------------------------------
# Parent B – temporal motif utilities
# ----------------------------------------------------------------------
def _z_scores(values: List[int]) -> List[float]:
    """Return z‑score for each value in `values`."""
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return [0.0 for _ in values]
    return [ (v - mean) / std for v in values ]


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great‑circle distance in kilometres."""
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ----------------------------------------------------------------------
# Fusion helpers
# ----------------------------------------------------------------------
def encode_patterns(motifs: List[TemporalMotif]) -> Tuple[Dict[Tuple[str, ...], np.ndarray], np.ndarray]:
    """
    Build a binary vocabulary from all pattern tokens and encode each motif as a
    0/1 vector.  Returns a mapping motif→vector and the prototype (mean) vector.
    """
    vocab: List[str] = sorted({token for m in motifs for token in m.pattern})
    index = {token: i for i, token in enumerate(vocab)}
    vectors: Dict[Tuple[str, ...], np.ndarray] = {}
    for m in motifs:
        vec = np.zeros(len(vocab), dtype=float)
        for token in m.pattern:
            vec[index[token]] = 1.0
        vectors[m.pattern] = vec
    prototype = np.mean(list(vectors.values()), axis=0) if vectors else np.zeros(len(vocab))
    return vectors, prototype


def compute_joint_scores(motifs: List[TemporalMotif]) -> List[HybridMotif]:
    """
    For each temporal motif:
      1. Compute its SSIM against the prototype vector.
      2. Compute its z‑score over the support distribution.
      3. Combine into a joint score:
         J = support · (1 + z) · (1 + SSIM)
      4. Assign a random geographic centroid.
      5. Map the motif to a group via hash modulo.
    Returns a list of `HybridMotif` objects.
    """
    if not motifs:
        return []

    # Encode patterns and obtain prototype
    vectors, prototype = encode_patterns(motifs)

    # Z‑scores of supports
    supports = [m.support for m in motifs]
    z_vals = _z_scores(supports)

    hybrid_list: List[HybridMotif] = []
    for m, z in zip(motifs, z_vals):
        vec = vectors[m.pattern]
        ssim = compute_ssim(vec, prototype) if prototype.size else 0.0
        joint = m.support * (1 + z) * (1 + ssim)
        # Random but reproducible centroid (seeded by pattern hash)
        rng = random.Random(hash(m.pattern))
        lat = rng.uniform(-90.0, 90.0)
        lon = rng.uniform(-180.0, 180.0)
        group = GROUPS[hash(m.pattern) % len(GROUPS)]
        hybrid = HybridMotif(
            pattern=m.pattern,
            support=m.support,
            centroid_lat=_pct(lat),
            centroid_lon=_pct(lon),
            joint_score=_pct(joint),
            group=group,
        )
        hybrid_list.append(hybrid)
    return hybrid_list


def possum_filter(motifs: List[HybridMotif], distance_km: float = 10.0) -> List[HybridMotif]:
    """
    Apply the possum spatial filter:
      - Build a binary mask M(i,j) = 1 iff distance ≤ Δ and patterns equal.
      - Return a maximal independent set using a greedy strategy (largest joint_score first).
    """
    if not motifs:
        return []

    # Sort descending by joint_score for greedy selection
    sorted_motifs = sorted(motifs, key=lambda m: m.joint_score, reverse=True)
    selected: List[HybridMotif] = []
    for cand in sorted_motifs:
        conflict = False
        for sel in selected:
            if cand.pattern == sel.pattern:
                d = haversine(cand.centroid_lat, cand.centroid_lon,
                              sel.centroid_lat, sel.centroid_lon)
                if d <= distance_km:
                    conflict = True
                    break
        if not conflict:
            selected.append(cand)
    return selected


def allocate_resources(
    total_units: float,
    hybrid_motifs: List[HybridMotif],
    deterministic_pct: float = DETERMINISTIC_PCT,
    groups: Tuple[str, ...] = GROUPS,
) -> Dict[str, float]:
    """
    Allocate `total_units` among `groups` using the hybrid scoring.
    Deterministic portion (α·U) is split equally.
    The stochastic portion is divided proportionally to the sum of joint scores
    of motifs belonging to each group.
    Returns a mapping group → allocated units (rounded to 6 decimals).
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0 <= deterministic_pct <= 100):
        raise ValueError("deterministic_pct must be in [0,100]")
    if not groups:
        raise ValueError("at least one group required")

    deterministic_units = total_units * deterministic_pct / 100.0
    stochastic_units = total_units - deterministic_units
    per_group_eq = deterministic_units / len(groups)

    # Aggregate joint scores per group
    group_scores: Dict[str, float] = {g: 0.0 for g in groups}
    for m in hybrid_motifs:
        if m.group in group_scores:
            group_scores[m.group] += m.joint_score

    total_score = sum(group_scores.values())
    allocations: Dict[str, float] = {}
    for g in groups:
        share = per_group_eq
        if total_score > 0:
            share += stochastic_units * (group_scores[g] / total_score)
        allocations[g] = _pct(share)
    return allocations


# ----------------------------------------------------------------------
# Example pipeline exposing three public functions
# ----------------------------------------------------------------------
def hybrid_pipeline(total_units: float, motifs: List[TemporalMotif]) -> Tuple[Dict[str, float], List[HybridMotif]]:
    """
    End‑to‑end hybrid operation:
      1. Compute joint scores for the supplied motifs.
      2. Filter duplicates with the possum filter.
      3. Allocate `total_units` across groups based on the filtered set.
    Returns the allocation dictionary and the filtered list of HybridMotif objects.
    """
    scored = compute_joint_scores(motifs)
    filtered = possum_filter(scored)
    allocation = allocate_resources(total_units, filtered)
    return allocation, filtered


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Construct a tiny synthetic motif catalogue
    sample_motifs = [
        TemporalMotif(pattern=("A", "B", "C"), support=120),
        TemporalMotif(pattern=("A", "D"), support=80),
        TemporalMotif(pattern=("B", "C", "E"), support=150),
        TemporalMotif(pattern=("F",), support=30),
        TemporalMotif(pattern=("A", "B", "C"), support=115),  # duplicate pattern
    ]

    total_resource = 1000.0
    allocation_result, final_motifs = hybrid_pipeline(total_resource, sample_motifs)

    print("Allocation per group:")
    for grp, amt in allocation_result.items():
        print(f"  {grp}: {amt}")

    print("\nFiltered Hybrid Motifs (joint_score, group, centroid):")
    for hm in final_motifs:
        print(
            f"  pattern={hm.pattern}, support={hm.support}, "
            f"joint_score={hm.joint_score}, group={hm.group}, "
            f"centroid=({hm.centroid_lat}, {hm.centroid_lon})"
        )