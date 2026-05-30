# DARWIN HAMMER — match 1701, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_ternary_route_hybrid_workshare_all_m171_s0.py (gen2)
# parent_b: hybrid_temporal_motifs_possum_filter_m87_s1.py (gen1)
# born: 2026-05-29T23:38:19Z

"""
This module fuses the hybrid_hybrid_ternary_route_hybrid_workshare_all_m171_s0 and 
hybrid_temporal_motifs_possum_filter_m87_s1 algorithms. The mathematical bridge 
between the two structures is the concept of allocation and distribution, 
where the workshare allocator distributes work units among different groups, 
and the temporal motifs produce a discrete support count for each temporal pattern.
The fusion builds a joint allocation score 
      A(p) = a(p) · (1 + z_s)  
  where a(p) is the allocation and z_s is the z-score of the support distribution 
  across patterns. The allocation is then used to filter motifs using a possum-style 
  spatial diversity filter.

"""

import numpy as np
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

# ---------- Data structures -------------------------------------------------

@dataclass(frozen=True)
class BurstSignal:
    key: str
    count: int
    z_score: float

@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int

@dataclass(frozen=True)
class HybridMotif:
    """Entity representing a spatio‑temporal motif."""
    pattern: Tuple[str, ...]
    support: int
    centroid_lat: float
    centroid_lon: float
    score: float  # combined temporal‑spatial score

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two vectors.
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
    return ssim

def allocate_workshare(*, total_units: float, deterministic_target_pct: float = 90.0, groups: tuple[str, ...] = GROUPS) -> dict[str, float]:
    """
    Allocate work units among different groups.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups)
    lanes = [
        {
            "group": group,
            "llm_units": _pct(per_group),
            "llm_share_pct": _pct(100.0 / len(groups)),
            "proof_required": True,
        }
        for group in groups
    ]
    return {
        "total_units": _pct(total_units),
        "lanes": lanes,
    }

def sessionize_events(events: List[dict], gap_seconds: float = 1800.0) -> List[List[dict]]:
    """Group events into sessions separated by a temporal gap."""
    sessions: List[List[dict]] = []
    cur: List[dict] = []
    last: float | None = None
    for e in sorted(events, key=lambda x: float(x.get('t', 0))):
        t = float(e.get('t', 0))
        if cur and last is not None and t - last > gap_seconds:
            sessions.append(cur)
            cur = []
        cur.append(e)
        last = t
    if cur:
        sessions.append(cur)
    return sessions

def compute_allocation_score(support: int, allocation: dict) -> float:
    """
    Compute the allocation score for a given support and allocation.
    """
    z_score = (support - np.mean([lane["llm_units"] for lane in allocation["lanes"]])) / np.std([lane["llm_units"] for lane in allocation["lanes"]])
    return support * (1 + z_score)

def filter_motifs(motifs: List[HybridMotif], allocation: dict) -> List[HybridMotif]:
    """
    Filter motifs using a possum-style spatial diversity filter.
    """
    # Create a proximity matrix
    proximity_matrix = np.zeros((len(motifs), len(motifs)))
    for i in range(len(motifs)):
        for j in range(i+1, len(motifs)):
            distance = math.sqrt((motifs[i].centroid_lat - motifs[j].centroid_lat)**2 + (motifs[i].centroid_lon - motifs[j].centroid_lon)**2)
            if distance <= 1:  # adjust this value as needed
                proximity_matrix[i, j] = 1
                proximity_matrix[j, i] = 1

    # Create a signature equality predicate
    signature_matrix = np.zeros((len(motifs), len(motifs)))
    for i in range(len(motifs)):
        for j in range(i+1, len(motifs)):
            if motifs[i].pattern == motifs[j].pattern:
                signature_matrix[i, j] = 1
                signature_matrix[j, i] = 1

    # Compute the mask
    mask = np.logical_and(proximity_matrix, signature_matrix)

    # Compute the maximal independent set
    max_independent_set = []
    for i in range(len(motifs)):
        if not np.any(mask[i, :]):
            max_independent_set.append(motifs[i])

    return max_independent_set

def hybrid_operation(events: List[dict], total_units: float) -> List[HybridMotif]:
    """
    Perform the hybrid operation.
    """
    sessions = sessionize_events(events)
    motifs = []
    for session in sessions:
        # Compute the support for each motif
        support_counts = Counter([tuple([event.get('pattern', '') for event in session])])
        for pattern, support in support_counts.items():
            motifs.append(HybridMotif(pattern, support, 0.0, 0.0, 0.0))

    # Compute the allocation
    allocation = allocate_workshare(total_units=total_units)

    # Compute the allocation scores
    scored_motifs = []
    for motif in motifs:
        score = compute_allocation_score(motif.support, allocation)
        scored_motifs.append(HybridMotif(motif.pattern, motif.support, motif.centroid_lat, motif.centroid_lon, score))

    # Filter the motifs
    filtered_motifs = filter_motifs(scored_motifs, allocation)

    return filtered_motifs

if __name__ == "__main__":
    events = [
        {"t": 1643723400, "pattern": "A"},
        {"t": 1643723410, "pattern": "B"},
        {"t": 1643723420, "pattern": "A"},
        {"t": 1643723430, "pattern": "C"},
    ]
    total_units = 100.0
    filtered_motifs = hybrid_operation(events, total_units)
    for motif in filtered_motifs:
        print(motif)