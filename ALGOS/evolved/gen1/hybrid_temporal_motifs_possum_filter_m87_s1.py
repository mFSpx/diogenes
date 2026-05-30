# DARWIN HAMMER — match 87, survivor 1
# gen: 1
# parent_a: temporal_motifs.py (gen0)
# parent_b: possum_filter.py (gen0)
# born: 2026-05-29T23:25:37Z

"""temporal_spatial_fusion.py
Hybrid algorithm merging temporal motif mining (temporal_motifs.py) with
possumm-style spatial diversity filtering (possum_filter.py).

Mathematical bridge:
- Parent A produces a discrete support count  s(p)  for each temporal pattern p.
- Parent B defines a binary proximity matrix D(i,j) = 1 if the haversine distance
  between two entities i and j is ≤ Δ (else 0) and a signature equality predicate.
- The fusion builds a joint score  
      S(p) = s(p) · (1 + z_s)  
  where z_s is the z‑score of the support distribution across patterns.
- Motif occurrences are represented as entities with a geographic centroid.
  A mask M(i,j) = D(i,j) ∧ (sig_i == sig_j) removes near‑duplicate motifs.
  The final set is the maximal independent set under M, exactly the
  possum filter applied to the temporally‑derived entities.

The code below implements this unified system, providing three core
functions that demonstrate the hybrid operation.
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

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


# ---------- Parent A – temporal utilities -----------------------------------

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


def mine_temporal_motifs(
    sessions: List[List[dict]],
    key: str = 'type',
    min_support: int = 2,
) -> List[TemporalMotif]:
    """Count identical sequences of the chosen key across sessions."""
    counter = Counter(
        tuple(str(e.get(key, '')) for e in s) for s in sessions
    )
    return [
        TemporalMotif(pattern=p, support=v)
        for p, v in counter.items()
        if v >= min_support
    ]


# ---------- Parent B – spatial utilities -------------------------------------

def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat, lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def signature_from_pattern(pattern: Tuple[str, ...]) -> str:
    """Canonical string signature for a motif pattern."""
    return " ".join(p.strip().lower() for p in pattern)


def keep_candidate(candidate: HybridMotif, selected: List[HybridMotif], delta_m: float) -> bool:
    """Return True if candidate is not a near‑duplicate of any already selected."""
    for existing in selected:
        same_kind = signature_from_pattern(candidate.pattern) == signature_from_pattern(
            existing.pattern
        )
        if same_kind and haversine_m(
            (candidate.centroid_lat, candidate.centroid_lon),
            (existing.centroid_lat, existing.centroid_lon),
        ) <= delta_m:
            return False
    return True


def filter_hybrid_motifs(
    motifs: Iterable[HybridMotif],
    delta_m: float = 75.0,
    sort_by_score: bool = True,
) -> List[HybridMotif]:
    """Possum‑style diversity filter applied to HybridMotif entities."""
    if delta_m < 0:
        raise ValueError("delta_m must be non‑negative")
    ordered = list(motifs)
    if sort_by_score:
        ordered.sort(key=lambda m: (-m.score, m.pattern))
    selected: List[HybridMotif] = []
    for motif in ordered:
        if keep_candidate(motif, selected, delta_m):
            selected.append(motif)
    return selected


# ---------- Hybrid core -----------------------------------------------------

def compute_pattern_centroids(
    sessions: List[List[dict]],
    pattern: Tuple[str, ...],
    key: str = 'type',
) -> Tuple[float, float]:
    """
    For all sessions whose event‑type sequence equals *pattern*,
    compute the geographic centroid of all contained events.
    """
    latitudes: List[float] = []
    longitudes: List[float] = []
    for sess in sessions:
        seq = tuple(str(e.get(key, '')) for e in sess)
        if seq == pattern:
            for e in sess:
                if 'lat' in e and 'lon' in e:
                    latitudes.append(float(e['lat']))
                    longitudes.append(float(e['lon']))
    if not latitudes:
        return (0.0, 0.0)
    # Vectorised centroid using numpy
    lat_arr = np.array(latitudes)
    lon_arr = np.array(longitudes)
    return (float(lat_arr.mean()), float(lon_arr.mean()))


def hybrid_motif_detection(
    sessions: List[List[dict]],
    key: str = 'type',
    min_support: int = 2,
) -> List[HybridMotif]:
    """
    Detect temporal motifs, enrich them with a spatial centroid,
    compute a z‑score over the support distribution and produce a combined score.
    """
    # Step 1 – raw temporal motifs
    raw_motifs = mine_temporal_motifs(sessions, key=key, min_support=min_support)

    # Step 2 – support distribution for z‑score
    supports = np.array([m.support for m in raw_motifs], dtype=float)
    if supports.size == 0:
        return []

    mean_s = supports.mean()
    std_s = supports.std(ddof=0) or 1.0
    z_scores = (supports - mean_s) / std_s

    # Step 3 – build HybridMotif list
    hybrid: List[HybridMotif] = []
    for motif, z in zip(raw_motifs, z_scores):
        centroid_lat, centroid_lon = compute_pattern_centroids(
            sessions, motif.pattern, key=key
        )
        # Combined score: temporal support weighted by (1 + z)
        combined_score = motif.support * (1.0 + z)
        hybrid.append(
            HybridMotif(
                pattern=motif.pattern,
                support=motif.support,
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                score=combined_score,
            )
        )
    return hybrid


def detect_spatiotemporal_motifs(
    events: List[dict],
    gap_seconds: float = 1800.0,
    delta_m: float = 75.0,
    key: str = 'type',
    min_support: int = 2,
) -> List[HybridMotif]:
    """
    End‑to‑end pipeline:
    1. Sessionize the raw events.
    2. Detect temporal motifs and attach spatial centroids.
    3. Apply the possum‑style diversity filter.
    Returns the final, non‑redundant set of spatio‑temporal motifs.
    """
    sessions = sessionize_events(events, gap_seconds=gap_seconds)
    hybrid = hybrid_motif_detection(
        sessions, key=key, min_support=min_support
    )
    filtered = filter_hybrid_motifs(hybrid, delta_m=delta_m, sort_by_score=True)
    return filtered


# ---------- Smoke test -------------------------------------------------------

def _generate_synthetic_events(num_events: int = 500) -> List[dict]:
    """Create a reproducible list of random events for demonstration."""
    random.seed(42)
    types = ['A', 'B', 'C', 'D']
    base_time = 1_600_000_000  # arbitrary epoch
    events: List[dict] = []
    for i in range(num_events):
        t = base_time + random.expovariate(1 / 300) * i  # loosely spaced
        lat = 37.0 + random.uniform(-0.05, 0.05)  # around San Francisco
        lon = -122.0 + random.uniform(-0.05, 0.05)
        typ = random.choice(types)
        events.append(
            {
                'id': f'e{i}',
                't': t,
                'lat': lat,
                'lon': lon,
                'type': typ,
            }
        )
    return events


if __name__ == "__main__":
    # Run a quick sanity check – should complete without exception.
    synthetic = _generate_synthetic_events()
    motifs = detect_spatiotemporal_motifs(
        synthetic,
        gap_seconds=1800.0,
        delta_m=75.0,
        key='type',
        min_support=2,
    )
    print(f"Detected {len(motifs)} spatio‑temporal motifs after filtering.")
    for m in motifs[:10]:
        print(
            f"Pattern: {m.pattern}, Support: {m.support}, "
            f"Centroid: ({m.centroid_lat:.5f}, {m.centroid_lon:.5f}), "
            f"Score: {m.score:.2f}"
        )
    sys.exit(0)