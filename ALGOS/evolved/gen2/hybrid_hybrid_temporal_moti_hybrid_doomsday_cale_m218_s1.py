# DARWIN HAMMER — match 218, survivor 1
# gen: 2
# parent_a: hybrid_temporal_motifs_possum_filter_m87_s0.py (gen1)
# parent_b: hybrid_doomsday_calendar_gini_coefficient_m49_s3.py (gen1)
# born: 2026-05-29T23:27:39Z

"""Hybrid Temporal Motif & Weekday Gini Fusion

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – spatial‑temporal motif mining.  It provides:
  - Haversine distance for geographic proximity.
  - Signature‑based candidate filtering (`keep_candidate`).
  - Sessionisation of timestamped events and mining of frequent
    categorical sequences (temporal motifs).

* **Parent B** – weekday inequality analysis.  It provides:
  - A vectorised Doomsday‑based weekday calculation (`doomsday_numpy`).
  - The Gini coefficient on an arbitrary 1‑D distribution
    (`gini_coefficient_numpy`), used here on the weekday count vector.

**Mathematical Bridge**

For every mined temporal motif we examine the *set of sessions* in which
the motif occurs.  The weekday distribution of the timestamps inside
those sessions is obtained via the Doomsday algorithm, yielding a 7‑element
count vector `c = (c₀,…,c₆)`.  The Gini coefficient `G(c)` measures the
inequality of weekday occurrence.  By coupling the motif’s raw support
`S` with `G(c)` we obtain a joint quality metric  


Q = S * (1 - G(c))


where `(1‑G)` rewards motifs that are spread uniformly across the week,
and the product preserves the original support scale.  The fusion thus
creates a unified system that simultaneously respects spatial proximity,
temporal ordering, and weekday inequality.

The module implements three public functions that showcase this hybrid
behaviour:

1. `sessionize_events`
2. `mine_temporal_motifs`
3. `hybrid_motif_gini_score`

All code is pure Python 3 with only `numpy`, `math`, `datetime`,
`random`, `sys`, and `pathlib` as external imports.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – spatial‑temporal utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat, lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def signature(e: Entity) -> str:
    """Canonical string used for similarity comparison."""
    return (e.address_signature or e.category).strip().lower()


def keep_candidate(candidate: Entity, selected: List[Entity], delta_m: float) -> bool:
    """Return True if *candidate* is sufficiently far (or different) from all *selected*."""
    for existing in selected:
        same_kind = (
            signature(candidate) == signature(existing)
            or candidate.category.strip().lower() == existing.category.strip().lower()
        )
        if same_kind and haversine_m((candidate.lat, candidate.lon), (existing.lat, existing.lon)) <= delta_m:
            return False
    return True


# ----------------------------------------------------------------------
# Parent B – Doomsday & Gini utilities
# ----------------------------------------------------------------------


def doomsday_numpy(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """
    Vectorised weekday calculation using the Doomsday algorithm.
    Returns values in the range 0‑6 where 0 = Monday.
    """
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    # Convert to POSIX seconds then to Python weekday (Mon=0,…)
    py_weekday = np.fromiter(
        (dt.datetime.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday() for d in dates.ravel()),
        dtype=np.int8,
        count=dates.size,
    )
    return py_weekday.reshape(dates.shape[:-1])


def gini_coefficient_numpy(values: np.ndarray) -> float:
    """Standard Gini coefficient for a 1‑D non‑negative array."""
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    xs = np.sort(values.astype(float))
    if xs.size == 0 or xs.sum() == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = xs.size
    i = np.arange(1, n + 1)  # 1‑based index
    numerator = np.sum((2 * i - n - 1) * xs)
    denominator = n * xs.sum()
    return numerator / denominator


def weekday_counts(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> np.ndarray:
    """Return a length‑7 array with counts for each weekday (Mon=0,…,Sun=6)."""
    years, months, days = [], [], []
    for d in dates:
        if isinstance(d, dt.date):
            y, m, day = d.year, d.month, d.day
        else:
            y, m, day = d
        years.append(y)
        months.append(m)
        days.append(day)
    years_np = np.array(years, dtype=np.int32)
    months_np = np.array(months, dtype=np.int32)
    days_np = np.array(days, dtype=np.int32)
    weekdays = doomsday_numpy(years_np, months_np, days_np)
    counts = np.bincount(weekdays, minlength=7)
    return counts.astype(int)


def weekday_gini(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> float:
    """Gini coefficient of the weekday distribution of *dates*."""
    counts = weekday_counts(dates)
    return gini_coefficient_numpy(counts)


# ----------------------------------------------------------------------
# Hybrid core: sessionisation, motif mining, and Gini‑enhanced scoring
# ----------------------------------------------------------------------


def sessionize_events(events: List[Dict[str, Any]], gap_seconds: float = 1800.0) -> List[List[Dict[str, Any]]]:
    """
    Group chronological events into sessions.
    A new session starts when the gap between consecutive timestamps exceeds *gap_seconds*.
    Each event dict must contain a numeric ``timestamp`` field (seconds since epoch).
    """
    if not events:
        return []
    # Ensure chronological order
    events_sorted = sorted(events, key=lambda e: e["timestamp"])
    sessions: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [events_sorted[0]]
    for prev, cur in zip(events_sorted, events_sorted[1:]):
        if cur["timestamp"] - prev["timestamp"] <= gap_seconds:
            current.append(cur)
        else:
            sessions.append(current)
            current = [cur]
    sessions.append(current)
    return sessions


def mine_temporal_motifs(
    sessions: List[List[Dict[str, Any]]],
    key: str = "category",
    min_support: int = 2,
    max_pattern_len: int = 3,
) -> List[Tuple[Tuple[str, ...], int]]:
    """
    Very lightweight motif miner.
    For each session we extract the ordered sequence of ``key`` values.
    All contiguous subsequences up to *max_pattern_len* are counted.
    Returns a list of (pattern, support) tuples with support >= *min_support*.
    """
    pattern_counts: Dict[Tuple[str, ...], int] = {}
    for sess in sessions:
        seq = [str(event.get(key, "")) for event in sess]
        n = len(seq)
        for length in range(1, max_pattern_len + 1):
            for i in range(n - length + 1):
                pat = tuple(seq[i : i + length])
                pattern_counts[pat] = pattern_counts.get(pat, 0) + 1
    # Filter by support
    return [(pat, sup) for pat, sup in pattern_counts.items() if sup >= min_support]


def hybrid_motif_gini_score(
    entities: Iterable[Entity],
    events: List[Dict[str, Any]],
    delta_m: float,
    gap_seconds: float = 1800.0,
    min_support: int = 2,
) -> List[Dict[str, Any]]:
    """
    Complete hybrid pipeline:

    1. Spatial filtering of *entities* using ``keep_candidate``.
    2. Sessionisation of *events*.
    3. Mining of temporal motifs.
    4. For each motif, compute the Gini coefficient of the weekday
       distribution of timestamps in the sessions where the motif appears.
    5. Produce a combined quality score Q = support * (1 - Gini).

    Returns a list of dictionaries, each describing one motif.
    """
    # ------------------------------------------------------------------
    # Step 1 – spatial filtering
    # ------------------------------------------------------------------
    ordered_entities = list(entities)
    ordered_entities.sort(key=lambda e: (-e.score, e.id))
    selected: List[Entity] = []
    for ent in ordered_entities:
        if keep_candidate(ent, selected, delta_m):
            selected.append(ent)

    # ------------------------------------------------------------------
    # Step 2 – sessionisation
    # ------------------------------------------------------------------
    sessions = sessionize_events(events, gap_seconds)

    # ------------------------------------------------------------------
    # Step 3 – motif mining
    # ------------------------------------------------------------------
    motifs = mine_temporal_motifs(sessions, key="category", min_support=min_support)

    # ------------------------------------------------------------------
    # Helper: extract dates from a session
    # ------------------------------------------------------------------
    def session_dates(sess: List[Dict[str, Any]]) -> List[Tuple[int, int, int]]:
        return [
            dt.datetime.utcfromtimestamp(ev["timestamp"]).date().timetuple()[:3] for ev in sess
        ]

    # ------------------------------------------------------------------
    # Step 4 & 5 – compute Gini per motif and final score
    # ------------------------------------------------------------------
    results: List[Dict[str, Any]] = []
    for pattern, support in motifs:
        # Find sessions containing the pattern (as a contiguous subsequence)
        matching_sessions = [
            sess
            for sess in sessions
            if any(
                tuple(str(ev.get("category", "")) for ev in sess[i : i + len(pattern)]) == pattern
                for i in range(len(sess) - len(pattern) + 1)
            )
        ]
        # Gather all dates from matching sessions
        all_dates = [d for sess in matching_sessions for d in session_dates(sess)]
        # Compute Gini on weekday distribution
        gini = weekday_gini(all_dates) if all_dates else 0.0
        combined_score = support * (1.0 - gini)
        results.append(
            {
                "pattern": pattern,
                "support": support,
                "gini": gini,
                "combined_score": combined_score,
                "matched_sessions": len(matching_sessions),
            }
        )
    # Sort by combined_score descending for convenience
    results.sort(key=lambda r: r["combined_score"], reverse=True)
    return results


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a few synthetic entities
    entities = [
        Entity(id="A", lat=40.0, lon=-73.0, category="restaurant", score=10.0),
        Entity(id="B", lat=40.001, lon=-73.001, category="restaurant", score=9.0),
        Entity(id="C", lat=41.0, lon=-74.0, category="cafe", score=8.0),
    ]

    # Synthetic events: timestamps spread over several days, with categories
    now = int(dt.datetime.utcnow().timestamp())
    events = []
    categories = ["restaurant", "cafe", "park"]
    for day_offset in range(7):
        for _ in range(random.randint(1, 3)):
            ts = now - day_offset * 86_400 + random.randint(0, 3_600)
            events.append(
                {
                    "timestamp": ts,
                    "category": random.choice(categories),
                    "entity_id": random.choice(entities).id,
                }
            )

    # Run the hybrid pipeline
    hybrid_results = hybrid_motif_gini_score(
        entities=entities,
        events=events,
        delta_m=200.0,          # 200 m spatial exclusion radius
        gap_seconds=1800.0,     # 30 min session gap
        min_support=2,
    )

    # Print results
    for res in hybrid_results:
        print(
            f"Pattern {res['pattern']} | support={res['support']} | "
            f"Gini={res['gini']:.3f} | combined={res['combined_score']:.2f} | "
            f"sessions={res['matched_sessions']}"
        )