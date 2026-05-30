# DARWIN HAMMER — match 4372, survivor 2
# gen: 2
# parent_a: chelydrid_ambush.py (gen0)
# parent_b: hybrid_temporal_motifs_possum_filter_m87_s0.py (gen1)
# born: 2026-05-29T23:55:21Z

"""Hybrid Kinematic‑Motif Engine
Combines:
- chelydrid_ambush.py (physics of a burst strike with quadratic drag)
- hybrid_temporal_motifs_possum_filter_m87_s0.py (spatial‑temporal burst detection,
  motif mining, and geographic filtering)

Mathematical bridge:
Both parents expose a notion of “distance” that can be expressed as an integral of a
velocity‑like quantity under a force minus a drag term.
· In the strike model, drag ∝ v·|v| and distance is the spatial travel of a head.
· In the motif model, successive events are separated by a geographic distance d_geo.
  We reinterpret d_geo as a quadratic drag coefficient acting on a virtual velocity
  that represents the temporal‑motif “force” (burst count or support).  The same ODE

      dv/dt = F(t)/m – C·d_geo·v·|v|

  is integrated, yielding a cumulative kinematic metric that fuses temporal‑motif
  support (F) with geographic dispersion (drag).  The result is a HybridState that
  can be used for scoring, filtering, or further analysis.
"""

import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class StrikeState:
    velocity: float
    distance: float
    peak_velocity: float


def integrate_strike(
    force_series: Iterable[float],
    dt: float,
    m_head: float,
    drag_cd: float = 0.3,
    fluid_density: float = 1000.0,
    area: float = 0.001,
    v0: float = 0.0,
) -> StrikeState:
    """Integrate a 1‑D strike under quadratic drag."""
    if dt <= 0 or m_head <= 0 or drag_cd < 0 or fluid_density < 0 or area < 0:
        raise ValueError("invalid strike parameters")
    v = v0
    x = 0.0
    peak = v0
    for force in force_series:
        drag = drag_cd * fluid_density * area * v * abs(v) / (2.0 * m_head)
        acc = force / m_head - drag
        v = max(0.0, v + acc * dt)
        x += v * dt
        peak = max(peak, v)
    return StrikeState(v, x, peak)


def pulse_force(peak_force: float, steps: int) -> List[float]:
    """Triangular pulse of length ``steps`` with maximum ``peak_force``."""
    if peak_force < 0 or steps <= 0:
        raise ValueError("peak_force non-negative and steps positive required")
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    return [
        peak_force * max(0.0, 1.0 - abs(i - mid) / denom) for i in range(steps)
    ]


# ----------------------------------------------------------------------
# Core structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


@dataclass(frozen=True)
class BurstSignal:
    key: str
    count: int
    z_score: float


@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()


def keep_candidate(candidate: Entity, selected: List[Entity], delta_m: float) -> bool:
    for existing in selected:
        same_kind = (
            signature(candidate) == signature(existing)
            or candidate.category.strip().lower()
            == existing.category.strip().lower()
        )
        if same_kind and haversine_m(
            (candidate.lat, candidate.lon), (existing.lat, existing.lon)
        ) <= delta_m:
            return False
    return True


def sessionize_events(events: List[dict], gap_seconds: float) -> List[List[dict]]:
    """Group chronologically sorted events; break when gap > gap_seconds."""
    if not events:
        return []
    events = sorted(events, key=lambda e: e["ts"])
    sessions = []
    cur = [events[0]]
    for prev, cur_evt in zip(events, events[1:]):
        if cur_evt["ts"] - prev["ts"] > gap_seconds:
            sessions.append(cur)
            cur = [cur_evt]
        else:
            cur.append(cur_evt)
    sessions.append(cur)
    return sessions


def detect_bursts(events: List[dict], key: str = "category") -> List[BurstSignal]:
    """Z‑score burst detection on simple count per key."""
    if not events:
        return []
    counts = Counter(e[key] for e in events)
    vals = np.array(list(counts.values()), dtype=float)
    mean = vals.mean()
    std = vals.std(ddof=0) if vals.size > 1 else 1.0
    bursts = []
    for k, c in counts.items():
        z = (c - mean) / std if std > 0 else 0.0
        if z > 1.5:  # arbitrary significance threshold
            bursts.append(BurstSignal(k, c, z))
    return bursts


def mine_temporal_motifs(
    sessions: List[List[dict]], key: str = "category", min_support: int = 2
) -> List[TemporalMotif]:
    """Extract length‑2 patterns and count their support across sessions."""
    pattern_counts = Counter()
    for sess in sessions:
        cats = [e[key] for e in sess]
        for i in range(len(cats) - 1):
            pat = (cats[i], cats[i + 1])
            pattern_counts[pat] += 1
    motifs = [
        TemporalMotif(pat, sup)
        for pat, sup in pattern_counts.items()
        if sup >= min_support
    ]
    return motifs


# ----------------------------------------------------------------------
# Hybrid structures and operations
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class HybridState:
    velocity: float
    cum_metric: float
    peak_velocity: float


def integrate_motif_strike(
    motif: TemporalMotif,
    geo_path: List[Tuple[float, float]],
    dt: float = 1.0,
    m_head: float = 1.0,
    drag_cd: float = 0.3,
    fluid_density: float = 1.0,
    area: float = 1.0,
) -> HybridState:
    """
    Treat the motif support as a constant “force” applied over the length of the
    geographic path.  Drag is proportional to the instantaneous haversine distance
    between successive points (quadratic in the virtual velocity).

    Parameters
    ----------
    motif : TemporalMotif
        Provides the force magnitude via ``support``.
    geo_path : list of (lat, lon)
        Ordered geographic coordinates that the virtual head traverses.
    dt : float
        Time step for integration.
    m_head, drag_cd, fluid_density, area : float
        Physical parameters (identical to Parent A).

    Returns
    -------
    HybridState
        Velocity, cumulative metric (analogue of travelled distance), and peak
        velocity reached during the integration.
    """
    if dt <= 0 or m_head <= 0:
        raise ValueError("dt and m_head must be positive")
    if len(geo_path) < 2:
        # No movement – return zeroed state
        return HybridState(0.0, 0.0, 0.0)

    force = float(motif.support)  # constant force for the whole path
    v = 0.0
    cum = 0.0
    peak = 0.0

    for p_prev, p_cur in zip(geo_path, geo_path[1:]):
        # Geographic distance in metres acts as a drag coefficient for this step
        d_geo = haversine_m(p_prev, p_cur)
        # Convert d_geo to a dimensionless drag factor (scale to typical drag magnitude)
        geo_drag = drag_cd * fluid_density * area * d_geo / (2.0 * m_head)

        # Quadratic drag term using current velocity
        drag = geo_drag * v * abs(v)
        acc = force / m_head - drag
        v = max(0.0, v + acc * dt)
        cum += v * dt
        peak = max(peak, v)

    return HybridState(v, cum, peak)


def hybrid_burst_score(
    entities: Iterable[Entity],
    events: List[dict],
    delta_m: float = 100.0,
    dt: float = 1.0,
) -> float:
    """
    End‑to‑end hybrid scoring:
    1. Filter entities spatially (Parent B logic).
    2. Detect bursts in the event stream.
    3. Mine temporal motifs from burst‑filtered sessions.
    4. For each motif, integrate a kinematic state over the geographic path of the
       filtered entities (using ``integrate_motif_strike``).
    5. Combine the cumulative metrics into a single dimensionless score.

    The returned score is the sum of ``cum_metric`` weighted by the burst z‑score.
    """
    # 1. Spatial filtering
    filtered = []
    ordered = list(entities)
    ordered.sort(key=lambda e: (-e.score, e.id))
    for e in ordered:
        if keep_candidate(e, filtered, delta_m):
            filtered.append(e)

    if not filtered:
        return 0.0

    # Build a simple geographic path (ordered by score)
    geo_path = [(e.lat, e.lon) for e in filtered]

    # 2. Burst detection
    bursts = detect_bursts(events, key="category")

    # 3. Sessionization & motif mining
    sessions = sessionize_events(events, gap_seconds=1800.0)
    motifs = mine_temporal_motifs(sessions, key="category", min_support=2)

    # 4. Integrate each motif over the same geographic path
    total = 0.0
    for motif in motifs:
        state = integrate_motif_strike(
            motif,
            geo_path,
            dt=dt,
            m_head=1.0,
            drag_cd=0.3,
            fluid_density=1.0,
            area=1.0,
        )
        # Find matching burst z‑score (if any)
        z = next((b.z_score for b in bursts if b.key == motif.pattern[0]), 1.0)
        total += state.cum_metric * z

    return total


def hybrid_filter_entities(
    entities: Iterable[Entity],
    events: List[dict],
    delta_m: float = 100.0,
    min_support: int = 2,
) -> List[Entity]:
    """
    Advanced filter that combines:
    • Spatial proximity (Parent B)
    • Temporal‑motif support (must exceed ``min_support``)
    • Kinematic viability (peak velocity from ``integrate_motif_strike`` above a
      small threshold).

    Returns the subset of entities that survive all three criteria.
    """
    # Spatial pre‑filter
    pre_filtered = []
    ordered = list(entities)
    ordered.sort(key=lambda e: (-e.score, e.id))
    for e in ordered:
        if keep_candidate(e, pre_filtered, delta_m):
            pre_filtered.append(e)

    if not pre_filtered:
        return []

    # Temporal motif mining
    sessions = sessionize_events(events, gap_seconds=1800.0)
    motifs = mine_temporal_motifs(sessions, min_support=min_support)

    # Keep only entities that belong to at least one high‑support motif
    # (We approximate by checking if the entity's category appears in any motif pattern)
    motif_categories = {cat for m in motifs for cat in m.pattern}
    candidates = [
        e for e in pre_filtered if e.category.strip().lower() in motif_categories
    ]

    # Final kinematic check
    geo_path = [(e.lat, e.lon) for e in candidates]
    viable = []
    for e in candidates:
        # Build a dummy motif with support = 1 for each entity to evaluate drag
        dummy_motif = TemporalMotif(pattern=(e.category,), support=1)
        state = integrate_motif_strike(
            dummy_motif,
            geo_path,
            dt=0.5,
            m_head=1.0,
            drag_cd=0.3,
            fluid_density=1.0,
            area=1.0,
        )
        if state.peak_velocity > 0.01:  # arbitrary tiny threshold
            viable.append(e)

    return viable


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic entities
    ents = [
        Entity(id="A", lat=40.0, lon=-74.0, category="alpha", score=5.0),
        Entity(id="B", lat=40.001, lon=-74.001, category="beta", score=4.5),
        Entity(id="C", lat=40.5, lon=-74.5, category="alpha", score=3.0),
        Entity(id="D", lat=41.0, lon=-75.0, category="gamma", score=2.0),
    ]

    # Synthetic events (timestamp in seconds since epoch)
    now = int(1_600_000_000)
    evts = [
        {"ts": now, "category": "alpha"},
        {"ts": now + 100, "category": "beta"},
        {"ts": now + 200, "category": "alpha"},
        {"ts": now + 4000, "category": "gamma"},
        {"ts": now + 4200, "category": "beta"},
        {"ts": now + 8000, "category": "alpha"},
    ]

    # Run hybrid score
    score = hybrid_burst_score(ents, evts, delta_m=150.0, dt=0.5)
    print(f"Hybrid burst score: {score:.4f}")

    # Run hybrid filter
    kept = hybrid_filter_entities(ents, evts, delta_m=150.0, min_support=2)
    print(f"Entities kept after hybrid filter: {[e.id for e in kept]}")