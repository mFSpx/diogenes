# DARWIN HAMMER — match 1711, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_temporal_moti_hybrid_doomsday_cale_m218_s1.py (gen2)
# parent_b: hybrid_hybrid_xgboost_objec_hybrid_hybrid_regret_m283_s1.py (gen4)
# born: 2026-05-29T23:38:23Z

"""Hybrid Temporal Motif‑Gini & Regret‑Weighted XGBoost Lens

Parents
-------
* **Parent A** – spatial‑temporal motif mining with weekday Gini
  (`hybrid_hybrid_temporal_moti_hybrid_doomsday_cale_m218_s1.py`).
* **Parent B** – regret‑weighted ternary lens audit and binary‑logistic
  gradient/Hessian computation (`hybrid_hybrid_xgboost_objec_hybrid_hybrid_regret_m283_s1.py`).

Mathematical Bridge
-------------------
For every temporal motif *m* we obtain

* raw support `S(m)`  – number of sessions containing the motif,
* weekday count vector `c(m) ∈ ℕ⁷` via the Doomsday algorithm,
* Gini coefficient `G(m) = gini_coefficient(c(m))`.

The **motif‑quality** is `Q(m) = S(m)·(1‑G(m))`.  
`Q(m)` is interpreted as a pseudo‑label for a binary‑logistic XGBoost
objective.  The gradient/hessian pair `(g, h)` is computed with the
standard binary‑logistic formulas.  To inject the regret‑aware
information from Parent B we generate a **regret‑weighted ternary vector**
`r(m)` (deterministic pseudo‑random ints in `{-1,0,1}`) and weight the
gradient component:


ĝ = g * exp( - r · Q )
ĥ = h                     # Hessian is left unchanged (still convex)


Thus the hybrid system fuses the motif‑centric statistics of Parent A with
the regret‑weighted optimisation machinery of Parent B in a single,
vector‑ised pipeline.

The module provides three public functions that showcase this hybrid
behaviour:

1. `sessionize_events` – turn a timestamped event stream into sessions.
2. `hybrid_motif_gini_score` – compute `Q(m)` for a motif.
3. `regret_weighted_gradients` – obtain regret‑weighted gradient/hessian
   pairs ready for an XGBoost‑style update.

Only the Python standard library and NumPy are used.
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
# Shared utilities (from Parent A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """Simple entity used for spatial‑temporal examples."""
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Return Haversine distance in metres between two (lat, lon) points."""
    R = 6371000.0  # Earth radius in metres
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def doomsday_numpy(dates: np.ndarray) -> np.ndarray:
    """
    Vectorised Doomsday weekday calculation.
    Returns an array of integers 0‑6 where 0 = Monday.
    """
    # Zeller's congruence (adjusted for Monday=0)
    y = dates.astype('datetime64[Y]').astype(int) + 1970
    m = dates.astype('datetime64[M]').astype(int) % 12 + 1
    d = dates.astype('datetime64[D]').astype(int) % 31 + 1
    # Adjust months Jan/Feb to be month 13/14 of previous year
    mask = m < 3
    y_adj = np.where(mask, y - 1, y)
    m_adj = np.where(mask, m + 12, m)
    K = y_adj % 100
    J = y_adj // 100
    h = (d + (13 * (m_adj + 1)) // 5 + K + K // 4 + J // 4 + 5 * J) % 7
    # Zeller: 0=Saturday, 1=Sunday, … 5=Thursday, 6=Friday
    # Convert to Monday=0 … Sunday=6
    return (h + 5) % 7


def gini_coefficient_numpy(x: np.ndarray) -> float:
    """Gini coefficient for a 1‑D non‑negative array."""
    if x.size == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = x.size
    cumulative = np.cumsum(sorted_x, dtype=float)
    sum_x = cumulative[-1]
    if sum_x == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_x) / n
    return float(gini)


def sessionize_events(
    events: List[Tuple[dt.datetime, Any]],
    max_gap_seconds: int = 1800,
) -> List[List[Tuple[dt.datetime, Any]]]:
    """
    Group chronologically ordered events into sessions.
    A new session starts when the gap between consecutive timestamps exceeds
    `max_gap_seconds`.
    """
    if not events:
        return []
    sessions: List[List[Tuple[dt.datetime, Any]]] = []
    current: List[Tuple[dt.datetime, Any]] = [events[0]]
    for ts, payload in events[1:]:
        prev_ts = current[-1][0]
        if (ts - prev_ts).total_seconds() > max_gap_seconds:
            sessions.append(current)
            current = [(ts, payload)]
        else:
            current.append((ts, payload))
    sessions.append(current)
    return sessions


def mine_temporal_motifs(
    sessions: List[List[Tuple[dt.datetime, str]]],
    min_support: int = 2,
) -> List[Dict[str, Any]]:
    """
    Very lightweight motif miner.
    Returns a list of motif dictionaries:
        {
            "pattern": tuple[str, ...],
            "support": int,
            "timestamps": List[dt.datetime]   # all timestamps where the motif appears
        }
    For demonstration we only consider length‑2 consecutive category pairs.
    """
    pattern_counts: Dict[Tuple[str, str], List[dt.datetime]] = {}
    for sess in sessions:
        cats = [cat for _, cat in sess]
        times = [ts for ts, _ in sess]
        for i in range(len(cats) - 1):
            pat = (cats[i], cats[i + 1])
            pattern_counts.setdefault(pat, []).append(times[i + 1])
    motifs = []
    for pat, tlist in pattern_counts.items():
        if len(tlist) >= min_support:
            motifs.append({"pattern": pat, "support": len(tlist), "timestamps": tlist})
    return motifs


def hybrid_motif_gini_score(motif: Dict[str, Any]) -> float:
    """
    Compute Q = S * (1 - G) for a motif.
    - S : raw support (number of occurrences)
    - G : Gini coefficient of the weekday distribution of its timestamps
    """
    timestamps = np.array(motif["timestamps"], dtype="datetime64[ns]")
    weekdays = doomsday_numpy(timestamps.astype("datetime64[D]"))
    counts = np.bincount(weekdays, minlength=7)
    gini = gini_coefficient_numpy(counts)
    support = motif["support"]
    return support * (1.0 - gini)


# ----------------------------------------------------------------------
# Regret‑weighted utilities (from Parent B)
# ----------------------------------------------------------------------


def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-margin))


def binary_logistic_grad_hess(
    y_true: np.ndarray, margin: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard binary‑logistic gradient and Hessian.
    y_true ∈ {0,1}
    """
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h


def regret_weighted_ternary_vector(token: str, seed: int, k: int = 128) -> List[int]:
    """
    Deterministic pseudo‑random ternary vector in {-1,0,1}.
    The vector length `k` is fixed for all tokens.
    """
    rng = random.Random(hash(token) ^ seed)
    vec = []
    for _ in range(k):
        r = rng.random()
        if r < 0.33:
            vec.append(-1)
        elif r < 0.66:
            vec.append(0)
        else:
            vec.append(1)
    return vec


def regret_weighted_gradients(
    motif_scores: List[float],
    token: str,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each motif quality `q` we treat it as a pseudo‑label `y_true`
    (scaled to [0,1]) and compute binary‑logistic gradients.
    The gradients are then regret‑weighted using a ternary vector.
    Returns the aggregated (weighted) gradient and Hessian arrays.
    """
    if not motif_scores:
        return np.array([]), np.array([])

    # Scale motif scores to [0,1] for logistic loss
    max_q = max(motif_scores)
    y = np.array([q / max_q if max_q > 0 else 0.0 for q in motif_scores])

    # Dummy margin – in a real XGBoost run this would be the model output.
    # Here we initialise with zeros.
    margin = np.zeros_like(y)

    g, h = binary_logistic_grad_hess(y, margin)

    # Regret‑weighted ternary vector
    r_vec = np.array(regret_weighted_ternary_vector(token, seed, k=len(g)), dtype=float)

    # Weight each gradient component by exp(-r_i * q_i)
    weighted_g = g * np.exp(-r_vec * y)

    # Aggregate (mean) to obtain a single update direction
    return weighted_g.mean().reshape(1), h.mean().reshape(1)


# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------


def hybrid_pipeline(
    raw_events: List[Tuple[dt.datetime, str]],
    token: str,
    max_gap_seconds: int = 1800,
    min_support: int = 2,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    End‑to‑end demo:
        1. Sessionise raw events.
        2. Mine temporal motifs.
        3. Compute Q‑scores (support × (1‑Gini)).
        4. Produce regret‑weighted gradient/Hessian for the set of Q‑scores.

    Returns
    -------
    total_q : float
        Sum of all Q‑scores (a simple aggregate quality metric).
    grad : np.ndarray
        Weighted gradient (shape (1,)).
    hess : np.ndarray
        Weighted Hessian (shape (1,)).
    """
    # 1. Sessionisation
    events_sorted = sorted(raw_events, key=lambda x: x[0])
    sessions = sessionize_events(events_sorted, max_gap_seconds)

    # 2. Motif mining
    motifs = mine_temporal_motifs(sessions, min_support)

    # 3. Quality scores
    q_scores = [hybrid_motif_gini_score(m) for m in motifs]
    total_q = float(sum(q_scores))

    # 4. Regret‑weighted gradient/Hessian
    grad, hess = regret_weighted_gradients(q_scores, token)

    return total_q, grad, hess


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic timestamped categorical events
    now = dt.datetime.utcnow()
    categories = ["A", "B", "C", "D"]
    rng = random.Random(0)

    synthetic_events: List[Tuple[dt.datetime, str]] = []
    for i in range(200):
        # Random walk in time (average 5 min between events)
        delta = dt.timedelta(seconds=rng.expovariate(1 / 300))
        now += delta
        cat = rng.choice(categories)
        synthetic_events.append((now, cat))

    token = "example_token"

    total_quality, gradient, hessian = hybrid_pipeline(
        synthetic_events,
        token,
        max_gap_seconds=1800,
        min_support=3,
    )

    print(f"Total hybrid quality (ΣQ): {total_quality:.3f}")
    print(f"Regret‑weighted gradient: {gradient}")
    print(f"Regret‑weighted hessian: {hessian}")