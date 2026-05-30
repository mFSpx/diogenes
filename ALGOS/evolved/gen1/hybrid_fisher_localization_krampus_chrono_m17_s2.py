# DARWIN HAMMER — match 17, survivor 2
# gen: 1
# parent_a: fisher_localization.py (gen0)
# parent_b: krampus_chrono.py (gen0)
# born: 2026-05-29T23:20:37Z

"""Hybrid algorithm merging Fisher‑information angular scoring (fisher_localization.py)
and chronological candidate extraction (krampus_chrono.py).

Mathematical bridge:
Both parents rank a set of candidates by a scalar “information” metric.
- In A the Fisher information for a Gaussian beam is I(θ)= (∂_θ I)^2 / I,
  where I(θ)=exp(−½((θ−c)/w)^2).
- In B each timestamp candidate carries a *priority* and *confidence* that
  can be interpreted as a weight w_t.  Treating the timestamp t as a noisy
  measurement of an underlying true time μ with variance σ², the Fisher
  information for a Gaussian time model is I(t)=((t−μ)²/σ⁴)/p(t) where
  p(t) is the Gaussian pdf.

The hybrid score simply normalises the angular Fisher score and the temporal
Fisher score to [0,1] and combines them linearly with configurable mixing
coefficients α,β (α+β=1).  This yields a single scalar that can be maximised
over mixed angle‑time candidate tuples."""


import math
import re
import datetime as dt
import random
import sys
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

import numpy as np


# ----------------------------------------------------------------------
# Parent A – Gaussian beam & Fisher score
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ)."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for the Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Parent B – Loose datetime parsing & candidate extraction
# ----------------------------------------------------------------------
MONTH_NAME_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+"
    r"([0-3]?\d)(?:st|nd|rd|th)?,?\s+((?:20|19)\d{2})\b",
    re.I,
)

ISO_INLINE_RE = re.compile(
    r"\b((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}"
    r"(?:[T\s_]\d{1,2}:?\d{2}(?::?\d{2})?(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)?)\b"
)

COMPACT_RE = re.compile(
    r"\b((?:20|19)\d{2})(\d{2})(\d{2})"
    r"(?:[T_-]?(\d{2})(\d{2})(\d{2})?)?Z?\b"
)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_loose_datetime(raw: str) -> Optional[dt.datetime]:
    """Parse a wide variety of human‑written timestamps into UTC."""
    text = _normalize_ws(raw).strip("'\"`[]()")
    if not text:
        return None

    # Month name format
    m = MONTH_NAME_RE.search(text)
    if m:
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        try:
            month = month_map[m.group(1)[:3].lower()]
            day = int(m.group(2))
            year = int(m.group(3))
            return dt.datetime(year, month, day, tzinfo=dt.timezone.utc)
        except Exception:
            return None

    # Normalise separators
    cleaned = text.replace("_", "-").replace("/", "-")
    # Insert T between date and compact time fragments
    cleaned = re.sub(
        r"^((?:20|19)\d{2}-\d{1,2}-\d{1,2})-(\d{2})(\d{2})(\d{2})$",
        r"\1T\2:\3:\4",
        cleaned,
    )
    cleaned = re.sub(
        r"^((?:20|19)\d{2}-\d{1,2}-\d{1,2})-(\d{2})(\d{2})$",
        r"\1T\2:\3",
        cleaned,
    )
    cleaned = cleaned.replace(" ", "T")

    # Final ISO‑like attempt
    try:
        dt_obj = dt.datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
        return dt_obj.astimezone(dt.timezone.utc)
    except Exception:
        return None


def chrono_candidates_for_path(path: Path) -> List[Dict[str, Any]]:
    """Extract deterministic datetime candidates from a file‑system path."""
    candidates: List[Dict[str, Any]] = []
    text = str(path)

    def _add(source: str, raw: str, priority: int, confidence: int) -> None:
        dt_obj = parse_loose_datetime(raw)
        if dt_obj:
            candidates.append(
                {
                    "timestamp": dt_obj,
                    "source": source,
                    "raw": raw,
                    "priority": priority,
                    "confidence_bps": confidence,
                }
            )

    # ISO‑inline fragments inside the path
    for m in ISO_INLINE_RE.finditer(text):
        raw = m.group(1)
        has_time = bool(re.search(r"[T\s_-]\d{1,2}:?\d{2}", raw))
        _add("path_iso_inline", raw, 20 if has_time else 32, 8800 if has_time else 8200)

    # Compact yyyymmdd[hhmmss] style
    for m in COMPACT_RE.finditer(text):
        y, mo, day, hh, mm, ss = m.groups()
        raw = f"{y}-{mo}-{day}"
        if hh and mm:
            raw += f"T{hh}:{mm}:{ss or '00'}"
        _add("path_compact", raw, 22 if hh and mm else 33, 8500 if hh and mm else 8000)

    # Fallback: use file's birth time if available (simulated here)
    try:
        st = path.stat()
        birth_ts = getattr(st, "st_birthtime", None)
        if birth_ts:
            dt_obj = dt.datetime.fromtimestamp(birth_ts, tz=dt.timezone.utc)
            candidates.append(
                {
                    "timestamp": dt_obj,
                    "source": "os_birthtime",
                    "raw": str(birth_ts),
                    "priority": 80,
                    "confidence_bps": 6000,
                }
            )
    except Exception:
        pass

    return candidates


# ----------------------------------------------------------------------
# Hybrid layer – unified information metric
# ----------------------------------------------------------------------
def temporal_fisher(t: dt.datetime, mu: dt.datetime, sigma: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian time measurement.
    t, mu are datetimes; sigma is standard deviation in seconds.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    dt_sec = (t - mu).total_seconds()
    pdf = (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * (dt_sec / sigma) ** 2)
    pdf = max(pdf, eps)
    derivative = -dt_sec / (sigma ** 2) * pdf
    return (derivative * derivative) / pdf


def hybrid_score(
    angle: Optional[float],
    timestamp: Optional[dt.datetime],
    *,
    angle_center: float,
    angle_width: float,
    time_mu: dt.datetime,
    time_sigma: float,
    alpha: float = 0.5,
) -> float:
    """
    Combined information score.
    - angle: optional angular measurement.
    - timestamp: optional datetime measurement.
    - alpha: weight for angular Fisher (0..1).  Temporal weight = 1‑alpha.
    Returns a normalised scalar in [0, 1].
    """
    # Angular part
    if angle is not None:
        raw_ang = fisher_score(angle, angle_center, angle_width)
        # Normalise by a heuristic maximum (when derivative is maximal)
        max_ang = fisher_score(angle_center + angle_width, angle_center, angle_width)
        norm_ang = raw_ang / max_ang if max_ang else 0.0
    else:
        norm_ang = 0.0

    # Temporal part
    if timestamp is not None:
        raw_time = temporal_fisher(timestamp, time_mu, time_sigma)
        # Normalise by the Fisher information at one sigma deviation
        max_time = temporal_fisher(time_mu + dt.timedelta(seconds=time_sigma), time_mu, time_sigma)
        norm_time = raw_time / max_time if max_time else 0.0
    else:
        norm_time = 0.0

    # Linear blend
    return alpha * norm_ang + (1.0 - alpha) * norm_time


def best_hybrid_candidate(
    angle_candidates: List[float],
    time_candidates: List[Dict[str, Any]],
    *,
    angle_center: float,
    angle_width: float,
    time_mu: dt.datetime,
    time_sigma: float,
    alpha: float = 0.5,
) -> Tuple[float, dt.datetime, float]:
    """
    Exhaustively search the Cartesian product of angle and time candidates,
    returning the tuple (angle, timestamp, score) with the highest hybrid_score.
    """
    best = (None, None, -math.inf)
    for ang in angle_candidates:
        for tc in time_candidates:
            ts = tc["timestamp"]
            sc = hybrid_score(
                ang,
                ts,
                angle_center=angle_center,
                angle_width=angle_width,
                time_mu=time_mu,
                time_sigma=time_sigma,
                alpha=alpha,
            )
            if sc > best[2]:
                best = (ang, ts, sc)
    if best[0] is None or best[1] is None:
        raise RuntimeError("No candidates supplied")
    return best  # type: ignore


def hybrid_grid_search(
    angle_range: Tuple[float, float],
    n_angle: int,
    path: Path,
    *,
    angle_center: float,
    angle_width: float,
    time_mu: dt.datetime,
    time_sigma: float,
    alpha: float = 0.5,
) -> Tuple[float, dt.datetime, float]:
    """
    Convenience wrapper that builds a uniform angular grid, extracts temporal
    candidates from *path*, and returns the optimal hybrid tuple.
    """
    angles = np.linspace(angle_range[0], angle_range[1], n_angle).tolist()
    time_cands = chrono_candidates_for_path(path)
    if not time_cands:
        raise RuntimeError("No temporal candidates found")
    return best_hybrid_candidate(
        angles,
        time_cands,
        angle_center=angle_center,
        angle_width=angle_width,
        time_mu=time_mu,
        time_sigma=time_sigma,
        alpha=alpha,
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a temporary dummy path containing several date patterns
    dummy_path = Path("2023-07-15_report_20230801T123045.txt")
    # Parameters for the angular model
    ANGLE_CENTER = 0.0
    ANGLE_WIDTH = 0.2

    # Temporal model parameters (assume we expect events around now)
    TIME_MU = dt.datetime.now(tz=dt.timezone.utc)
    TIME_SIGMA = 48 * 3600.0  # two days in seconds

    # Run hybrid search
    try:
        angle_opt, ts_opt, score_opt = hybrid_grid_search(
            angle_range=(-1.0, 1.0),
            n_angle=41,
            path=dummy_path,
            angle_center=ANGLE_CENTER,
            angle_width=ANGLE_WIDTH,
            time_mu=TIME_MU,
            time_sigma=TIME_SIGMA,
            alpha=0.6,
        )
        print(f"Best angle: {angle_opt:.4f} rad")
        print(f"Best timestamp: {ts_opt.isoformat()}")
        print(f"Hybrid score: {score_opt:.4f}")
    except Exception as exc:
        print(f"Hybrid test failed: {exc}", file=sys.stderr)
        sys.exit(1)