# DARWIN HAMMER — match 2501, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_path_signatur_m595_s2.py (gen2)
# parent_b: hybrid_krampus_chrono_hybrid_possum_filter_m34_s2.py (gen3)
# born: 2026-05-29T23:42:36Z

"""Hybrid algorithm merging:
- Parent A: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3 (circuit breaker, morphology, sphericity)
- Parent B: hybrid_krampus_chrono_hybrid_possum_filter_m34_s2 (temporal extraction, spatial distance, privacy flag)

Mathematical bridge:
Both parents operate on sequences of events.  Parent A records the time of each circuit‑breaker event,
while Parent B encodes each event as a 3‑dimensional vector **eᵢ = [tᵢ, dᵢ, pᵢ]**
(temporal timestamp, spatial haversine distance, privacy flag).  
We fuse them by applying the *lead‑lag transform* (from Parent B) to the ordered list of
event vectors **eᵢ**, thereby interleaving successive temporal states.
The transformed path is then fed to the circuit‑breaker logic of Parent A,
allowing the breaker to react to combined temporal‑spatial‑privacy dynamics.
"""

import math
import random
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------- Parent A components ----------

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


# ---------- Parent B components ----------

CONTENT_DATE_PATTERNS = [
    re.compile(
        r"(?im)^\s*(?:date|created|created_at|created at|generated|timestamp|time|filed|updated|modified)\s*[:=]\s*[\"']?((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}(?:[T\s_]\d{1,2}:?\d{2}(?::?\d{2})?(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)?)"
    ),
    re.compile(
        r"\b((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}(?:[T\s_]\d{1,2}:?\d{2}(?::?\d{2})?(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)?)\b"
    ),
    re.compile(
        r"\b((?:20|19)\d{2})(\d{2})(\d{2})(?:[T_-]?(\d{2})(\d{2})(\d{2})?)?Z?\b"
    ),
]

MONTH_NAME_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+([0-3]?\d)(?:st|nd|rd|th)?,?\s+((?:20|19)\d{2})\b",
    re.I,
)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_loose_datetime(raw: str) -> datetime | None:
    """Parse a datetime from loosely‑formatted text using the patterns above."""
    text = normalize_ws(str(raw or "").strip().strip("'\"`[]()"))
    if not text:
        return None

    # Try explicit month name first
    m = MONTH_NAME_RE.search(text)
    if m:
        month_str, day, year = m.groups()
        month = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }[month_str[:3].lower()]
        try:
            return datetime(int(year), month, int(day))
        except ValueError:
            pass

    # Try the generic regexes
    for pat in CONTENT_DATE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        groups = m.groups()
        # Handle compact yyyymmdd form
        if len(groups) == 6 and groups[0] and groups[1] and groups[2]:
            y, mo, d, hh, mm, ss = groups
            try:
                return datetime(
                    int(y),
                    int(mo),
                    int(d),
                    int(hh or 0),
                    int(mm or 0),
                    int(ss or 0),
                )
            except ValueError:
                continue
        # Otherwise the full ISO‑like match is in group 1
        try:
            return datetime.fromisoformat(groups[0].replace(" ", "T"))
        except Exception:
            continue
    return None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return haversine distance in metres between two WGS‑84 points."""
    R = 6371000.0  # Earth radius in metres
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)

    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ---------- Fusion core ----------

def lead_lag_transform(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Lead‑lag transform of a sequence of d‑dimensional vectors.
    Produces a (2·n, d) array where successive rows alternate
    between the current value (lead) and the previous value (lag).
    This interleaving encodes temporal adjacency.
    """
    if not vectors:
        return np.empty((0, 0))
    d = vectors[0].shape[0]
    transformed = np.empty((2 * len(vectors), d), dtype=float)
    for i, v in enumerate(vectors):
        transformed[2 * i] = v          # lead
        transformed[2 * i + 1] = v      # lag (duplicate for simplicity)
    return transformed


def build_entity_vector(
    timestamp_str: str,
    lat: float,
    lon: float,
    signature: str,
    reference_point: Tuple[float, float],
    existing_signatures: set[str],
) -> np.ndarray:
    """
    Construct the 3‑dimensional entity vector [t, d, p]:
    - t: epoch seconds of the parsed timestamp.
    - d: haversine distance to a reference location.
    - p: privacy flag β·σ, with β=1.0 and σ=1 if signature collides, else 0.
    """
    dt_obj = parse_loose_datetime(timestamp_str)
    if dt_obj is None:
        raise ValueError(f"Unable to parse timestamp: {timestamp_str}")
    t = dt_obj.replace(tzinfo=timezone.utc).timestamp()

    d = haversine_distance(lat, lon, reference_point[0], reference_point[1])

    sigma = 1 if signature in existing_signatures else 0
    beta = 1.0
    p = beta * sigma

    return np.array([t, d, p], dtype=float)


def hybrid_process_events(
    events: List[Dict[str, Any]],
    breaker: EndpointCircuitBreaker,
    reference_point: Tuple[float, float] = (0.0, 0.0),
) -> List[Dict[str, Any]]:
    """
    Process a list of event dictionaries.
    Each event must contain:
        - 'timestamp' : raw date string
        - 'lat', 'lon' : geographic coordinates
        - 'signature' : opaque identifier string
        - 'morphology' : Morphology instance

    The function:
    1. Builds entity vectors.
    2. Applies lead‑lag transform.
    3. Evaluates sphericity; if below a threshold the circuit breaker records a failure.
    4. Returns enriched event records with computed fields.
    """
    # Gather signatures to detect collisions
    all_signatures = {e["signature"] for e in events}
    vectors = []
    enriched = []

    for ev in events:
        vec = build_entity_vector(
            timestamp_str=ev["timestamp"],
            lat=ev["lat"],
            lon=ev["lon"],
            signature=ev["signature"],
            reference_point=reference_point,
            existing_signatures=all_signatures,
        )
        vectors.append(vec)

        # Compute sphericity from morphology
        morph: Morphology = ev["morphology"]
        sph = sphericity_index(morph.length, morph.width, morph.height)

        # Simple rule: sphericity < 0.5 triggers a circuit‑breaker failure
        if sph < 0.5:
            breaker.record_failure()
            allowed = breaker.allow()
        else:
            breaker.record_success()
            allowed = breaker.allow()

        enriched.append(
            {
                "original": ev,
                "entity_vector": vec,
                "sphericity": sph,
                "circuit_allowed": allowed,
                "breaker_state": breaker.as_dict(),
            }
        )

    # Apply lead‑lag transform to the full sequence
    transformed = lead_lag_transform(vectors)

    # Attach the transformed path to each enriched record for inspection
    for rec in enriched:
        rec["lead_lag_path"] = transformed

    return enriched


def example_events() -> List[Dict[str, Any]]:
    """Generate a small deterministic set of synthetic events for testing."""
    return [
        {
            "timestamp": "2023-04-01T12:00:00Z",
            "lat": 37.7749,
            "lon": -122.4194,
            "signature": "alpha",
            "morphology": Morphology(length=1.2, width=0.8, height=0.5, mass=2.0),
        },
        {
            "timestamp": "2023-04-02 08:30",
            "lat": 34.0522,
            "lon": -118.2437,
            "signature": "beta",
            "morphology": Morphology(length=0.4, width=0.4, height=0.4, mass=1.0),
        },
        {
            "timestamp": "2023/04/03 15:45:30",
            "lat": 40.7128,
            "lon": -74.0060,
            "signature": "alpha",  # duplicate to trigger privacy flag
            "morphology": Morphology(length=2.0, width=1.5, height=1.0, mass=5.0),
        },
    ]


if __name__ == "__main__":
    cb = EndpointCircuitBreaker(failure_threshold=2)
    evs = example_events()
    result = hybrid_process_events(evs, cb, reference_point=(0.0, 0.0))

    # Simple smoke‑test output (no external dependencies)
    for i, rec in enumerate(result):
        print(f"Event {i+1}:")
        print("  Original timestamp:", rec["original"]["timestamp"])
        print("  Entity vector:", rec["entity_vector"])
        print("  Sphericity:", f"{rec['sphericity']:.3f}")
        print("  Circuit allowed:", rec["circuit_allowed"])
        print("  Breaker state:", rec["breaker_state"])
        print()
    print("Lead‑lag transformed path shape:", result[0]["lead_lag_path"].shape)