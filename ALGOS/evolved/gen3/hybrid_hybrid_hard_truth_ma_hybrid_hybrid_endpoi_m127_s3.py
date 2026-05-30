# DARWIN HAMMER — match 127, survivor 3
# gen: 3
# parent_a: hybrid_hard_truth_math_model_pool_m8_s2.py (gen1)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_krampus_brain_m24_s1.py (gen2)
# born: 2026-05-29T23:25:56Z

from __future__ import annotations

import datetime as dt
import hashlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import math
import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry / LSM utilities (simplified)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    """Lower‑case alphabetic tokens (apostrophe‑aware)."""
    return [w for w in text.lower().split() if w.isalpha()]


def stylometry_features(text: str) -> np.ndarray:
    """
    Produce a deterministic 96‑dimensional numeric representation of *text*.
    The implementation uses a SHA‑256 hash to seed a pseudo‑random generator,
    guaranteeing reproducibility without external corpora.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.random(96)


def lsm_vector(text: str) -> np.ndarray:
    """
    Compute the proportion of words belonging to each FUNCTION_CAT.
    Returns a |FUNCTION_CATS|‑dimensional vector ordered by dict insertion.
    """
    toks = words(text)
    total = len(toks) or 1
    vec = []
    for cat in FUNCTION_CATS.values():
        count = sum(1 for w in toks if w in cat)
        vec.append(count / total)
    return np.array(vec, dtype=float)


def extract_features(text: str) -> np.ndarray:
    """
    Concatenate stylometry (96) and LSM (|FUNCTION_CATS|) into a single
    feature vector **v** ∈ ℝⁿ.
    """
    return np.concatenate([stylometry_features(text), lsm_vector(text)])


# ----------------------------------------------------------------------
# Parent B – circuit breaker and brain‑map utilities
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

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


# ----------------------------------------------------------------------
# Hybrid mathematics
# ----------------------------------------------------------------------
def compatibility_score(v: np.ndarray, m: np.ndarray) -> float:
    """
    Bilinear compatibility s = vᵀ P m.
    P projects the first two components of v onto the two‑dimensional model space.
    """
    if v.shape[0] < 2:
        raise ValueError("Feature vector must have at least two dimensions.")
    # Projection matrix P ∈ ℝⁿˣ² : identity on the first two rows, zeros elsewhere
    P = np.zeros((v.shape[0], 2), dtype=float)
    P[0, 0] = 1.0
    P[1, 1] = 1.0
    return float(v.T @ P @ m)


def recovery_priority(cb: EndpointCircuitBreaker, mass: float, shape_factor: float) -> float:
    """
    Compute a reliability‑aware priority:
        r = (mass * shape_factor) / (failures + 1)
    Larger mass/shape → higher priority; more failures → lower priority.
    """
    return (mass * shape_factor) / (cb.failures + 1)


def curvature_factor(v: np.ndarray) -> float:
    """
    Derive a geometric curvature from the feature vector.
    Here we use the Euclidean norm of the first two stylometry dimensions.
    """
    return float(np.linalg.norm(v[:2]))


def hybrid_brainmap(s: float, r: float, c: float) -> np.ndarray:
    """
    Combine compatibility (s), recovery priority (r) and curvature (c) into a
    scaling factor that modulates a 2‑D identity brain‑map.
        factor = s * r * c
        brainmap = factor * I₂
    """
    factor = s * r * c
    return factor * np.eye(2)


# ----------------------------------------------------------------------
# Public API demonstrating the hybrid operation
# ----------------------------------------------------------------------
def model_resource_vector(ram_gb: float, tier: int) -> np.ndarray:
    """
    Encode model resources as a 2‑dimensional vector.
    RAM is scaled to [0, 1] assuming a maximum of 128 GB.
    Tier is taken as an integer (e.g. 0 = basic, 1 = standard, 2 = premium).
    """
    max_ram = 128.0
    ram_scaled = min(max(ram_gb / max_ram, 0.0), 1.0)
    return np.array([ram_scaled, float(tier)], dtype=float)


def select_model(
    text: str,
    ram_gb: float,
    tier: int,
    cb: EndpointCircuitBreaker,
    mass: float,
    shape_factor: float,
) -> Tuple[bool, np.ndarray, float]:
    """
    Full hybrid pipeline:

    1. Extract high‑dimensional text features **v**.
    2. Build model resource vector **m**.
    3. Compute compatibility scalar **s**.
    4. Derive reliability **r** and curvature **c**.
    5. Produce a modulated brain‑map matrix.
    6. Update the circuit breaker based on whether the compatibility exceeds
       a simple heuristic threshold (here 0.5 after normalisation).

    Returns a tuple ``(allowed, brainmap, s)`` where ``allowed`` reflects the
    circuit‑breaker state after the step.
    """
    v = extract_features(text)
    m = model_resource_vector(ram_gb, tier)
    s = compatibility_score(v, m)
    r = recovery_priority(cb, mass, shape_factor)
    c = curvature_factor(v)
    brainmap = hybrid_brainmap(s, r, c)
    
    # Update circuit breaker
    if s > 0.5:
        cb.record_success()
    else:
        cb.record_failure()

    return (cb.allow(), brainmap, s)


def advanced_select_model(
    text: str,
    ram_gb: float,
    tier: int,
    cb: EndpointCircuitBreaker,
    mass: float,
    shape_factor: float,
    threshold: float = 0.5,
) -> Tuple[bool, np.ndarray, float]:
    """
    Advanced hybrid pipeline with a tunable threshold.

    Returns a tuple ``(allowed, brainmap, s)`` where ``allowed`` reflects the
    circuit‑breaker state after the step.
    """
    v = extract_features(text)
    m = model_resource_vector(ram_gb, tier)
    s = compatibility_score(v, m)
    r = recovery_priority(cb, mass, shape_factor)
    c = curvature_factor(v)
    brainmap = hybrid_brainmap(s, r, c)
    
    # Update circuit breaker
    if s > threshold:
        cb.record_success()
    else:
        cb.record_failure()

    return (cb.allow(), brainmap, s)


def evaluate_model_selection(
    text: str,
    ram_gb: float,
    tier: int,
    cb: EndpointCircuitBreaker,
    mass: float,
    shape_factor: float,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Evaluate model selection with a given threshold.

    Returns a tuple of (s, r * c).
    """
    v = extract_features(text)
    m = model_resource_vector(ram_gb, tier)
    s = compatibility_score(v, m)
    r = recovery_priority(cb, mass, shape_factor)
    c = curvature_factor(v)
    
    return (s, r * c)