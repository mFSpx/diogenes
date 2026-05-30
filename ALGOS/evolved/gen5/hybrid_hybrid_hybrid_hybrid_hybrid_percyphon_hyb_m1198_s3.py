# DARWIN HAMMER — match 1198, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m264_s2.py (gen4)
# parent_b: hybrid_percyphon_hybrid_endpoint_circ_m45_s1.py (gen2)
# born: 2026-05-29T23:33:34Z

import sys
import math
import random
import pathlib
import hashlib
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from typing import Any, Tuple, List

import numpy as np
import re

# ----------------------------------------------------------------------
# Regex feature extraction (Parent A core)
# ----------------------------------------------------------------------
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

REGEX_PATTERNS = [
    ("evidence", EVIDENCE_RE),
    ("planning", PLANNING_RE),
    ("delay", DELAY_RE),
    ("support", SUPPORT_RE),
    ("boundary", BOUNDARY_RE),
]

def extract_regex_features(text: str) -> np.ndarray:
    """Return a unit‑length count vector for the defined regex categories."""
    counts = np.array([len(pat.findall(text)) for _, pat in REGEX_PATTERNS], dtype=float)
    norm = np.linalg.norm(counts) + 1e-12
    return counts / norm


# ----------------------------------------------------------------------
# Morphology and metrics (Parent B core)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float = 1.0  # default mass if not supplied

def sphericity_index(morph: Morphology) -> float:
    """Classic sphericity based on dimensions."""
    l, w, h = morph.length, morph.width, morph.height
    if min(l, w, h) <= 0:
        return 0.0
    volume = l * w * h
    surface = 2 * (l * w + l * h + w * h)
    return (math.pi ** (1.0 / 3.0) * (6 * volume) ** (2.0 / 3.0)) / surface

def flatness_index(morph: Morphology) -> float:
    """Flatness defined as the ratio of the smallest to the largest dimension."""
    dims = [morph.length, morph.width, morph.height]
    if max(dims) == 0:
        return 0.0
    return min(dims) / max(dims)


# ----------------------------------------------------------------------
# EndpointCircuitBreaker (Parent B core) – now with exponential back‑off and auto‑reset
# ----------------------------------------------------------------------
def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

class EndpointCircuitBreaker:
    """Circuit breaker that opens after `failure_threshold` consecutive failures,
    stays open for an exponentially increasing cooldown, and auto‑resets after the
    cooldown expires."""
    def __init__(self, failure_threshold: int = 3, base_cooldown: float = 5.0):
        self.failure_threshold = max(1, failure_threshold)
        self.base_cooldown = max(0.1, base_cooldown)  # seconds
        self.failures = 0
        self.open = False
        self.last_event_at = ""
        self._open_since: datetime | None = None
        self._cooldown_factor = 1

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self._cooldown_factor = 1
        self._open_since = None
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.last_event_at = now_z()
        if self.failures >= self.failure_threshold:
            self.open = True
            self._open_since = self._now()
            self._cooldown_factor = min(2 ** (self.failures - self.failure_threshold), 32)

    def allow(self) -> bool:
        if not self.open:
            return True
        # compute elapsed time since opening
        elapsed = (self._now() - self._open_since).total_seconds()
        if elapsed >= self.base_cooldown * self._cooldown_factor:
            # auto‑reset
            self.record_success()
            return True
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
            "cooldown_factor": self._cooldown_factor,
        }


# ----------------------------------------------------------------------
# RBF Surrogate (Parent B core)
# ----------------------------------------------------------------------
def rbf_phi(x: np.ndarray, c: np.ndarray, gamma: float) -> float:
    diff = x - c
    return math.exp(-gamma * np.dot(diff, diff))

def rbf_output(x: np.ndarray, centers: np.ndarray, weights: np.ndarray, gamma: float) -> float:
    phi_vals = np.fromiter((rbf_phi(x, c, gamma) for c in centers), dtype=float, count=centers.shape[0])
    return float(np.dot(weights, phi_vals))


# ----------------------------------------------------------------------
# Hybrid System – deeper integration and robust adaptation
# ----------------------------------------------------------------------
@dataclass
class HybridSystem:
    """Encapsulates the hybrid state: RBF parameters, a circuit breaker,
    and the immutable reference centers used for stable adaptation."""
    input_dim: int
    n_centers: int = 5
    gamma: float = 0.5
    alpha: float = 0.2   # sphericity influence on centers
    beta: float = 0.3    # flatness influence on weights
    centers: np.ndarray = field(init=False)          # mutable working centers
    _base_centers: np.ndarray = field(init=False)    # immutable reference
    weights: np.ndarray = field(init=False)
    breaker: EndpointCircuitBreaker = field(default_factory=EndpointCircuitBreaker)

    def __post_init__(self) -> None:
        rng = np.random.default_rng()
        self._base_centers = rng.normal(size=(self.n_centers, self.input_dim))
        self.centers = self._base_centers.copy()
        self.weights = rng.uniform(-1.0, 1.0, size=self.n_centers)

    def _scale_centers(self, sigma: float) -> None:
        """Scale each center radially based on sphericity without cumulative drift."""
        scale = 1.0 + self.alpha * sigma
        self.centers = self._base_centers * scale

    def _adjust_weights(self, flatness: float) -> None:
        """Modulate weights by flatness – more flat objects bias towards lower magnitude."""
        factor = 1.0 - self.beta * (1.0 - flatness)  # flatness close to 1 → factor ~1
        self.weights = self.weights * factor

    def step(self, text: str, morph: Morphology) -> float:
        """Perform a single hybrid update.

        Returns the RBF output if the circuit breaker permits; otherwise returns the
        previously cached output.
        """
        if not hasattr(self, "_last_output"):
            self._last_output = 0.0

        if not self.breaker.allow():
            return self._last_output

        # 1️⃣ Feature extraction
        x = extract_regex_features(text)

        # Guard against dimension mismatch (should never happen, but safe)
        if x.shape[0] != self.input_dim:
            raise ValueError(f"Feature vector dimension {x.shape[0]} does not match expected {self.input_dim}")

        # 2️⃣ Morphology metrics
        sigma = sphericity_index(morph)
        flat = flatness_index(morph)

        # 3️⃣ Adapt RBF parameters (non‑cumulative)
        self._scale_centers(sigma)
        self._adjust_weights(flat)

        # 4️⃣ Compute surrogate output
        y = rbf_output(x, self.centers, self.weights, self.gamma)

        # 5️⃣ Success/failure logic – use a moving‑average based threshold
        # Maintain a simple exponential moving average of absolute outputs
        if not hasattr(self, "_ema_abs"):
            self._ema_abs = abs(y)
        else:
            self._ema_abs = 0.9 * self._ema_abs + 0.1 * abs(y)

        # Consider a "success" when the current absolute output exceeds 80 % of the EMA
        if abs(y) >= 0.8 * self._ema_abs:
            self.breaker.record_success()
        else:
            self.breaker.record_failure()

        self._last_output = y
        return y


# ----------------------------------------------------------------------
# Demonstration utilities (kept minimal for testing)
# ----------------------------------------------------------------------
def demo_feature_extraction() -> None:
    sample = "Please verify the source and provide a screenshot of the log."
    print("Feature vector:", extract_regex_features(sample))

def demo_hybrid_step() -> None:
    hs = HybridSystem(input_dim=len(REGEX_PATTERNS))
    morph = Morphology(length=2.0, width=1.5, height=0.5)
    texts = [
        "Please verify the source and provide a screenshot of the log.",
        "We should plan the next steps after a short pause.",
        "I need support from a friend and a lawyer.",
    ]
    for txt in texts:
        out = hs.step(txt, morph)
        print(f"Input: {txt[:40]:<45} → Output: {out:.4f}  |  Breaker open: {not hs.breaker.allow()}")

if __name__ == "__main__":
    demo_feature_extraction()
    demo_hybrid_step()