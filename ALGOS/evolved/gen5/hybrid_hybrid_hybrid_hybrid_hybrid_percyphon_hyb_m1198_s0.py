# DARWIN HAMMER — match 1198, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m264_s2.py (gen4)
# parent_b: hybrid_percyphon_hybrid_endpoint_circ_m45_s1.py (gen2)
# born: 2026-05-29T23:33:34Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_liquid_m39_s0 (regex feature extraction & LTC concepts)
- Parent B: hybrid_percyphon_hybrid_endpoint_circ_m45_s1 (Morphology‑driven RBF surrogate & EndpointCircuitBreaker)

Mathematical Bridge
------------------
The bridge is built by feeding the regex‑derived feature vector **x** (from Parent A) into a Radial Basis Function (RBF) surrogate (from Parent B).  
The RBF centers **C** and weights **w** are dynamically modulated by the morphology metrics
* sphericity σ = V · π^{1/3} · (6 V)^{2/3} / S  (V = l·w·h, S = 2(lw+lh+wh))
* flatness φ = min(l,w,h) / max(l,w,h)

Thus the hybrid state update reads

    y = Σ_i w_i · exp(‑γ‖x − C_i·(1+α·σ)‖²)

where γ is the RBF spread and α scales the influence of sphericity.  
An EndpointCircuitBreaker guards the update: if a configurable failure threshold is exceeded,
the system halts further updates until a manual reset or a successful step occurs.
"""

import sys
import math
import random
import pathlib
import hashlib
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
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
    """Return a normalized count vector for the defined regex categories."""
    counts = np.array([len(pat.findall(text)) for _, pat in REGEX_PATTERNS], dtype=float)
    norm = np.linalg.norm(counts) + 1e-12
    return counts / norm  # unit‑length feature vector


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
    """Approximate sphericity based on dimensions."""
    l, w, h = morph.length, morph.width, morph.height
    if min(l, w, h) <= 0:
        return 0.0
    volume = l * w * h
    surface = 2 * (l * w + l * h + w * h)
    # Classic sphericity formula
    return (math.pi ** (1.0 / 3.0) * (6 * volume) ** (2.0 / 3.0)) / surface

def flatness_index(morph: Morphology) -> float:
    """Flatness defined as the ratio of the smallest to the largest dimension."""
    dims = [morph.length, morph.width, morph.height]
    if max(dims) == 0:
        return 0.0
    return min(dims) / max(dims)


# ----------------------------------------------------------------------
# EndpointCircuitBreaker (Parent B core)
# ----------------------------------------------------------------------
def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

class EndpointCircuitBreaker:
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
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


# ----------------------------------------------------------------------
# RBF Surrogate (Parent B core)
# ----------------------------------------------------------------------
def rbf_phi(x: np.ndarray, c: np.ndarray, gamma: float) -> float:
    """Gaussian RBF kernel."""
    diff = x - c
    return math.exp(-gamma * np.dot(diff, diff))

def rbf_output(x: np.ndarray, centers: np.ndarray, weights: np.ndarray, gamma: float) -> float:
    """Compute Σ_i w_i·φ(x,c_i)."""
    phi_vals = np.array([rbf_phi(x, c, gamma) for c in centers])
    return float(np.dot(weights, phi_vals))


# ----------------------------------------------------------------------
# Hybrid System – integration of both parents
# ----------------------------------------------------------------------
@dataclass
class HybridSystem:
    """Encapsulates the hybrid state: RBF parameters and a circuit breaker."""
    input_dim: int
    n_centers: int = 5
    gamma: float = 0.5
    alpha: float = 0.2  # scaling factor for sphericity influence
    centers: np.ndarray = field(init=False)   # shape (n_centers, input_dim)
    weights: np.ndarray = field(init=False)   # shape (n_centers,)
    breaker: EndpointCircuitBreaker = field(default_factory=EndpointCircuitBreaker)

    def __post_init__(self) -> None:
        rng = np.random.default_rng()
        self.centers = rng.normal(size=(self.n_centers, self.input_dim))
        self.weights = rng.uniform(-1.0, 1.0, size=self.n_centers)

    def adapt_centers(self, sigma: float) -> None:
        """Modulate centers by sphericity σ."""
        scale = 1.0 + self.alpha * sigma
        self.centers *= scale

    def step(self, text: str, morph: Morphology) -> float:
        """Perform a single hybrid update.

        Returns the RBF output if the circuit breaker permits; otherwise returns the
        previous output (cached in the instance). Failure/success are logged to the
        breaker based on a simple threshold.
        """
        if not hasattr(self, "_last_output"):
            self._last_output = 0.0

        if not self.breaker.allow():
            # Circuit open – no state change
            return self._last_output

        # 1️⃣ Regex feature extraction (Parent A)
        x = extract_regex_features(text)

        # 2️⃣ Morphology metrics (Parent B)
        sigma = sphericity_index(morph)
        phi = flatness_index(morph)  # currently unused but kept for extensibility

        # 3️⃣ Adapt RBF centers using sphericity
        self.adapt_centers(sigma)

        # 4️⃣ Compute surrogate output
        y = rbf_output(x, self.centers, self.weights, self.gamma)

        # 5️⃣ Record success/failure (arbitrary threshold for demo)
        if abs(y) > 0.3:
            self.breaker.record_success()
        else:
            self.breaker.record_failure()

        self._last_output = y
        return y


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def demo_feature_extraction() -> None:
    sample = "Please verify the source and provide a screenshot of the evidence."
    vec = extract_regex_features(sample)
    print("Feature vector:", vec)

def demo_morphology_metrics() -> None:
    morph = Morphology(length=2.0, width=1.0, height=0.5)
    print("Sphericity:", sphericity_index(morph))
    print("Flatness :", flatness_index(morph))

def demo_hybrid_run() -> None:
    system = HybridSystem(input_dim=len(REGEX_PATTERNS))
    texts = [
        "We need a plan and a checklist before the deadline.",
        "Can you call a friend for support?",
        "Pause the operation and wait until tomorrow.",
        "Provide proof and audit the logs.",
    ]
    morph = Morphology(length=1.2, width=0.9, height=0.4)
    for i, txt in enumerate(texts, 1):
        out = system.step(txt, morph)
        print(f"Step {i}: RBF output = {out:.4f}, breaker open = {system.breaker.open}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: Regex Feature Extraction ===")
    demo_feature_extraction()
    print("\n=== Demo: Morphology Metrics ===")
    demo_morphology_metrics()
    print("\n=== Demo: Hybrid System Run ===")
    demo_hybrid_run()