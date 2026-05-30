# DARWIN HAMMER — match 65, survivor 0
# gen: 3
# parent_a: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s5.py (gen2)
# parent_b: hybrid_ssim_hybrid_decision_hygi_m9_s2.py (gen2)
# born: 2026-05-29T23:26:34Z

"""
Hybrid algorithm that mathematically fuses the Voronoi partition and circuit-breaker from `hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s5.py`
with the SSIM and decision-hygiene from `hybrid_ssim_hybrid_decision_hygi_m9_s2.py`.

The mathematical bridge is established by treating the Voronoi partition as a signal and
the SSIM formula as a similarity measure between two signals. The resulting similarity
index is then combined with the individual hygiene scores and entropy values to obtain a
single hybrid metric that reflects both content similarity and decision-hygiene quality.

The module implements:
* `voronoi_ssim_vector` – Voronoi partition similarity on numeric vectors.
* `hygiene_and_entropy` – hygiene score, raw entropy and normalised entropy.
* `hybrid_voronoi_similarity` – full fusion of the two parents for a pair of Voronoi partitions,
  returning a detailed report.
"""

import math
import re
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Parent A – Voronoi helpers
# ----------------------------------------------------------------------

Point = Tuple[float, float]

def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Parent B – Circuit-breaker and Morphology
# ----------------------------------------------------------------------

class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
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

    def reliability(self) -> float:
        """Return a smooth reliability score in (0, 1]."""
        # Linear decay with a floor to avoid exact zero.
        return max(0.01, 1.0 - self.failures / (self.failure_threshold * 2))

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

# ----------------------------------------------------------------------
# Parent B – regexes and raw count extraction
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
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission)\b",
    re.I,
)

# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------

def voronoi_ssim_vector(a: np.ndarray, b: np.ndarray) -> float:
    """Voronoi partition similarity on numeric vectors."""
    # Compute the mean and variance of each vector
    mean_a, var_a = np.mean(a), np.var(a)
    mean_b, var_b = np.mean(b), np.var(b)

    # Compute the covariance of the two vectors
    cov_ab = np.cov(a, b)[0, 1]

    # Compute the standard deviations of the two vectors
    std_a, std_b = np.sqrt(var_a), np.sqrt(var_b)

    # Compute the SSIM between the two vectors
    ssim = (2 * mean_a * mean_b + cov_ab) / (mean_a ** 2 + mean_b ** 2 + std_a ** 2 + std_b ** 2)

    return ssim

def hygiene_and_entropy(text: str) -> Tuple[float, float, float]:
    """Hygiene score, raw entropy and normalised entropy."""
    # Extract the evidence, planning, delay and support keywords
    evidence_count = len(EVIDENCE_RE.findall(text))
    planning_count = len(PLANNING_RE.findall(text))
    delay_count = len(DELAY_RE.findall(text))
    support_count = len(SUPPORT_RE.findall(text))

    # Compute the hygiene score as the sum of evidence and support keywords
    hygiene_score = evidence_count + support_count

    # Compute the raw entropy as the sum of delay and planning keywords
    raw_entropy = delay_count + planning_count

    # Compute the normalised entropy as the raw entropy divided by the total number of keywords
    normalised_entropy = raw_entropy / (evidence_count + planning_count + delay_count + support_count)

    return hygiene_score, raw_entropy, normalised_entropy

def hybrid_voronoi_similarity(a: np.ndarray, b: np.ndarray, text: str) -> Tuple[float, float, float, float]:
    """Full fusion of the two parents for a pair of Voronoi partitions and a text, returning a detailed report."""
    # Compute the Voronoi partition similarity between the two vectors
    ssim = voronoi_ssim_vector(a, b)

    # Compute the hygiene score, raw entropy and normalised entropy of the text
    hygiene_score, raw_entropy, normalised_entropy = hygiene_and_entropy(text)

    # Compute the hybrid metric as the weighted sum of the Voronoi partition similarity and the hygiene score
    hybrid_metric = 0.5 * ssim + 0.5 * hygiene_score

    return ssim, hygiene_score, raw_entropy, hybrid_metric

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Generate two random Voronoi partitions
    np.random.seed(0)
    a = np.random.rand(10, 2)
    b = np.random.rand(10, 2)

    # Generate a sample text
    text = "Evidence and support are important keywords."

    # Compute the hybrid similarity
    ssim, hygiene_score, raw_entropy, hybrid_metric = hybrid_voronoi_similarity(a, b, text)

    print(f"Voronoi partition similarity: {ssim:.4f}")
    print(f"Hygiene score: {hygiene_score:.4f}")
    print(f"Raw entropy: {raw_entropy:.4f}")
    print(f"Hybrid metric: {hybrid_metric:.4f}")