# DARWIN HAMMER — match 2897, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_decision_hygi_hybrid_sketches_rlct_m6_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_fisher_locali_m711_s0.py (gen5)
# born: 2026-05-29T23:46:37Z

"""Hybrid Decision Hygiene & Fisher Risk Analyzer

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – computes Shannon entropy over decision‑hygiene feature
  counts and uses a Count‑Min sketch to approximate the sum of per‑sample
  log‑likelihoods.
* **Parent B** – evaluates reconstruction‑risk scores, weights them with
  Fisher‑information‑derived Gaussian‑beam scores, and analyses their
  evolution over chronological timestamps.

**Mathematical bridge** – Both families operate on *information measures*:
entropy quantifies uncertainty while Fisher information quantifies the
sensitivity of a likelihood to its parameters.  By interpreting the
Count‑Min sketch’s estimated log‑likelihood sum as a proxy for the
observed Fisher information, we can weight the reconstruction‑risk
scores with a Gaussian‑beam derived from the entropy‑driven variance.
The resulting hybrid metric captures (i) uncertainty of decision‑hygiene
features, (ii) privacy‑risk of quasi‑identifiers, and (iii) their temporal
dynamics.

The public API provides three representative functions demonstrating the
fusion, plus a convenience ``hybrid_decision_anonymization_score`` that
combines them into a single scalar per dataset.
"""

import re
import math
import random
import sys
import pathlib
from datetime import datetime, timezone
from collections import Counter
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Regex collections – derived from Parent A
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
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no slee)",
    re.I,
)

REGEX_GROUPS = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
    "scarcity": SCARCITY_RE,
}

# ----------------------------------------------------------------------
# Count‑Min Sketch (lightweight implementation)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Simple Count‑Min Sketch for integer counts.

    Parameters
    ----------
    width : int
        Number of columns (hash buckets) per hash function.
    depth : int
        Number of independent hash functions.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, width: int = 2_048, depth: int = 5, seed: int = 0) -> None:
        self.width = width
        self.depth = depth
        self.seed = seed
        rng = np.random.default_rng(seed)
        # Generate pairwise‑independent hash parameters (a, b) for each row
        self._hash_params = [(rng.integers(1, 2**31 - 1), rng.integers(0, 2**31 - 1)) for _ in range(depth)]
        self.table = np.zeros((depth, width), dtype=np.int64)

    def _hash(self, item: Any, row: int) -> int:
        a, b = self._hash_params[row]
        # Use built‑in hash, ensure positive, then linear hash modulo width
        return (a * (hash(item) & 0xFFFFFFFFFFFFFFFF) + b) % self.width

    def add(self, item: Any, increment: int = 1) -> None:
        for r in range(self.depth):
            col = self._hash(item, r)
            self.table[r, col] += increment

    def estimate(self, item: Any) -> int:
        """Return the minimum count across rows (the CMS estimate)."""
        estimates = [self.table[r, self._hash(item, r)] for r in range(self.depth)]
        return int(min(estimates))

# ----------------------------------------------------------------------
# Entropy utilities – derived from Parent A
# ----------------------------------------------------------------------
def shannon_entropy(counts: Iterable[int]) -> float:
    """Compute Shannon entropy (base‑e) from a sequence of integer counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts if c > 0], dtype=float)
    return -np.sum(probs * np.log(probs))


def extract_feature_counts(texts: List[str]) -> Dict[str, int]:
    """Count occurrences of each regex group across a list of texts."""
    counter = Counter()
    for txt in texts:
        for name, regex in REGEX_GROUPS.items():
            if regex.search(txt):
                counter[name] += 1
    return dict(counter)


def shannon_entropy_from_texts(texts: List[str]) -> float:
    """Convenient wrapper: compute entropy of decision‑hygiene feature counts."""
    counts = extract_feature_counts(texts).values()
    return shannon_entropy(counts)


# ----------------------------------------------------------------------
# Log‑likelihood sketch – approximating Σ log p(x) (Parent A)
# ----------------------------------------------------------------------
def countmin_sketch_loglikelihood(texts: List[str]) -> float:
    """
    Approximate the sum of per‑sample log‑likelihoods using a Count‑Min sketch.
    For demonstration, we treat each unique token as an event whose frequency
    is stored in the sketch; the log‑likelihood contribution of a token is
    log( frequency / total_tokens ).
    """
    sketch = CountMinSketch(width=4096, depth=7, seed=42)
    total_tokens = 0

    for txt in texts:
        tokens = txt.split()
        total_tokens += len(tokens)
        for tk in tokens:
            sketch.add(tk)

    # Estimate frequencies and compute log‑likelihood sum
    loglik_sum = 0.0
    for txt in texts:
        for tk in txt.split():
            est = sketch.estimate(tk)
            # Laplace smoothing to avoid log(0)
            prob = (est + 1) / (total_tokens + sketch.width * sketch.depth)
            loglik_sum += math.log(prob)

    return loglik_sum


# ----------------------------------------------------------------------
# Reconstruction risk & Fisher weighting – derived from Parent B
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Normalized risk in [0,1] based on the proportion of unique identifiers."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Fisher‑information‑style Gaussian kernel."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_weighted_risk(
    risk_scores: List[float],
    timestamps: List[datetime],
    reference: datetime | None = None,
    time_scale_seconds: float = 86_400.0,
) -> float:
    """
    Weight risk scores using a Gaussian beam centred on ``reference``.
    The beam width is expressed in seconds (default: one day).
    """
    if not risk_scores:
        return 0.0
    if reference is None:
        reference = max(timestamps)  # most recent as default centre

    # Convert timestamps to seconds offset from reference
    offsets = np.array(
        [(ref - reference).total_seconds() for ref in timestamps], dtype=float
    )
    # Compute Gaussian weights
    weights = np.array([gaussian_beam(off, 0.0, time_scale_seconds) for off in offsets])
    weighted = np.sum(np.array(risk_scores) * weights) / (np.sum(weights) + 1e-12)
    return float(weighted)


# ----------------------------------------------------------------------
# Hybrid operation – three public functions
# ----------------------------------------------------------------------
def hybrid_entropy_risk_score(texts: List[str]) -> float:
    """
    Combine Shannon entropy of decision‑hygiene features with a reconstruction
    risk score derived from the number of distinct regex groups observed.
    The risk denominator is the total number of texts.
    """
    # Entropy component (Parent A)
    entropy = shannon_entropy_from_texts(texts)

    # Risk component (Parent B)
    feature_counts = extract_feature_counts(texts)
    unique_groups = len([c for c in feature_counts.values() if c > 0])
    risk = reconstruction_risk_score(unique_groups, len(texts))

    # Fuse: treat entropy as a variance proxy; higher entropy reduces effective risk
    fused = risk * math.exp(-entropy)  # exponential decay links entropy to Fisher info
    return fused


def hybrid_fisher_temporal_score(
    texts: List[str],
    timestamps: List[datetime],
) -> float:
    """
    Compute a Fisher‑weighted risk that also incorporates the log‑likelihood
    sketch estimate.  The sketch estimate acts as an information density term.
    """
    # Step 1 – risk per sample (simple proxy: 1 if any regex matches, else 0)
    per_sample_risk = [
        1.0 if any(regex.search(txt) for regex in REGEX_GROUPS.values()) else 0.0
        for txt in texts
    ]

    # Step 2 – Fisher‑weighted aggregate risk over time
    weighted_risk = fisher_weighted_risk(per_sample_risk, timestamps)

    # Step 3 – information density from Count‑Min sketch log‑likelihood
    info_density = countmin_sketch_loglikelihood(texts)

    # Fuse: higher information density (more certain) amplifies the weighted risk
    fused_score = weighted_risk * (1.0 + info_density / (abs(info_density) + 1.0))
    return fused_score


def hybrid_decision_anonymization_score(
    texts: List[str],
    timestamps: List[datetime],
) -> float:
    """
    End‑to‑end hybrid metric that merges:
      * Decision‑hygiene entropy (uncertainty)
      * Sketch‑based log‑likelihood (information density)
      * Reconstruction risk weighted by Fisher information and temporality

    The formula is:
        S = (entropy_factor) * (fisher_temporal_score)

    where ``entropy_factor = exp(-entropy)``.
    """
    if len(texts) != len(timestamps):
        raise ValueError("texts and timestamps must have the same length")

    entropy = shannon_entropy_from_texts(texts)
    entropy_factor = math.exp(-entropy)

    fisher_score = hybrid_fisher_temporal_score(texts, timestamps)

    return entropy_factor * fisher_score


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I have verified the document and recorded the proof.",
        "Plan the roadmap and set the schedule for next week.",
        "Wait before you act, we need to check the evidence.",
        "Contact the therapist for support and keep the boundary safe.",
        "The budget is approved and the project is shipped.",
        "I can't afford the rent, I'm desperate.",
    ]

    # Generate synthetic timestamps spread over the past week
    now = datetime.now(timezone.utc)
    timestamps = [
        now - datetime.timedelta(seconds=random.randint(0, 7 * 24 * 3600))
        for _ in sample_texts
    ]

    print("Shannon entropy:", shannon_entropy_from_texts(sample_texts))
    print("Log‑likelihood sketch sum:", countmin_sketch_loglikelihood(sample_texts))
    print("Hybrid entropy‑risk:", hybrid_entropy_risk_score(sample_texts))
    print("Hybrid Fisher temporal score:", hybrid_fisher_temporal_score(sample_texts, timestamps))
    print("Overall hybrid decision‑anonymization score:", hybrid_decision_anonymization_score(sample_texts, timestamps))