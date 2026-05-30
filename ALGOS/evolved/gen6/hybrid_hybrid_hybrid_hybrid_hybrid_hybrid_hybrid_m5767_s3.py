# DARWIN HAMMER — match 5767, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2368_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m2489_s0.py (gen5)
# born: 2026-05-30T00:04:36Z

"""Hybrid Algorithm integrating:
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_lens__m159_s1.py (sheaf‑based ternary DAM with SSIM & entropy)
- Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m2489_s0.py (regex‑driven regret, Gini & Fisher‑adjusted circuit breaker)

Mathematical Bridge:
Both parents manipulate a scalar “temperature” β that governs a Dense Associative Memory (DAM).
Parent A scales β with SSIM(section, prototype) and the Shannon entropy of a ternary vector.
Parent B derives a regret term from the Gini coefficient of feature distributions and a Fisher‑score
adjustment based on binary regex classifications.  

The hybrid therefore defines  

    β = β₀·(1 + SSIM)·(1 + H)·(1 + R)·(1 + F)

where  
H – Shannon entropy of the ternary sheaf section,  
R – regret factor = Gini · log(1 + weekday‑regret),  
F – normalized Fisher score from regex‑feature classes,  
β₀ – base temperature.  

β then drives both the DAM energy evaluation (conceptual) and the dynamic failure threshold of an
EndpointCircuitBreaker.  This unified formulation fuses the topologies of both parents into a
single, mathematically coherent system.
"""

import numpy as np
import math
import random
import sys
import pathlib
import re
from collections import Counter
from datetime import date

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Structural Similarity Index between two vectors."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    )


def shannon_entropy(ternary_vector: np.ndarray) -> float:
    """Entropy of a ternary distribution (-1,0,1)."""
    counts = np.array(
        [
            np.count_nonzero(ternary_vector == -1),
            np.count_nonzero(ternary_vector == 0),
            np.count_nonzero(ternary_vector == 1),
        ]
    )
    probs = counts / counts.sum()
    # avoid log(0) by masking zero probabilities
    mask = probs > 0
    return -np.sum(probs[mask] * np.log2(probs[mask]))


class Sheaf:
    """Very small subset of the sheaf wrapper used in Parent A."""

    def __init__(self, node_dims, edges):
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._sections = {}  # edge -> vector
        self._restrictions = {}  # edge -> (src_map, dst_map)

    def set_section(self, edge, vector):
        self._sections[edge] = np.asarray(vector, dtype=int)

    def get_section(self, edge):
        return self._sections.get(edge, None)

    def set_restriction(self, edge, src_map, dst_map):
        self._restrictions[edge] = (np.asarray(src_map, float), np.asarray(dst_map, float))

    def restrict(self, edge):
        """Apply restriction maps to the stored section (if any)."""
        if edge not in self._sections or edge not in _restrictions:
            return None
        src_map, dst_map = self._restrictions[edge]
        vec = self._sections[edge]
        return dst_map @ (src_map @ vec)


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
# Regex feature groups (same as Parent B)
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
OUTCOME_RE = re.compile(r"\b(?:done|shippe)\b", re.I)

REGEX_GROUPS = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
}


def extract_regex_features(text: str) -> np.ndarray:
    """Count matches for each regex group; returns a feature vector."""
    counts = [len(pattern.findall(text)) for pattern in REGEX_GROUPS.values()]
    return np.asarray(counts, dtype=float)


def gini_coefficient(x: np.ndarray) -> float:
    """Gini coefficient of a non‑negative vector."""
    if x.size == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = x.size
    cumulative = np.cumsum(sorted_x, dtype=float)
    sum_x = cumulative[-1]
    if sum_x == 0:
        return 0.0
    gini = 1 - (2 / (n - 1)) * (np.sum(cumulative) / sum_x - (n + 1) / 2)
    return gini


def fisher_score(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Simple Fisher score for binary classification.
    features: (n_samples, n_features)
    labels:   (n_samples,) binary {0,1}
    Returns the mean Fisher score across features.
    """
    if features.ndim == 1:
        features = features[:, None]
    scores = []
    for i in range(features.shape[1]):
        f = features[:, i]
        f0 = f[labels == 0]
        f1 = f[labels == 1]
        if f0.size == 0 or f1.size == 0:
            continue
        mean0, mean1 = f0.mean(), f1.mean()
        var0, var1 = f0.var(), f1.var()
        denom = var0 + var1
        if denom == 0:
            continue
        scores.append(((mean1 - mean0) ** 2) / denom)
    return float(np.mean(scores)) if scores else 0.0


class EndpointCircuitBreaker:
    """Failure counter that opens after a dynamic threshold."""

    def __init__(self, failure_threshold: int = 5):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True

    def reset(self):
        self.failures = 0
        self.open = False

    def adjust_threshold(self, beta: float):
        """Scale the threshold proportionally to the current temperature β."""
        # Keep a minimum threshold of 1
        new_thresh = max(1, int(round(beta * 3)))
        self.failure_threshold = new_thresh
        # If we lowered the threshold below current failures, open the breaker
        if self.failures >= self.failure_threshold:
            self.open = True


# ----------------------------------------------------------------------
# Hybrid core: temperature β and integrated operations
# ----------------------------------------------------------------------
def compute_regret_factor(gini: float, weekday_seq: np.ndarray) -> float:
    """
    Regret = Gini * log(1 + variance of weekday counts)
    Weekday sequence is an integer array where 0=Mon … 6=Sun.
    """
    if weekday_seq.size == 0:
        return 0.0
    counts = np.bincount(weekday_seq, minlength=7)
    variance = np.var(counts)
    return gini * math.log1p(variance)


def hybrid_beta(
    base_beta: float,
    ternary_section: np.ndarray,
    prototype: np.ndarray,
    text: str,
    weekday_seq: np.ndarray,
) -> float:
    """
    Compute the unified temperature β using:
    - SSIM between ternary_section and prototype
    - Shannon entropy of ternary_section
    - Regret factor derived from Gini of regex feature counts and weekday variance
    - Fisher score from regex features (binary labeling based on presence of 'outcome')
    """
    # 1. SSIM term
    ssim = compute_ssim(ternary_section.astype(float), prototype.astype(float))

    # 2. Entropy term
    entropy = shannon_entropy(ternary_section)

    # 3. Regret term
    features = extract_regex_features(text)
    gini = gini_coefficient(features)
    regret = compute_regret_factor(gini, weekday_seq)

    # 4. Fisher term – label 1 if outcome keyword present, else 0
    outcome_present = int(bool(OUTCOME_RE.search(text)))
    labels = np.full(features.shape, outcome_present, dtype=int)
    fisher = fisher_score(features[:, None], labels)

    # Combine multiplicatively (adding 1 to keep positivity)
    beta = base_beta * (1 + ssim) * (1 + entropy) * (1 + regret) * (1 + fisher)
    return beta


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def generate_random_ternary_vector(dim: int) -> np.ndarray:
    """Create a random ternary vector of given dimension."""
    return np.random.choice([-1, 0, 1], size=dim)


def weekday_sequence(start_date: date, days: int) -> np.ndarray:
    """Return an array of weekday indices for a consecutive range."""
    seq = [(start_date + np.timedelta64(i, "D")).astype('datetime64[D]').astype(int) % 7 for i in range(days)]
    return np.asarray(seq, dtype=int)


def run_hybrid_cycle(text: str, sheaf: Sheaf, edge: tuple, prototype: np.ndarray, cb: EndpointCircuitBreaker) -> float:
    """
    Execute one hybrid iteration:
    1. Generate a ternary section and store it.
    2. Compute β via hybrid_beta().
    3. Adjust the circuit breaker threshold.
    4. Optionally record a synthetic failure based on a random event.
    Returns the computed β.
    """
    # 1. Section generation
    dim = prototype.shape[0]
    section = generate_random_ternary_vector(dim)
    sheaf.set_section(edge, section)

    # 2. Compute β
    weekday_seq = weekday_sequence(date.today(), 14)  # two‑week window
    beta = hybrid_beta(
        base_beta=0.5,
        ternary_section=section,
        prototype=prototype,
        text=text,
        weekday_seq=weekday_seq,
    )

    # 3. Adjust circuit breaker
    cb.adjust_threshold(beta)

    # 4. Simulate a failure with probability inversely related to β
    if random.random() > (beta / (beta + 1)):
        cb.record_failure()

    return beta


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal sheaf topology
    node_dims = {"A": 8, "B": 8}
    edges = [("A", "B")]
    sheaf = Sheaf(node_dims, edges)

    # Prototype vector (could be a learned pattern; here random)
    prototype = generate_random_ternary_vector(8)

    # Circuit breaker instance
    cb = EndpointCircuitBreaker(failure_threshold=5)

    # Sample text containing various regex cues
    sample_text = (
        "We have verified the source and attached the screenshot. "
        "The next steps include planning the roadmap and scheduling a review. "
        "If any boundary issues arise, please contact support."
    )

    beta = run_hybrid_cycle(sample_text, sheaf, ("A", "B"), prototype, cb)

    print(f"Computed beta: {beta:.4f}")
    print(f"Circuit breaker threshold: {cb.failure_threshold}")
    print(f"Failures recorded: {cb.failures}")
    print(f"Circuit breaker open: {cb.open}")