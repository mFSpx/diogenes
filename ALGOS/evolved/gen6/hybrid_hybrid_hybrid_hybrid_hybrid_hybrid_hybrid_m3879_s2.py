# DARWIN HAMMER — match 3879, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_possum_filter_m1639_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m2613_s0.py (gen5)
# born: 2026-05-29T23:52:13Z

"""
Hybrid Algorithm: Fisher‑Information Guided Regret‑Weighted Sketch Privacy Score
==========================================================================

Parents:
- **Algorithm A** (`hybrid_hybrid_hybrid_fisher_hybrid_possum_filter_m1639_s0.py`):  
  Provides a Gaussian beam model, Fisher information computation, and a count‑min sketch
  for frequency estimation.
- **Algorithm B** (`hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m2613_s0.py`):  
  Defines feature‑wise positive/negative weights, regex‑based feature extraction,
  and a regret‑weighted expected reward calculation with a privacy‑preserving utility term.

Mathematical Bridge:
The bridge is the **use of Fisher information as a data‑driven weighting factor**
for the regret‑weighted reward.  Frequencies of textual features are estimated with a
count‑min sketch (A).  Those estimated frequencies are turned into a feature vector
that is multiplied by the positive/negative weight vectors (B).  The resulting raw
reward is scaled by a *regret factor* (the gap to the maximal possible reward) and
further modulated by a *privacy utility* that decays exponentially with the total
sketch count.  Finally, the whole expression is multiplied by the sum of Fisher
scores computed over a set of Gaussian‑beam parameters, giving a Fisher‑information
guided scaling.

The unified system therefore consists of three tightly coupled stages:
1. **Sketch construction & query** – fast, sub‑linear frequency estimation.
2. **Feature extraction & regret‑weighted reward** – deterministic linear algebra.
3. **Fisher‑information scaling & privacy utility** – non‑linear modulation.

All three stages are expressed as pure NumPy / std‑lib operations, keeping the
implementation lightweight and fully reproducible.
"""

import math
import random
import sys
import pathlib
import hashlib
import re
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Gaussian beam & Fisher information
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def total_fisher_score(thetas: List[float], center: float, width: float) -> float:
    """Sum of Fisher scores over a list of angles."""
    return sum(fisher_score(t, center, width) for t in thetas)


# ----------------------------------------------------------------------
# Parent A – Count‑Min Sketch utilities
# ----------------------------------------------------------------------
def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Build a count‑min sketch for *items*."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = hashlib.sha256(f"{d}:{item}".encode()).hexdigest()
            idx = int(h, 16) % width
            table[d][idx] += 1
    return table


def query_sketch(sketch: List[List[int]], token: str) -> int:
    """Estimate frequency of *token* using the sketch (minimum over rows)."""
    estimates = []
    depth = len(sketch)
    width = len(sketch[0])
    for d in range(depth):
        h = hashlib.sha256(f"{d}:{token}".encode()).hexdigest()
        idx = int(h, 16) % width
        estimates.append(sketch[d][idx])
    return min(estimates)


# ----------------------------------------------------------------------
# Parent B – Feature regexes and weight vectors
# ----------------------------------------------------------------------
_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

_REGEX_MAP: Dict[str, re.Pattern] = {
    "evidence": re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
        re.I,
    ),
    "planning": re.compile(
        r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
        re.I,
    ),
    "delay": re.compile(
        r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
        re.I,
    ),
    "support": re.compile(r"\b(?:support|help|assist|aid|service|maintenance)\b", re.I),
    "boundary": re.compile(r"\b(?:boundary|limit|threshold|cap|ceil|floor)\b", re.I),
    "outcome": re.compile(r"\b(?:outcome|result|conclusion|effect|impact)\b", re.I),
    "impulsive": re.compile(r"\b(?:impulsive|rash|hasty|quick|instant)\b", re.I),
    "scarcity": re.compile(r"\b(?:scarcity|limited|rare|shortage|few)\b", re.I),
    "risk": re.compile(r"\b(?:risk|danger|hazard|threat|exposure)\b", re.I),
}


def extract_feature_counts(items: List[str]) -> np.ndarray:
    """
    Count occurrences of each regex in *items* and return a vector ordered by
    ``_FEATURE_ORDER``.  The raw counts are returned as a NumPy int64 array.
    """
    counts = np.zeros(len(_FEATURE_ORDER), dtype=np.int64)
    for i, feat in enumerate(_FEATURE_ORDER):
        pattern = _REGEX_MAP[feat]
        cnt = sum(bool(pattern.search(text)) for text in items)
        counts[i] = cnt
    return counts


# ----------------------------------------------------------------------
# Hybrid core – Regret‑weighted reward with Fisher scaling
# ----------------------------------------------------------------------
def regret_weighted_reward(
    feature_counts: np.ndarray,
    positive_weights: np.ndarray = _POSITIVE_WEIGHTS,
    negative_weights: np.ndarray = _NEGATIVE_WEIGHTS,
) -> Tuple[float, float]:
    """
    Compute the raw reward and the regret term.

    * ``reward``  = Σ (pos_i * count_i) – Σ (neg_i * count_i)
    * ``max_reward`` = Σ pos_i * max_count_i   (where max_count_i is the
      maximum possible count, approximated by the total number of items)
    * ``regret`` = max_reward – reward   (non‑negative)

    Returns (reward, regret).
    """
    reward = float((positive_weights * feature_counts).sum() - (negative_weights * feature_counts).sum())
    total_items = int(feature_counts.sum())
    max_reward = float((positive_weights * total_items).sum())
    regret = max(max_reward - reward, 0.0)
    return reward, regret


def privacy_utility(total_sketch_counts: int, alpha: float = 0.001) -> float:
    """
    Exponential privacy utility that decays with the overall number of
    observations stored in the sketch.

    u = exp(-α * N)
    """
    return math.exp(-alpha * total_sketch_counts)


def hybrid_fisher_regret_score(
    items: List[str],
    thetas: List[float],
    center: float = 0.0,
    width: float = 1.0,
    sketch_width: int = 64,
    sketch_depth: int = 4,
    alpha: float = 0.001,
) -> float:
    """
    End‑to‑end hybrid score:

    1. Build a count‑min sketch for *items*.
    2. Estimate feature frequencies via the sketch (instead of raw counting) to
       preserve privacy.
    3. Compute regret‑weighted reward.
    4. Scale by a privacy utility term.
    5. Finally multiply by the sum of Fisher scores over *thetas* (the bridge).

    The function returns a single float that can be used for ranking or decision
    making.
    """
    # 1. Sketch construction
    sketch = count_min_sketch(items, width=sketch_width, depth=sketch_depth)

    # 2. Sketch‑based feature estimation
    # For each feature we query the sketch with a representative token.
    # We use the first matching word from the regex pattern as the token.
    feature_estimates = np.zeros(len(_FEATURE_ORDER), dtype=np.int64)
    for i, feat in enumerate(_FEATURE_ORDER):
        pattern = _REGEX_MAP[feat]
        # Find a representative token from the corpus; fallback to the feature name.
        token = None
        for text in items:
            m = pattern.search(text)
            if m:
                token = m.group(0)
                break
        token = token or feat  # guaranteed non‑None
        feature_estimates[i] = query_sketch(sketch, token)

    # 3. Regret‑weighted reward
    reward, regret = regret_weighted_reward(feature_estimates)

    # 4. Privacy utility
    total_counts = sum(sum(row) for row in sketch)
    util = privacy_utility(total_counts, alpha=alpha)

    # 5. Fisher scaling
    fisher_factor = total_fisher_score(thetas, center, width)

    # Combine: (reward - regret) * util * fisher_factor
    # Subtracting regret penalises low‑performing configurations.
    hybrid_score = (reward - regret) * util * fisher_factor
    return hybrid_score


# ----------------------------------------------------------------------
# Additional helper demonstrating independent components
# ----------------------------------------------------------------------
def sketch_feature_vector(items: List[str], sketch_width: int = 64, sketch_depth: int = 4) -> np.ndarray:
    """
    Build a sketch and return the estimated feature vector (using the sketch for
    each feature).  This isolates the sketch‑based estimation logic for reuse.
    """
    sketch = count_min_sketch(items, width=sketch_width, depth=sketch_depth)
    vec = np.zeros(len(_FEATURE_ORDER), dtype=np.int64)
    for i, feat in enumerate(_FEATURE_ORDER):
        pattern = _REGEX_MAP[feat]
        token = None
        for text in items:
            m = pattern.search(text)
            if m:
                token = m.group(0)
                break
        token = token or feat
        vec[i] = query_sketch(sketch, token)
    return vec


def fisher_scaled_regret(
    reward: float,
    regret: float,
    thetas: List[float],
    center: float = 0.0,
    width: float = 1.0,
) -> float:
    """
    Apply only the Fisher scaling to a (reward, regret) pair.
    """
    fisher = total_fisher_score(thetas, center, width)
    return (reward - regret) * fisher


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_items = [
        "The evidence was verified by the audit team.",
        "We need to plan the roadmap and schedule the next phase.",
        "Please wait until tomorrow before proceeding.",
        "Support will be provided during the test.",
        "The boundary limit is set to 1000.",
        "The outcome of the experiment was positive.",
        "An impulsive decision was made.",
        "Scarcity of resources caused delays.",
        "Risk assessment is required for this operation.",
    ]

    # Choose a set of angles for Fisher computation
    thetas = [i * 0.1 for i in range(-30, 31)]  # -3.0 … 3.0

    score = hybrid_fisher_regret_score(
        items=sample_items,
        thetas=thetas,
        center=0.0,
        width=1.0,
        sketch_width=128,
        sketch_depth=5,
        alpha=0.0005,
    )
    print(f"Hybrid Fisher‑Regret score: {score:.6f}")

    # Demonstrate the independent helper functions
    vec = sketch_feature_vector(sample_items)
    print("Sketch‑estimated feature vector:", vec.tolist())

    reward, regret = regret_weighted_reward(vec)
    scaled = fisher_scaled_regret(reward, regret, thetas)
    print(f"Regret‑weighted reward: {reward:.2f}, Regret: {regret:.2f}, Fisher‑scaled: {scaled:.6f}")