# DARWIN HAMMER — match 1, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_endpoint_circ_state_space_duality_m1_s0.py (gen2)
# parent_b: hybrid_ssim_hybrid_decision_hygi_m9_s1.py (gen2)
# born: 2026-05-29T23:25:07Z

"""Hybrid Morphology‑SSIM‑Hygiene Algorithm
================================================

Parents
-------
* **hybrid_hybrid_endpoint_circ_state_space_duality_m1_s0.py** – provides
  morphology‑based indices (sphericity, flatness, righting‑time, recovery‑priority)
  and a simple endpoint‑circuit model.
* **hybrid_ssim_hybrid_decision_hygi_m9_s1.py** – offers a Structural Similarity
  Index (SSIM) for equal‑length numeric samples and a decision‑hygiene module
  that extracts evidence‑related token frequencies and computes a weighted
  Shannon entropy.

Mathematical Bridge
-------------------
Both parents produce *scalar* descriptors of a system:

* The morphology module maps a physical object to a **state vector**  
  `v = [sphericity, flatness, righting_time, recovery_priority]`.
* The SSIM module measures similarity between two equal‑length vectors.
* The hygiene module treats a piece of text as a **categorical distribution**
  over regex‑matched token classes and evaluates its information content via
  Shannon entropy.

The hybrid algorithm therefore:

1. **Encodes** each endpoint (or any object) into a morphology state vector.
2. **Computes** the SSIM similarity of two such vectors – a continuous measure of
   how alike their physical states are.
3. **Derives** a hygiene weight from associated free‑form text using the
   regex‑based token counts and their Shannon entropy.
4. **Fuses** the similarity and hygiene weight into a single “hybrid score”

   \[
   \text{HybridScore}= \text{SSIM}(v_1,v_2)\;\times\;
   \bigl(1 - \lambda\,\frac{H_{\text{text}}}{\log N}\bigr)
   \]

   where \(H_{\text{text}}\) is the Shannon entropy of the token distribution,
   \(N\) the number of distinct token classes, and \(\lambda\in[0,1]\) a
   tunable attenuation factor (default 0.5).  The term in parentheses
   down‑weights the similarity when the accompanying text is noisy or
   low‑information.

The resulting hybrid metric can be used for robust endpoint‑pair ranking,
state‑space fusion, or any scenario where physical similarity and textual
confidence must be considered together.
"""

import math
import random
import sys
import pathlib
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A – Morphology & Endpoint definitions
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity index."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Geometric flatness index."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    """Physical right‑ing time proxy."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] based on righting time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def morphology_state_vector(m: Morphology) -> np.ndarray:
    """Collect the four scalar descriptors into a NumPy vector."""
    sph = sphericity_index(m.length, m.width, m.height)
    flat = flatness_index(m.length, m.width, m.height)
    rt = righting_time_index(m)
    rp = recovery_priority(m)
    return np.array([sph, flat, rt, rp], dtype=float)


# ----------------------------------------------------------------------
# Parent‑B – SSIM and Hygiene utilities
# ----------------------------------------------------------------------


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 1.0, k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Structural Similarity Index for equal‑length numeric vectors.
    The implementation mirrors the classic SSIM formula but works on 1‑D data.
    """
    if x.shape != y.shape:
        raise ValueError("vectors must have identical shape")
    if x.size == 0:
        raise ValueError("vectors must not be empty")
    mx = x.mean()
    my = y.mean()
    vx = ((x - mx) ** 2).mean()
    vy = ((y - my) ** 2).mean()
    cov = ((x - mx) * (y - my)).mean()
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator


# Regex groups used by the hygiene module (excerpted from the parent)
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
    r"before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|"
    r"advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk a)\b",
    re.I,
)


REGEX_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("evidence", EVIDENCE_RE),
    ("planning", PLANNING_RE),
    ("delay", DELAY_RE),
    ("support", SUPPORT_RE),
    ("boundary", BOUNDARY_RE),
]


def token_frequencies(text: str) -> Counter:
    """Count occurrences of each regex‑defined token class in *text*."""
    counts = Counter()
    for label, pattern in REGEX_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            counts[label] += len(matches)
    return counts


def shannon_entropy(freqs: Counter) -> float:
    """Compute Shannon entropy of a categorical distribution."""
    total = sum(freqs.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in freqs.values():
        p = count / total
        entropy -= p * math.log(p, 2)
    return entropy


def hygiene_weight(text: str, lambda_: float = 0.5) -> float:
    """
    Produce a weight in [0,1] that down‑weights similarity when the text is
    low‑information (high entropy).  The weight is

        w = 1 - λ * H / log2(N)

    where N is the number of distinct token classes present.
    """
    freqs = token_frequencies(text)
    if not freqs:
        return 1.0  # No hygiene signal → neutral weight
    H = shannon_entropy(freqs)
    N = len(freqs)
    max_entropy = math.log(N, 2) if N > 1 else 1.0
    weight = 1.0 - lambda_ * (H / max_entropy)
    return max(0.0, min(1.0, weight))


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------


def hybrid_morphology_similarity(m1: Morphology, m2: Morphology) -> float:
    """
    Compute SSIM similarity between the morphology state vectors of two objects.
    The vectors are normalised to unit dynamic range before SSIM evaluation.
    """
    v1 = morphology_state_vector(m1)
    v2 = morphology_state_vector(m2)
    # Normalise each vector to [0,1] to satisfy the SSIM dynamic_range assumption.
    mins = np.minimum(v1, v2)
    maxs = np.maximum(v1, v2)
    # Avoid division by zero
    scale = np.where(maxs - mins == 0, 1.0, maxs - mins)
    vn1 = (v1 - mins) / scale
    vn2 = (v2 - mins) / scale
    return ssim(vn1, vn2, dynamic_range=1.0)


def hybrid_hygiene_score(text1: str, text2: str, lambda_: float = 0.5) -> float:
    """
    Combine hygiene weights from two texts by geometric mean.
    This yields a symmetric factor in [0,1].
    """
    w1 = hygiene_weight(text1, lambda_)
    w2 = hygiene_weight(text2, lambda_)
    return math.sqrt(w1 * w2)


def hybrid_assessment(
    m1: Morphology,
    m2: Morphology,
    text1: str,
    text2: str,
    lambda_: float = 0.5,
) -> float:
    """
    Full hybrid metric:
        HybridScore = SSIM(morphology) * HygieneFactor
    """
    sim = hybrid_morphology_similarity(m1, m2)
    hy = hybrid_hygiene_score(text1, text2, lambda_)
    return sim * hy


# ----------------------------------------------------------------------
# Simple state‑space demonstration (optional extra)
# ----------------------------------------------------------------------


def semiseparable_matrix(v: np.ndarray) -> np.ndarray:
    """
    Construct a simple semiseparable matrix A where
        A[i,j] = v[i] if i <= j else v[j]
    This mimics a lower‑upper split without external libraries.
    """
    n = v.size
    A = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            A[i, j] = v[i] if i <= j else v[j]
    return A


def apply_state_space(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    One‑step linear state‑space update:
        x_next = A @ x + B @ u
    where A is the semiseparable matrix derived from *v* and B is identity.
    """
    A = semiseparable_matrix(v)
    B = np.eye(A.shape[0])
    return A @ v + B @ u


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create two simple morphology instances
    morph_a = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)
    morph_b = Morphology(length=2.2, width=0.9, height=0.45, mass=2.8)

    # Sample explanatory texts
    txt_a = (
        "The evidence was verified and the plan was executed after a short pause."
    )
    txt_b = (
        "Source documents were logged, the checklist was completed, and support was called."
    )

    # Compute hybrid assessment
    score = hybrid_assessment(morph_a, morph_b, txt_a, txt_b, lambda_=0.5)
    print(f"Hybrid similarity‑hygiene score: {score:.4f}")

    # Demonstrate state‑space operation (purely illustrative)
    vec = morphology_state_vector(morph_a)
    control = np.array([0.1, 0.2, 0.3, 0.4])
    next_state = apply_state_space(vec, control)
    print("Next state after semiseparable update:", next_state)