# DARWIN HAMMER — match 3303, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hdc_hybrid_hy_m996_s1.py (gen4)
# parent_b: hybrid_hybrid_pheromone_hyb_chelydrid_ambush_m183_s2.py (gen3)
# born: 2026-05-29T23:49:09Z

"""
Hybrid Decision‑Hygiene Hyperdimensional Pheromone‑Hash (DDHPH)

Parents
-------
* **Parent A** – *Hybrid Decision Hygiene & Shannon Entropy* combined with
  *Hyperdimensional Computing & Hybrid Ternary Lens Audit*.  
  Core: regex‑driven feature extraction → weighted feature vector → bipolar
  hyperdimensional (HDC) encoding (DIM = 10 000).

* **Parent B** – *Hybrid Pheromone & Chelydrid Ambush*.  
  Core: perceptual hashing of numeric series (`compute_phash`), Hamming
  distance, broadcast probability, and a triangular pulse force model.

Mathematical Bridge
-------------------
The bridge maps the **decision‑hygiene feature vector** (9‑dimensional) onto a
bipolar HDC vector.  The first 64 components of that HDC vector are treated as
a numeric series and fed to the **perceptual‑hash** routine from Parent B.
The resulting 64‑bit hash is compared (Hamming distance) against a reference
hash derived from a neutral HDC vector.  The distance is then modulated by
the **broadcast probability** (phase = sum of positive weights) and by a
time‑varying **pulse force** series, yielding a scalar “hybrid score” that
simultaneously reflects decision‑hygiene strength, hyperdimensional similarity,
and pheromone‑style dynamics.

The module provides three high‑level functions that demonstrate the fused
pipeline:
1. `extract_features(text)` – regex feature extraction (Parent A).
2. `encode_hdc(features)` – weighted bipolar HDC encoding (Parent A).
3. `hybrid_score(hdc_vec, phase, steps)` – perceptual hash, Hamming distance,
   broadcast probability and pulse force integration (Parent B).

The `select_top_entities` helper shows a complete end‑to‑end use‑case.
"""

import numpy as np
import re
import sys
import math
import random
from pathlib import Path
from typing import List, Tuple, Iterable

# ----------------------------------------------------------------------
# Parent A constants and regexes
# ----------------------------------------------------------------------
DIM = 10_000

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

# compiled regexes (case‑insensitive)
_EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
_PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
_DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
    r"before i|first|after|review)\b",
    re.I,
)
_SUPPORT_RE = re.compile(r"\b(?:support|help|assist|aid|backing|sustain)\b", re.I)
_BOUNDARY_RE = re.compile(r"\b(?:boundary|limit|threshold|cap|border|edge)\b", re.I)
_OUTCOME_RE = re.compile(r"\b(?:outcome|result|conclusion|effect|impact)\b", re.I)
_IMPULSIVE_RE = re.compile(r"\b(?:impulsive|rash|hasty|quick|spontaneous)\b", re.I)
_SCARCITY_RE = re.compile(r"\b(?:scarcity|rare|limited|shortage|few)\b", re.I)
_RISK_RE = re.compile(r"\b(?:risk|danger|threat|hazard|uncertainty)\b", re.I)

_REGEX_MAP = {
    "evidence": _EVIDENCE_RE,
    "planning": _PLANNING_RE,
    "delay": _DELAY_RE,
    "support": _SUPPORT_RE,
    "boundary": _BOUNDARY_RE,
    "outcome": _OUTCOME_RE,
    "impulsive": _IMPULSIVE_RE,
    "scarcity": _SCARCITY_RE,
    "risk": _RISK_RE,
}


def extract_features(text: str) -> np.ndarray:
    """
    Count occurrences of each feature regex in *text* and return a 9‑element
    integer array ordered according to ``_FEATURE_ORDER``.
    """
    counts = np.zeros(len(_FEATURE_ORDER), dtype=np.int64)
    for idx, name in enumerate(_FEATURE_ORDER):
        pattern = _REGEX_MAP[name]
        counts[idx] = len(pattern.findall(text))
    return counts


# ----------------------------------------------------------------------
# Parent A – Hyperdimensional encoding
# ----------------------------------------------------------------------
_rng = np.random.default_rng(42)  # deterministic for reproducibility

def _random_bipolar_vector() -> np.ndarray:
    """Return a bipolar vector of length DIM with entries -1 or +1."""
    bits = _rng.integers(0, 2, size=DIM, endpoint=False)
    return np.where(bits, 1, -1).astype(np.int8)


# Pre‑generate a base vector for each feature index
_BASE_VECTORS = [_random_bipolar_vector() for _ in range(len(_FEATURE_ORDER))]


def encode_hdc(feature_counts: np.ndarray) -> np.ndarray:
    """
    Encode *feature_counts* (9‑dim integer array) into a bipolar HDC vector.

    For each feature *i*:
        v_i = base_vector_i * weight_i
    where ``weight_i`` is taken from ``_POSITIVE_WEIGHTS`` if the count is
    positive, otherwise from ``_NEGATIVE_WEIGHTS``.  All weighted vectors are
    summed and the sign of each component yields the final bipolar vector.
    """
    if feature_counts.shape[0] != len(_FEATURE_ORDER):
        raise ValueError("feature_counts must have length 9")
    acc = np.zeros(DIM, dtype=np.int64)
    for i, cnt in enumerate(feature_counts):
        if cnt == 0:
            continue
        weight = _POSITIVE_WEIGHTS[i] if cnt > 0 else _NEGATIVE_WEIGHTS[i]
        acc += _BASE_VECTORS[i] * weight * cnt
    # Binarize to bipolar {-1, +1}
    hdc_vec = np.where(acc >= 0, 1, -1).astype(np.int8)
    return hdc_vec


# ----------------------------------------------------------------------
# Parent B – Perceptual hashing and dynamic models
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """
    Compute a 64‑bit perceptual hash of *values*.
    The first 64 elements are thresholded against the mean of the list.
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Return the Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


def broadcast_probability(phase: int, step: int) -> float:
    """
    Probability that a broadcast (or “burst”) is accepted at *step* given a
    *phase* budget.  Mirrors the exponential decay used in the ambush model.
    """
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


def pulse_force(peak_force: float, steps: int) -> List[float]:
    """
    Triangular pulse: rises to *peak_force* at the centre and falls symmetrically.
    Used to weight successive hash‑distance contributions.
    """
    if peak_force < 0 or steps <= 0:
        raise ValueError("peak_force non‑negative and steps positive required")
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    return [
        peak_force * max(0.0, 1.0 - abs(i - mid) / denom) for i in range(steps)
    ]


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
_REFERENCE_HDC = encode_hdc(np.zeros(len(_FEATURE_ORDER), dtype=np.int64))
_REFERENCE_HASH = compute_phash(_REFERENCE_HDC[:64].astype(float).tolist())


def hybrid_score(hdc_vec: np.ndarray, phase: int, steps: int) -> Tuple[int, int, float, float]:
    """
    Compute a hybrid score tuple for *hdc_vec*.

    Returns
    -------
    (hash_val, hamming, prob, weighted_force)

    *hash_val* – 64‑bit perceptual hash of the first 64 HDC components.
    *hamming* – distance to the reference hash.
    *prob*    – broadcast probability using *phase* and current *step* (steps==1).
    *weighted_force* – pulse‑force weighting applied to the distance.
    """
    # 1. perceptual hash of the HDC vector
    hash_val = compute_phash(hdc_vec[:64].astype(float).tolist())

    # 2. similarity to a neutral reference
    hamming = hamming_distance(hash_val, _REFERENCE_HASH)

    # 3. broadcast probability (use step=1 for a single‑entity score)
    prob = broadcast_probability(phase, 1)

    # 4. pulse force weighting (peak force inversely proportional to distance)
    pulse_series = pulse_force(peak_force=1.0 / (1 + hamming), steps=steps)
    weighted_force = pulse_series[0] if pulse_series else 0.0

    return hash_val, hamming, prob, weighted_force


def select_top_entities(texts: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    """
    End‑to‑end pipeline:

    1. Extract decision‑hygiene features from each *text*.
    2. Encode into an HDC vector.
    3. Compute a hybrid score (lower hamming ⇒ higher quality).
    4. Return the *top_k* texts sorted by a composite metric:
       ``score = (1 / (1 + hamming)) * prob * weighted_force``.
    """
    results = []
    phase = int(_POSITIVE_WEIGHTS.sum())  # a natural phase budget from Parent A
    for txt in texts:
        feats = extract_features(txt)
        hdc = encode_hdc(feats)
        _, hamming, prob, wforce = hybrid_score(hdc, phase, steps=1)
        composite = (1.0 / (1 + hamming)) * prob * wforce
        results.append((txt, composite))
    # sort descending by composite score
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The audit confirmed the evidence and source hash, and the plan was detailed.",
        "We need to wait, maybe later. The risk is high and scarcity of resources.",
        "Immediate action! Impulsive decision without proper support or boundary checks.",
        "A thorough checklist was created, supporting the outcome with clear evidence.",
        "Delay the deployment; the risk assessment shows potential hazards."
    ]

    top = select_top_entities(sample_texts, top_k=3)
    print("Top entities based on hybrid DDHPH score:")
    for idx, (txt, score) in enumerate(top, 1):
        print(f"{idx}. Score: {score:.6f} | Text: {txt}")