# DARWIN HAMMER — match 836, survivor 2
# gen: 5
# parent_a: honeybee_store.py (gen0)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ssim_hybrid_d_m24_s1.py (gen4)
# born: 2026-05-29T23:31:08Z

"""Hybrid Store‑SSIM Algorithm
Combines the decentralized store dynamics of **honeybee_store.py** (Parent A) with the
SSIM‑based decision hygiene scoring of **hybrid_hybrid_hybrid_hybrid_hybrid_ssim_hybrid_d_m24_s1.py**
(Parent B).

Mathematical bridge:
1. The store update yields a change Δstore (eq. 1 of Parent A).
2. A feature vector is extracted from input text; SSIM is computed between this
   vector and a reference vector (eq. 2 of Parent B).
3. The SSIM score weights Δstore to produce a hybrid score (eq. 3 of Parent B).

Thus the SSIM similarity acts as a dynamic gain on the store’s delta, fusing the
two topologies into a single feedback loop."""
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np

# -------------------- Constants (from Parent B) --------------------
ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
K1 = 0.01
K2 = 0.03
L = 255.0

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

# Simple regex to catch evidence‑related tokens (excerpt from Parent B)
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)

# -------------------- Core Functions --------------------

def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = ALPHA,
    beta: float = BETA,
    dt: float = DT,
) -> Tuple[float, float]:
    """
    Parent‑A store dynamics.
    Returns the updated store value and the raw delta (Δstore).
    """
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta


def _ssim_index(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute a simplified SSIM index between two 1‑D feature vectors.
    Uses the classic luminance‑contrast formulation.
    """
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)

    # Guard against division by zero
    if denominator == 0:
        return 0.0
    return numerator / denominator


def text_to_feature_vector(text: str) -> np.ndarray:
    """
    Very lightweight feature extraction:
    - Counts of each keyword in _FEATURE_ORDER.
    - Evidence tokens get an extra boost via _POSITIVE_WEIGHTS.
    Returns a float vector suitable for SSIM comparison.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = Counter(tokens)

    vec = np.zeros(len(_FEATURE_ORDER), dtype=np.float64)
    for idx, feat in enumerate(_FEATURE_ORDER):
        base = counts.get(feat, 0)
        # Apply positive / negative weighting heuristics
        weight = _POSITIVE_WEIGHTS[idx] - _NEGATIVE_WEIGHTS[idx]
        vec[idx] = base * weight

    # Add a bonus for any evidence‑type token found
    if EVIDENCE_RE.search(text):
        vec[0] += 500.0  # arbitrary boost to the "evidence" dimension
    return vec


def hybrid_score(
    delta_store: float,
    text_vec: np.ndarray,
    reference_vec: np.ndarray,
) -> float:
    """
    Combine Δstore with SSIM similarity.
    hybrid_score = ssim_score * Δstore
    """
    ssim_score = _ssim_index(text_vec, reference_vec)
    return ssim_score * delta_store


def hybrid_step(
    store: float,
    inflow: List[float],
    outflow: List[float],
    text: str,
    reference_vec: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Perform one iteration:
    1. Update the store (Parent A).
    2. Convert the incoming text to a feature vector.
    3. Compute the hybrid score (Parent B).
    Returns (new_store, delta_store, hybrid_score).
    """
    new_store, delta = update_store(store, inflow, outflow)
    txt_vec = text_to_feature_vector(text)
    h_score = hybrid_score(delta, txt_vec, reference_vec)
    return new_store, delta, h_score


def dance_duration(delta_store: float, base: float = 1.0, gain: float = 1.0, limit: float = 10.0) -> float:
    """
    Parent‑A auxiliary function – maps Δstore to a bounded “dance” duration.
    """
    return max(0.0, min(limit, base + gain * delta_store))


# -------------------- Smoke Test --------------------
if __name__ == "__main__":
    # Initial conditions
    store_val = 5.0
    inflow_vals = [2.0, 1.5]
    outflow_vals = [1.0]

    # Reference feature vector (could be derived from a corpus)
    reference_text = "evidence planning support outcome risk"
    reference_vector = text_to_feature_vector(reference_text)

    # Example incoming text
    incoming_text = "The evidence was verified, planning is solid, but risk remains high."

    # Run a hybrid step
    new_store, delta, h_score = hybrid_step(
        store=store_val,
        inflow=inflow_vals,
        outflow=outflow_vals,
        text=incoming_text,
        reference_vec=reference_vector,
    )

    # Print results (simple sanity check)
    print(f"Old store: {store_val:.2f}")
    print(f"Δstore: {delta:.2f}")
    print(f"New store: {new_store:.2f}")
    print(f"Hybrid score: {h_score:.4f}")
    print(f"Dance duration: {dance_duration(delta):.2f}")

    # Ensure no negative store
    assert new_store >= 0.0, "Store became negative!"

    # Quick sanity for SSIM (should be between -1 and 1)
    txt_vec = text_to_feature_vector(incoming_text)
    ssim_val = _ssim_index(txt_vec, reference_vector)
    assert -1.0 <= ssim_val <= 1.0, "SSIM out of expected bounds"

    sys.exit(0)