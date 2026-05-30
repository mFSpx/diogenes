# DARWIN HAMMER — match 1915, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_cockpi_hybrid_hybrid_decisi_m516_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_dense_associa_m1179_s1.py (gen4)
# born: 2026-05-29T23:39:48Z

import numpy as np
import re
import math
import datetime
from typing import Dict, Tuple, List

# ----------------------------------------------------------------------
# Regular expressions for linguistic style matching (LSM)
# ----------------------------------------------------------------------
_EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|sha256|"
    r"screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
_PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|"
    r"criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
_DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
    r"before i|first|after|review)\b",
    re.I,
)
_SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|"
    r"support|help|community|team|handoff|delegate)\b",
    re.I,
)
_BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact)\b", re.I)

# ----------------------------------------------------------------------
# Feature order and weight vectors (positive & negative)
# ----------------------------------------------------------------------
_FEATURE_ORDER: List[str] = [
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

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.float64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.float64)

# ----------------------------------------------------------------------
# Helper: safe division
# ----------------------------------------------------------------------
def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, guarding against zero denominator."""
    return numerator / denominator if denominator != 0 else 0.0

# ----------------------------------------------------------------------
# Linguistic Style Matching vector
# ----------------------------------------------------------------------
def lsm_vector(text: str) -> Dict[str, float]:
    """
    Compute a normalized frequency vector for the five LSM categories.
    Missing categories are reported with a frequency of 0.0.
    """
    matches = {
        "evidence": _EVIDENCE_RE.findall(text),
        "planning": _PLANNING_RE.findall(text),
        "delay": _DELAY_RE.findall(text),
        "support": _SUPPORT_RE.findall(text),
        "boundary": _BOUNDARY_RE.findall(text),
    }

    total = sum(len(v) for v in matches.values())
    # Guard against empty text – return a zero‑vector.
    if total == 0:
        return {cat: 0.0 for cat in matches}

    return {cat: _safe_divide(len(words), total) for cat, words in matches.items()}


# ----------------------------------------------------------------------
# Feature weighting that respects the full feature order
# ----------------------------------------------------------------------
def feature_weights(text: str) -> np.ndarray:
    """
    Produce a trust‑weighted feature vector by:
      1. Computing the LSM frequencies.
      2. Scaling the positive weight vector by the LSM frequencies (where defined).
      3. Adding the negative weight vector (which is independent of LSM).
    The result is a dense 9‑dimensional float array.
    """
    lsm = lsm_vector(text)

    # Build a full frequency vector aligned with _FEATURE_ORDER.
    freq_vec = np.array(
        [_safe_divide(lsm.get(cat, 0.0), 1.0) for cat in _FEATURE_ORDER], dtype=np.float64
    )

    # Positive contribution modulated by LSM, negative contribution added directly.
    weighted = _POSITIVE_WEIGHTS * freq_vec + _NEGATIVE_WEIGHTS
    return weighted


# ----------------------------------------------------------------------
# Pheromone signal handling (decay based on half‑life)
# ----------------------------------------------------------------------
class PheromoneBank:
    """
    Simple in‑memory store for pheromone signals with exponential decay.
    Thread‑safety is not required for the current single‑threaded usage.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, float]] = {}

    def signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        """
        Register or update a pheromone signal.
        If the key exists, decay the previous value according to its half‑life
        before adding the new contribution.
        """
        now = datetime.datetime.utcnow()

        if surface_key not in self._store:
            self._store[surface_key] = {
                "kind": signal_kind,
                "value": signal_value,
                "half_life": half_life_seconds,
                "timestamp": now,
            }
            return signal_value

        entry = self._store[surface_key]
        elapsed = (now - entry["timestamp"]).total_seconds()
        decay_factor = math.exp(-elapsed * math.log(2) / entry["half_life"])
        decayed = entry["value"] * decay_factor

        # Combine decayed old value with new contribution.
        entry["value"] = decayed + signal_value
        entry["timestamp"] = now
        entry["half_life"] = half_life_seconds  # allow dynamic update
        entry["kind"] = signal_kind
        return entry["value"]


# ----------------------------------------------------------------------
# Non‑linear B‑spline transformation of a memory matrix
# ----------------------------------------------------------------------
def _b_spline_basis(x: float, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Evaluate all B‑spline basis functions of a given degree at point x.
    This implementation follows the Cox‑de Boor recursion formula.
    """
    n = len(knots) - degree - 1
    # Initialise zero‑order basis.
    N = np.zeros(n + degree)
    for i in range(n + degree):
        N[i] = 1.0 if knots[i] <= x < knots[i + 1] else 0.0
    # Recursive construction.
    for d in range(1, degree + 1):
        for i in range(n + degree - d):
            left = (
                (x - knots[i]) / (knots[i + d] - knots[i]) * N[i]
                if knots[i + d] != knots[i]
                else 0.0
            )
            right = (
                (knots[i + d + 1] - x) / (knots[i + d + 1] - knots[i + 1]) * N[i + 1]
                if knots[i + d + 1] != knots[i + 1]
                else 0.0
            )
            N[i] = left + right
    return N[: n]  # only the first n basis functions are non‑zero


def nonlinear_transformation(memory_matrix: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Apply a B‑spline based non‑linear transformation to each element of
    ``memory_matrix``.  For each element x we compute the vector of basis
    values B(x) and then return the dot product B(x)·coeffs.
    The function works for any 2‑D matrix and any coefficient vector whose
    length matches the number of basis functions.
    """
    if memory_matrix.ndim != 2:
        raise ValueError("memory_matrix must be 2‑dimensional")

    # Create a uniform knot vector that spans the data range.
    min_val, max_val = memory_matrix.min(), memory_matrix.max()
    degree = 3
    # Ensure enough knots: len(knots) = len(coeffs) + degree + 1
    num_knots = len(coeffs) + degree + 1
    knots = np.linspace(min_val, max_val, num_knots)

    transformed = np.empty_like(memory_matrix, dtype=np.float64)
    it = np.nditer(memory_matrix, flags=["multi_index"])
    while not it.finished:
        x = float(it[0])
        basis = _b_spline_basis(x, knots, degree=degree)
        transformed[it.multi_index] = float(np.dot(basis, coeffs))
        it.iternext()
    return transformed


# ----------------------------------------------------------------------
# Core hybrid operation
# ----------------------------------------------------------------------
_pheromone_bank = PheromoneBank()
_memory_matrix = np.random.rand(10, 10)          # Example memory matrix
_b_spline_coeffs = np.random.rand(7)             # 7 coefficients → 7 basis functions


def hybrid_operation(text: str) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Execute the fused algorithm:

    1. Compute trust‑weighted feature vector from the input text.
    2. Update/obtain a pheromone signal.
    3. Apply a deep non‑linear B‑spline transformation to the memory matrix.

    Returns
    -------
    weights : np.ndarray
        The 9‑dimensional feature vector.
    signal_value : float
        Current pheromone signal after decay and addition.
    transformed_matrix : np.ndarray
        The memory matrix after B‑spline transformation.
    """
    weights = feature_weights(text)
    signal_value = _pheromone_bank.signal(
        surface_key="global_surface",
        signal_kind="trust",
        signal_value=1.0,
        half_life_seconds=3600,
    )
    transformed_matrix = nonlinear_transformation(_memory_matrix, _b_spline_coeffs)
    return weights, signal_value, transformed_matrix


# ----------------------------------------------------------------------
# Simple demo when run as a script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_text = "The audit confirmed evidence and a solid planning phase, but we need more support."
    w, s, tm = hybrid_operation(demo_text)
    print("Feature weights:", w)
    print("Pheromone signal:", s)
    print("Transformed memory matrix shape:", tm.shape)