# DARWIN HAMMER — match 821, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_nlms_omni_cha_m142_s0.py (gen3)
# parent_b: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s5.py (gen3)
# born: 2026-05-29T23:31:05Z

"""Hybrid Endpoint‑NLMS & Cognitive‑Risk Decision Model

Parents
-------
* **Parent A** – `hybrid_hybrid_endpoint_circ_hybrid_workshare_all_m12_s1.py`  
  Provides endpoint health tracking, morphological indices and a normalized least‑mean‑squares
  (NLMS) adaptation whose step‑size μ is modulated by a health‑score and the day of week.

* **Parent B** – `hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s5.py`  
  Extracts a scalar “cognitive‑risk” `c_i` from free‑form text via weighted regex features.
  The risk is interpreted as a *privacy‑load* `p_i`. Together with a spatial load `d_i`
  (e.g. haversine distance) each entity obtains a resource vector `r_i = [d_i, p_i]`
  that must satisfy linear budgets.

Mathematical Bridge
-------------------
The bridge is the *privacy‑load* dimension `p_i`.  
In the hybrid system each endpoint `e_i` carries:

* a health‑score `h_i` (from failure statistics and morphology) – used to scale the NLMS
  step‑size `μ_i = μ₀·h_i·day_factor`.
* a spatial load `d_i` (distance to a reference point) – used as the NLMS input vector.
* a privacy‑load `p_i = c_i` (cognitive‑risk extracted from associated text) – incorporated
  as an additional dimension of the NLMS error signal, allowing the adaptive filter to
  minimise a joint error that respects both spatial and privacy budgets.

The hybrid algorithm therefore solves a constrained NLMS problem:


w_{k+1} = w_k + μ_i * (e_k / (δ + ‖x_k‖²)) * x_k
subject to   Aᵀ·x ≤ [spatial_budget, privacy_budget]


where `x` is the binary selection vector of endpoints, `A` stacks `[d_i, p_i]`,
and `e_k` is the instantaneous error between desired and actual resource usage.

The implementation below materialises this fusion with three core functions:
`compute_health_score`, `nlms_step`, and `select_endpoints`.  A smoke test
exercises the whole pipeline on synthetic data."""

import math
import random
import sys
from datetime import date
from pathlib import Path
from typing import List, Tuple

import numpy as np
import re

# ----------------------------------------------------------------------
# Shared constants and utilities
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
DAY_FACTOR = np.array([0.9, 1.0, 1.1, 1.0, 0.9, 0.8, 0.7])  # Mon‑Sun scaling


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Return an integer 0‑6 representing the day of week (Mon=0 … Sun=6)."""
    return (date(year, month, day).weekday() + 1) % 7


# ----------------------------------------------------------------------
# Parent A – endpoint health & morphology
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple circuit‑breaker tracking failures."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        return not self.open


class Morphology:
    """Geometric description of an endpoint device."""

    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    """Dimensionless index: volume^(1/3) / max dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Dimensionless index: min dimension / max dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return min(length, width, height) / max(length, width, height)


def compute_health_score(cb: EndpointCircuitBreaker, morph: Morphology) -> float:
    """
    Combine circuit‑breaker state and morphology into a scalar health score `h_i`
    in (0, 1].  Lower failure count and more spherical shapes yield higher scores.
    """
    # Base score from circuit‑breaker: open => 0.2, otherwise 1.0
    cb_score = 0.2 if not cb.allow() else 1.0

    # Morphology contribution: weighted geometric mean of sphericity and flatness
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flat = flatness_index(morph.length, morph.width, morph.height)
    morph_score = (sph * flat) ** 0.5  # range (0,1]

    # Combine (geometric mean) and clamp
    health = (cb_score * morph_score) ** 0.5
    return max(min(health, 1.0), 0.01)  # avoid zero


# ----------------------------------------------------------------------
# Parent B – cognitive‑risk extraction
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)

POSITIVE_WEIGHT = 1.2
NEGATIVE_WEIGHT = -0.8


def cognitive_risk_score(text: str) -> float:
    """
    Compute a scalar privacy‑load `p_i` from free‑form text.
    Positive cues (evidence‑related) increase risk, planning cues decrease it.
    """
    pos = len(EVIDENCE_RE.findall(text))
    neg = len(PLANNING_RE.findall(text))
    return POSITIVE_WEIGHT * pos + NEGATIVE_WEIGHT * neg


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def nlms_step(
    w: np.ndarray,
    x: np.ndarray,
    d: float,
    mu: float,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """
    One NLMS adaptation step.

    Parameters
    ----------
    w : np.ndarray
        Current weight vector (shape (n,)).
    x : np.ndarray
        Input vector (shape (n,)).
    d : float
        Desired scalar response.
    mu : float
        Step‑size (scaled by health score and day factor).
    epsilon : float
        Regularisation term to avoid division by zero.

    Returns
    -------
    w_new : np.ndarray
        Updated weight vector.
    e : float
        Instantaneous error after the update.
    """
    y = np.dot(w, x)
    e = d - y
    norm_x = np.dot(x, x) + epsilon
    w_new = w + (mu * e / norm_x) * x
    return w_new, e


def build_resource_matrix(
    distances: List[float], texts: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the resource matrix A (shape (2, N)) where the first row holds
    spatial loads `d_i` and the second row holds privacy loads `p_i`.

    Returns
    -------
    A : np.ndarray
        2‑by‑N matrix of resources.
    p_vec : np.ndarray
        Vector of privacy loads (used as desired responses in NLMS).
    """
    d_vec = np.array(distances, dtype=float)
    p_vec = np.array([cognitive_risk_score(t) for t in texts], dtype=float)
    A = np.vstack([d_vec, p_vec])
    return A, p_vec


def select_endpoints(
    health_scores: List[float],
    A: np.ndarray,
    spatial_budget: float,
    privacy_budget: float,
    max_iters: int = 200,
) -> np.ndarray:
    """
    Hybrid selector that simultaneously:

    1. Performs NLMS adaptation of a weight vector `w` that predicts the
       combined resource usage.
    2. Enforces linear budget constraints `A.T @ x <= [spatial_budget, privacy_budget]`
       by projecting the binary selection vector `x` after each iteration.

    Parameters
    ----------
    health_scores : List[float]
        Health scores `h_i` for each endpoint (used to scale μ).
    A : np.ndarray
        2‑by‑N resource matrix.
    spatial_budget, privacy_budget : float
        Upper limits for the two resources.
    max_iters : int
        Maximum number of adaptation iterations.

    Returns
    -------
    x : np.ndarray
        Binary selection vector of shape (N,).
    """
    N = A.shape[1]
    # Initialise NLMS weight vector (size N) and selection vector (binary)
    w = np.zeros(N)
    x = np.ones(N, dtype=int)  # start with all endpoints selected

    # Desired total resource usage: aim for a fraction (e.g. 80%) of budgets
    desired = np.array([0.8 * spatial_budget, 0.8 * privacy_budget])

    # Pre‑compute day factor for today
    today_factor = DAY_FACTOR[date.today().weekday()]

    for _ in range(max_iters):
        # Compute current resource usage
        usage = A @ x

        # If within budgets, break early
        if np.all(usage <= desired):
            break

        # Choose a random endpoint to update (stochastic NLMS)
        i = random.randrange(N)

        # Input vector for NLMS is the resource column of endpoint i
        x_i = A[:, i]  # shape (2,)

        # Desired response is the corresponding component of `desired`
        d_i = desired @ (x_i / np.linalg.norm(x_i) ** 2)  # scalar projection

        # Scale step‑size by health and day factor
        mu_i = 0.1 * health_scores[i] * today_factor  # base μ = 0.1

        # Perform NLMS update on weight w_i (scalar weight per endpoint)
        w_i, _ = nlms_step(np.array([w[i]]), np.array([1.0]), d_i, mu_i)
        w[i] = w_i[0]

        # Update selection based on weight magnitude:
        # high weight → keep selected, low weight → possibly drop
        threshold = np.median(w)  # simple adaptive threshold
        x[i] = 1 if w[i] >= threshold else 0

        # Project onto feasible set (simple clipping)
        # Ensure we never exceed budgets by turning off highest‑load endpoints
        while np.any(A @ x > np.array([spatial_budget, privacy_budget])):
            # Find endpoint with largest contribution to the violated resource
            viol = (A @ x) - np.array([spatial_budget, privacy_budget])
            # Pick the resource with greatest excess
            res_idx = int(np.argmax(viol))
            # Among selected endpoints, pick the one with max load on that resource
            candidates = np.where(x == 1)[0]
            if candidates.size == 0:
                break
            loads = A[res_idx, candidates]
            drop_idx = candidates[np.argmax(loads)]
            x[drop_idx] = 0

    return x


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic endpoints
    num = 10
    endpoints = []
    healths = []
    distances = []
    texts = []

    for i in range(num):
        # Random morphology
        morph = Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(1.0, 5.0),
        )
        # Random circuit‑breaker state
        cb = EndpointCircuitBreaker(failure_threshold=3)
        # Simulate some failures
        for _ in range(random.randint(0, 4)):
            if random.random() < 0.3:
                cb.record_failure()
            else:
                cb.record_success()
        h = compute_health_score(cb, morph)
        healths.append(h)

        # Random spatial load (distance in km)
        distances.append(random.uniform(10, 500))

        # Random textual evidence
        evidence_words = ["evidence", "verified", "source", "log"]
        planning_words = ["plan", "schedule", "budget", "timeline"]
        txt = " ".join(
            random.choices(evidence_words + planning_words, k=random.randint(5, 12))
        )
        texts.append(txt)

    # Budgets
    spatial_budget = 1500.0
    privacy_budget = 5.0

    # Build resource matrix
    A, p_vec = build_resource_matrix(distances, texts)

    # Run hybrid selector
    selection = select_endpoints(healths, A, spatial_budget, privacy_budget)

    # Report
    print("Selected endpoints (index):", np.where(selection == 1)[0].tolist())
    print("Total spatial load:", float(A[0] @ selection))
    print("Total privacy load:", float(A[1] @ selection))
    print("Health scores of selected:", [round(healths[i], 3) for i in np.where(selection == 1)[0]])