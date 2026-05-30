# DARWIN HAMMER — match 1217, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1091_s5.py (gen4)
# parent_b: hybrid_hybrid_fisher_locali_jepa_energy_m52_s1.py (gen2)
# born: 2026-05-29T23:34:28Z

"""
Hybrid Decision‑Hygiene + Fisher‑JEPA Engine
==========================================

This module fuses the two parent algorithms:

* **Parent A** – extracts a categorical regex‑count vector ``f`` from text,
  builds a Bayesian edge‑belief model producing an expected‑length vector
  ``L`` and finally feeds the weighted hygiene vector ``h = (L/‖L‖₁) ⊙ f``
  to an NLMS predictor.

* **Parent B** – parses loose ISO‑like datetimes, evaluates the Fisher
  information density (``fisher_score``) for each candidate and uses a
  JEPA‑style linear embedding to map observations into a representation
  space.  The energy of the system is the weighted squared prediction error
  in that space.

**Mathematical bridge**

1. The hygiene vector ``h`` (size *k*) is interpreted as a *query* that
   selects a sub‑space of the JEPA embedding matrix ``R ∈ ℝ^{d×k}``.
2. Each parsed datetime ``τ_i`` is converted to an angular coordinate
   ``θ_i`` (seconds‑of‑day normalized to ``[0, 2π)``).  The Fisher score
   ``φ_i = fisher_score(θ_i, μ, σ)`` measures the information density of
   that candidate; the scores are normalised to a probability vector
   ``w = φ / Σ φ``.
3. The JEPA representation of a candidate is ``r_i = R @ h`` (the same for
   all candidates because the embedding is linear).  The predicted
   representation is the weighted average ``\hat r = Σ w_i r_i = r``.
4. The **energy** is defined as the Fisher‑weighted mean‑squared error
   between each candidate representation and the prediction:

   .. math::
       E = \\sum_i w_i \\|r_i - \\hat r\\|_2^2.

   Because ``r_i = \\hat r`` for a linear embedding, we instead inject the
   edge‑belief vector ``L`` as a second linear map ``Q ∈ ℝ^{d×m}`` (``m`` is
   the number of graph edges).  The final representation becomes

   .. math::
       r_i = R h + Q L.

   This couples the two parent topologies: the hygiene‑edge product
   ``h`` and the Bayesian edge expectation ``L`` jointly influence the JEPA
   space, while the Fisher weights modulate their contribution to the
   energy.

The public API consists of three demonstration functions:

* ``hybrid_vector`` – builds ``h`` from raw text and a graph description.
* ``fisher_weighted_energy`` – computes the JEPA‑energy using datetime
  candidates extracted from a path.
* ``hybrid_decision`` – returns a scalar decision score (NLMS output) together
  with the energy and a certainty dictionary.

All computations rely only on the Python standard library and ``numpy``.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
import re
from typing import List, Tuple, Dict, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regex‑based hygiene feature extraction
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|validation|proof|confirm|assert|test)\b", re.I
)
QUESTION_RE = re.compile(r"\b(?:why|how|what|when|where|who)\b", re.I)
NEGATION_RE = re.compile(r"\b(?:no|not|never|none|without)\b", re.I)

def extract_hygiene_features(text: str) -> np.ndarray:
    """Return a 3‑dim vector of regex counts: [evidence, question, negation]."""
    ev = len(EVIDENCE_RE.findall(text))
    qs = len(QUESTION_RE.findall(text))
    ng = len(NEGATION_RE.findall(text))
    return np.array([ev, qs, ng], dtype=float)

# ----------------------------------------------------------------------
# Parent A – Bayesian edge belief and expected length vector
# ----------------------------------------------------------------------
def expected_edge_lengths(
    edges: Sequence[Tuple[Tuple[float, float], Tuple[float, float], int]],
    prior: float = 0.5,
    false_pos: float = 0.01,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    edges: iterable of ((x1,y1), (x2,y2), impedance)
    Returns L_i = P(e_i|data) * Euclidean length of edge i.
    """
    L = []
    for (p, q, z) in edges:
        length = math.hypot(p[0] - q[0], p[1] - q[1])
        likelihood = 1.0 / (z + eps)
        posterior = (prior * likelihood) / (
            prior * likelihood + (1.0 - prior) * false_pos
        )
        L.append(posterior * length)
    return np.array(L, dtype=float)


def hybrid_vector(text: str, edges: Sequence[Tuple[Tuple[float, float], Tuple[float, float], int]]) -> np.ndarray:
    """
    Core hybrid: h = (L / ||L||_1) ⊙ f
    """
    f = extract_hygiene_features(text)
    L = expected_edge_lengths(edges)
    if L.sum() == 0:
        L_norm = np.zeros_like(L)
    else:
        L_norm = L / L.sum()
    # Broadcast L_norm to match f length (take first k elements or pad)
    k = f.shape[0]
    L_expanded = np.pad(L_norm, (0, max(0, k - L_norm.size)), mode='constant')[:k]
    h = L_expanded * f
    return h

# ----------------------------------------------------------------------
# Parent B – Fisher‑Krampus information density
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def parse_loose_datetime(raw: str) -> datetime | None:
    text = raw.strip().strip("'\"`[]()")
    if not text:
        return None
    try:
        val = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    except ValueError:
        return None

def datetime_to_angle(dt: datetime) -> float:
    """
    Convert a datetime to an angle in radians based on seconds of day.
    """
    secs = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    return (secs / 86400.0) * 2.0 * math.pi

def fisher_weights(datetimes: Sequence[datetime]) -> np.ndarray:
    """
    Compute normalized Fisher scores for a collection of datetimes.
    """
    if not datetimes:
        return np.array([], dtype=float)
    angles = np.array([datetime_to_angle(dt) for dt in datetimes])
    mu = angles.mean()
    sigma = angles.std() if angles.std() > 0 else 1.0
    scores = np.array([fisher_score(theta, mu, sigma) for theta in angles])
    total = scores.sum()
    return scores / total if total > 0 else np.full_like(scores, 1.0 / len(scores))

# ----------------------------------------------------------------------
# JEPA‑style linear embedding and energy computation
# ----------------------------------------------------------------------
def build_embedding_matrices(k: int, m: int, d: int = 8, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (R, Q):
    * R ∈ ℝ^{d×k} maps hygiene vector h.
    * Q ∈ ℝ^{d×m} maps edge‑expectation vector L.
    """
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((d, k))
    Q = rng.standard_normal((d, m))
    return R, Q

def representation(h: np.ndarray, L: np.ndarray, R: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Linear JEPA representation: r = R h + Q L
    """
    return R @ h + Q @ L

def fisher_weighted_energy(
    h: np.ndarray,
    L: np.ndarray,
    datetimes: Sequence[datetime],
    R: np.ndarray,
    Q: np.ndarray,
) -> float:
    """
    Compute E = Σ w_i ||r_i - r̂||² where r_i = R h + Q L (identical for all i)
    and w_i are Fisher‑derived probabilities.
    Because r_i is identical, we inject a tiny perturbation proportional to
    the Fisher weight to obtain a non‑zero energy, mimicking JEPA prediction error.
    """
    if not datetimes:
        return 0.0
    w = fisher_weights(datetimes)  # shape (n,)
    base = representation(h, L, R, Q)  # shape (d,)
    # Create perturbed representations
    rng = np.random.default_rng(0)
    perturb = rng.standard_normal((len(datetimes), base.shape[0])) * 1e-3
    reps = base + perturb * w[:, None]  # each row weighted
    # Weighted MSE
    diff = reps - base
    sq = np.sum(diff * diff, axis=1)
    return float(np.dot(w, sq))

# ----------------------------------------------------------------------
# NLMS predictor (Parent B utility)
# ----------------------------------------------------------------------
class NLMSPredictor:
    """
    Normalized Least‑Mean‑Squares adaptive filter.
    For demonstration we train on a single synthetic sample (target = 1.0).
    """
    def __init__(self, dim: int, mu: float = 0.1, eps: float = 1e-8):
        self.w = np.zeros(dim, dtype=float)
        self.mu = mu
        self.eps = eps

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x))

    def adapt(self, x: np.ndarray, target: float = 1.0):
        y = self.predict(x)
        e = target - y
        norm = np.dot(x, x) + self.eps
        self.w += (self.mu / norm) * e * x

# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_decision(
    text: str,
    edges: Sequence[Tuple[Tuple[float, float], Tuple[float, float], int]],
    datetime_strings: Sequence[str],
) -> Dict[str, object]:
    """
    Executes the full hybrid pipeline:
    1. Build h and L.
    2. Parse datetimes and compute Fisher weights.
    3. Build embeddings, compute JEPA energy.
    4. Run NLMS predictor on h to obtain a scalar decision.
    Returns a dictionary with keys:
    * 'score'      – NLMS output
    * 'energy'     – Fisher‑weighted JEPA energy
    * 'certainty' – simple confidence based on score magnitude
    * 'h_vector'   – the hybrid vector h
    """
    # Step 1
    h = hybrid_vector(text, edges)
    L = expected_edge_lengths(edges)

    # Step 2
    dts = [parse_loose_datetime(s) for s in datetime_strings]
    dts = [dt for dt in dts if dt is not None]

    # Step 3
    R, Q = build_embedding_matrices(k=h.shape[0], m=L.shape[0])
    energy = fisher_weighted_energy(h, L, dts, R, Q)

    # Step 4
    nlms = NLMSPredictor(dim=h.shape[0])
    nlms.adapt(h)  # one adaptation step
    score = nlms.predict(h)

    certainty = {
        "high": abs(score) > 2.0,
        "moderate": 0.5 < abs(score) <= 2.0,
        "low": abs(score) <= 0.5,
    }

    return {
        "score": score,
        "energy": energy,
        "certainty": certainty,
        "h_vector": h,
    }

# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def demo_hybrid_vector() -> None:
    txt = "Evidence suggests that the test failed because no verification was performed."
    edges = [((0, 0), (1, 1), 3), ((1, 0), (0, 1), 5)]
    h = hybrid_vector(txt, edges)
    print("Hybrid vector h:", h)

def demo_fisher_energy() -> None:
    txt = "Proof of concept."
    edges = [((0, 0), (2, 2), 2)]
    dt_strs = ["2023-03-15T12:34:56Z", "2022-11-01T08:00:00+00:00", "invalid"]
    h = hybrid_vector(txt, edges)
    L = expected_edge_lengths(edges)
    R, Q = build_embedding_matrices(k=h.shape[0], m=L.shape[0])
    dts = [parse_loose_datetime(s) for s in dt_strs if parse_loose_datetime(s)]
    energy = fisher_weighted_energy(h, L, dts, R, Q)
    print("Fisher‑weighted energy:", energy)

def demo_full_decision() -> None:
    txt = "Why does the evidence not confirm the hypothesis?"
    edges = [((0, 0), (3, 4), 1), ((2, 2), (5, 5), 4)]
    dt_strs = ["2021-01-01T00:00:00Z", "2021-06-30T23:59:59Z"]
    result = hybrid_decision(txt, edges, dt_strs)
    print("Hybrid decision result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: hybrid_vector ===")
    demo_hybrid_vector()
    print("\n=== Demo: fisher_weighted_energy ===")
    demo_fisher_energy()
    print("\n=== Demo: full hybrid decision ===")
    demo_full_decision()