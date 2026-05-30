# DARWIN HAMMER — match 1217, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1091_s5.py (gen4)
# parent_b: hybrid_hybrid_fisher_locali_jepa_energy_m52_s1.py (gen2)
# born: 2026-05-29T23:34:28Z

"""Hybrid Decision‑Hygiene ↔ Fisher‑Information Engine
===================================================

This module fuses the two parent algorithms *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1091_s5.py*
and *hybrid_hybrid_fisher_locali_jepa_energy_m52_s1.py*.

Mathematical bridge
-------------------

* **Parent A** supplies a hygiene feature vector **f** ∈ ℝᵏ obtained by counting
  regex‑based evidence tokens in a text.
* **Parent B** supplies a Fisher‑information scalar for each geometric edge
  *eᵢ* = (pᵢ,qᵢ,zᵢ) via the Fisher score  

  .. math:: ϕ_i = \\frac{\\bigl(\\partial_θ\\,g(θ_i)\\bigr)^2}{g(θ_i)},

  where *g* is a Gaussian beam centred at 0 with unit width and  

  .. math::  θ_i = \\operatorname{atan2}(q_{i,y}\\! -\\! p_{i,y},
                                      q_{i,x}\\! -\\! p_{i,x}).

* The Bayesian edge‑belief of Parent B yields an expected length  

  .. math::  L_i = P(e_i\\mid\\text{data})\\,\\|p_i-q_i\\|,

  with  

  .. math::  P(e_i\\mid\\text{data}) = 
          \\frac{\\pi\\,\\ell_i}{\\pi\\,\\ell_i + (1-\\pi)\\,\\beta},
  \\quad \\ell_i = \\frac{1}{z_i+\\varepsilon}.

The hybrid combines the two structures by **modulating the hygiene
feature vector with a normalized Fisher‑information‑weighted edge‑length
vector**:

.. math::  \\mathbf{h}=\\bigl(\\frac{\\mathbf{L}}{\\|\\mathbf{L}\\|_1}\\bigr)
           \\odot \\bigl(\\frac{\\boldsymbol{ϕ}}{\\|\\boldsymbol{ϕ}\\|_1}\\bigr)
           \\odot \\mathbf{f}.

The resulting hybrid vector *h* is fed to a simple Normalized Least‑Mean‑Squares
(NLMS) predictor, producing a scalar decision score.  The three public
functions below expose the full pipeline, a batch variant and the low‑level
feature extractor.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Sequence, Dict, Optional

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regex‑based hygiene feature extraction
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|validated?|confirmed?|proof|certified?)\b",
    flags=re.IGNORECASE,
)

def extract_features(text: str, categories: Optional[Sequence[str]] = None) -> np.ndarray:
    """
    Count occurrences of each regex category in *text*.
    If *categories* is None a single count of evidence‑type words is returned.
    """
    if categories is None:
        count = len(EVIDENCE_RE.findall(text))
        return np.array([count], dtype=float)

    counts = []
    for pat in categories:
        regex = re.compile(pat, flags=re.IGNORECASE)
        counts.append(len(regex.findall(text)))
    return np.array(counts, dtype=float)


# ----------------------------------------------------------------------
# Parent B – Fisher‑information utilities (Gaussian beam & Fisher score)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float = 0.0, width: float = 1.0) -> float:
    """Gaussian beam g(θ) = exp(-½ ((θ‑center)/width)²)."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam:
    φ = (∂_θ g)² / g .
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Bayesian edge belief (Parent B) and expected length computation
# ----------------------------------------------------------------------
def edge_posterior(
    impedance: float,
    prior: float = 0.5,
    beta: float = 0.01,
    eps: float = 1e-12,
) -> float:
    """Posterior probability that an edge is true."""
    likelihood = 1.0 / (impedance + eps)
    numerator = prior * likelihood
    denominator = numerator + (1.0 - prior) * beta
    return numerator / denominator


def expected_edge_lengths(
    edges: List[Tuple[Tuple[float, float], Tuple[float, float], float]],
    prior: float = 0.5,
    beta: float = 0.01,
) -> np.ndarray:
    """
    Compute the vector L of expected edge lengths.
    Each edge is ((x1,y1), (x2,y2), impedance).
    """
    L = []
    for p, q, z in edges:
        prob = edge_posterior(z, prior, beta)
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        L.append(prob * length)
    return np.array(L, dtype=float)


# ----------------------------------------------------------------------
# Fisher information per edge (derived from geometric direction)
# ----------------------------------------------------------------------
def edge_fisher_information(
    edges: List[Tuple[Tuple[float, float], Tuple[float, float], float]],
    center: float = 0.0,
    width: float = 1.0,
) -> np.ndarray:
    """
    Compute Fisher score for each edge using its polar angle.
    """
    phi = []
    for p, q, _ in edges:
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        theta = math.atan2(dy, dx)  # range (-π, π)
        phi.append(fisher_score(theta, center, width))
    return np.array(phi, dtype=float)


# ----------------------------------------------------------------------
# Normalized Least‑Mean‑Squares predictor
# ----------------------------------------------------------------------
class NLMSPredictor:
    """
    Simple NLMS predictor:
        ŷ = wᵀ x
        w ← w + μ (ŷ – y) x / (‖x‖² + δ)
    For our hybrid we only need the forward pass; the update step is kept
    for completeness.
    """

    def __init__(self, dim: int, mu: float = 0.1, delta: float = 1e-6):
        self.w = np.zeros(dim, dtype=float)
        self.mu = mu
        self.delta = delta

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x))

    def update(self, x: np.ndarray, target: float) -> None:
        y_hat = self.predict(x)
        error = target - y_hat
        norm_sq = np.dot(x, x) + self.delta
        self.w += (self.mu * error / norm_sq) * x


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_vector(
    text: str,
    edges: List[Tuple[Tuple[float, float], Tuple[float, float], float]],
    categories: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Build the hybrid feature vector **h** = (L̂ ⊙ φ̂ ⊙ f).

    * L̂ – L normalized by its ℓ₁ norm.
    * φ̂ – Fisher scores normalized by their ℓ₁ norm.
    * f   – hygiene feature vector (may be multi‑dimensional).
    """
    # 1. Hygiene features
    f = extract_features(text, categories)

    # 2. Expected edge lengths
    L = expected_edge_lengths(edges)
    if L.sum() == 0:
        L_norm = np.zeros_like(L)
    else:
        L_norm = L / L.sum()

    # 3. Fisher information per edge
    phi = edge_fisher_information(edges)
    if phi.sum() == 0:
        phi_norm = np.zeros_like(phi)
    else:
        phi_norm = phi / phi.sum()

    # 4. Align dimensions: repeat f to match edge count if necessary
    if f.size == 1:
        f_rep = np.full(L_norm.shape, f.item(), dtype=float)
    else:
        # Truncate or pad to the edge count
        f_rep = np.resize(f, L_norm.shape)

    # 5. Element‑wise multiplication
    h = L_norm * phi_norm * f_rep
    return h


def hybrid_score(
    text: str,
    edges: List[Tuple[Tuple[float, float], Tuple[float, float], float]],
    categories: Optional[Sequence[str]] = None,
    target: float = 1.0,
) -> float:
    """
    Compute a scalar decision score for *text* using the hybrid vector and an NLMS predictor.
    The predictor is instantiated on‑the‑fly; in a real system it would be persisted.
    """
    h = hybrid_vector(text, edges, categories)

    # Initialise NLMS with the correct dimension
    predictor = NLMSPredictor(dim=h.size)

    # One‑step update using a synthetic target (here we use *target* as the desired output)
    predictor.update(h, target)

    # Return the current prediction after the update
    return predictor.predict(h)


def batch_hybrid_scores(
    texts: List[str],
    edges_list: List[List[Tuple[Tuple[float, float], Tuple[float, float], float]]],
    categories: Optional[Sequence[str]] = None,
) -> List[float]:
    """
    Apply :func:`hybrid_score` to a batch of texts / edge collections.
    The NLMS predictor is shared across the batch to accumulate learning.
    """
    if len(texts) != len(edges_list):
        raise ValueError("texts and edges_list must have the same length")

    # Determine a common dimension (max edge count across the batch)
    max_len = max(len(e) for e in edges_list)
    predictor = NLMSPredictor(dim=max_len)

    scores = []
    for txt, edges in zip(texts, edges_list):
        h = hybrid_vector(txt, edges, categories)
        # Pad h to max_len if needed (NLMS expects fixed dimension)
        if h.size < max_len:
            h = np.pad(h, (0, max_len - h.size), mode="constant")
        predictor.update(h, target=1.0)
        scores.append(predictor.predict(h))
    return scores


# ----------------------------------------------------------------------
# Minimal datetime parsing helper (kept from Parent B for completeness)
# ----------------------------------------------------------------------
def parse_loose_datetime(raw: str) -> Optional[datetime]:
    """
    Attempt to parse *raw* into a timezone‑aware ``datetime``.
    Returns ``None`` on failure.
    """
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


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The evidence was verified and the proof was confirmed. "
        "Additional verification was required."
    )

    # Define a tiny geometric graph (3 edges)
    sample_edges = [
        ((0.0, 0.0), (1.0, 0.0), 2.0),   # horizontal edge, impedance 2
        ((1.0, 0.0), (1.0, 1.0), 5.0),   # vertical edge, impedance 5
        ((1.0, 1.0), (0.0, 0.0), 1.0),   # diagonal edge, impedance 1
    ]

    # Compute a single hybrid score
    score = hybrid_score(sample_text, sample_edges)
    print(f"Hybrid decision score: {score:.6f}")

    # Batch test
    texts = [sample_text, "No relevant evidence here.", "Proof is lacking."]
    edges_batch = [sample_edges, sample_edges, sample_edges]
    batch_scores = batch_hybrid_scores(texts, edges_batch)
    print("Batch scores:", ["{:.6f}".format(s) for s in batch_scores])