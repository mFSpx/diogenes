# DARWIN HAMMER — match 5567, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_rbf_surrogate_m348_s0.py (gen4)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_gliner_zero_s_m66_s1.py (gen4)
# born: 2026-05-30T00:02:52Z

"""Hybrid algorithm combining:
- hybrid_hybrid_hybrid_hybrid_hybrid_rbf_surrogate_m348_s0.py
- hybrid_hybrid_physarum_netw_hybrid_gliner_zero_s_m66_s1.py

Mathematical bridge:
The radial‑basis surrogate produces a Gaussian kernel matrix K where
K_{ij}=exp(-ε²‖x_i-x_j‖²).  In sheaf‑cohomology this matrix can be viewed as a
discrete coboundary operator Δ assigning a weight (dimension) to each
edge of the abstract simplicial complex formed by the data points.
Physarum‑type dynamics treat edge weights as electrical conductances.
We therefore reinterpret K as a conductance matrix and feed it into the
flux‑based update equations of the Physarum model.  The conductance
updates modify the kernel in‑place, yielding a dynamically‑adapted
surrogate that reflects both geometric similarity (Gaussian kernel) and
resource‑flow dynamics (flux, conductance decay).  The resulting hybrid
provides label‑scoring that couples spatial similarity with a
physically‑motivated diffusion process."""
import math
import random
import sys
import pathlib
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """A simple proxy for the righting time of a body."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck lever must be positive")
    return b * (m.mass ** k) * neck_lever


def morph_state_vector(m: Morphology) -> np.ndarray:
    """Combine morphological descriptors into a state vector."""
    sph = sphericity_index(m.length, m.width, m.height)
    flat = flatness_index(m.length, m.width, m.height)
    rt = righting_time_index(m)
    return np.array([sph, flat, rt], dtype=float)


# ----------------------------------------------------------------------
# Physarum‑style flux dynamics from Parent B
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 1.0, gain: float = 1.0,
                       decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def label_extraction(text: str, labels: List[str]) -> List[Tuple[int, int, str]]:
    """Return spans (start, end, label) for each occurrence of a label."""
    spans = []
    for label in labels:
        pattern = label.replace(" / ", " ").replace("-", " ")
        regex = re.compile(r"(?<!\w)" + re.escape(pattern) + r"(?!\w)")
        for m in regex.finditer(text):
            spans.append((m.start(), m.end(), label))
    return spans


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def build_coboundary_operator(points: np.ndarray,
                              epsilon: float = 1.0) -> np.ndarray:
    """
    Construct the Gaussian kernel matrix K_{ij}=exp(-ε²‖x_i-x_j‖²).
    Interpreted as a coboundary operator Δ whose entries serve as
    initial conductances for the Physarum dynamics.
    """
    n = points.shape[0]
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            r = np.linalg.norm(points[i] - points[j])
            K[i, j] = gaussian(r, epsilon)
    return K


def update_conductance_matrix(K: np.ndarray,
                              q_matrix: np.ndarray,
                              dt: float = 1.0,
                              gain: float = 1.0,
                              decay: float = 0.05) -> np.ndarray:
    """
    Apply the Physarum update rule element‑wise:
        K_ij ← max(0, K_ij + dt·(gain·|q_ij| - decay·K_ij))
    where q_ij is the flux on edge (i,j).
    """
    if K.shape != q_matrix.shape:
        raise ValueError("K and q_matrix must have the same shape")
    updated = np.vectorize(update_conductance)(
        K, q_matrix, dt=dt, gain=gain, decay=decay)
    return updated


def hybrid_label_scoring(text: str,
                         labels: List[str],
                         points: np.ndarray,
                         morphology: Morphology,
                         epsilon: float = 1.0,
                         dt: float = 1.0) -> List[Tuple[int, int, str, float]]:
    """
    End‑to‑end hybrid pipeline:

    1. Extract label spans from the raw text.
    2. Build a Gaussian kernel (coboundary) from the supplied points.
    3. Derive a pressure value from the morphological state vector.
    4. Compute fluxes on the fully connected graph using the pressure.
    5. Update the conductance matrix with Physarum dynamics.
    6. Aggregate the updated conductances into a score for each span.

    Returns a list of (start, end, label, score).
    """
    # 1. label spans
    spans = label_extraction(text, labels)
    if not spans:
        return []

    # 2. kernel as conductance matrix
    K = build_coboundary_operator(points, epsilon)

    # 3. pressure derived from morphology (simple linear mapping)
    state_vec = morph_state_vector(morphology)
    pressure = float(state_vec.mean())  # scalar pressure for all nodes

    # 4. edge lengths (Euclidean distances)
    n = points.shape[0]
    edge_lengths = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            edge_lengths[i, j] = np.linalg.norm(points[i] - points[j])

    # 5. flux matrix (pressure_a = pressure, pressure_b = 0)
    q_matrix = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            q_matrix[i, j] = flux(K[i, j], edge_lengths[i, j], pressure, 0.0)

    # 6. update conductances
    K_updated = update_conductance_matrix(K, q_matrix, dt=dt)

    # 7. Score each span by summing outgoing conductances of its node
    #    (we map span index ↔ point index by order)
    scores = []
    for idx, (start, end, label) in enumerate(spans):
        if idx >= n:
            # safety: more spans than points – use average conductance
            score = K_updated.mean()
        else:
            score = K_updated[idx].sum()
        scores.append((start, end, label, score))
    return scores


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic data
    sample_text = "The quick brown fox jumps over the lazy dog. " \
                  "A quick blue hare hops under the active cat."
    sample_labels = ["quick", "lazy dog", "blue hare", "active cat"]

    # Create a point for each label span (random 2‑D coordinates)
    random.seed(42)
    np.random.seed(42)
    points = np.random.rand(len(sample_labels), 2) * 10.0

    # Dummy morphology
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # Run hybrid pipeline
    results = hybrid_label_scoring(
        text=sample_text,
        labels=sample_labels,
        points=points,
        morphology=morph,
        epsilon=0.5,
        dt=0.8,
    )

    for start, end, label, score in results:
        snippet = sample_text[start:end]
        print(f"Label: '{label}' ({snippet}) -> score: {score:.4f}")