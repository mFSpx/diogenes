# DARWIN HAMMER — match 3325, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_semant_hybrid_hybrid_hybrid_m1352_s1.py (gen5)
# parent_b: hybrid_fractional_hdc_hybrid_endpoint_circ_m119_s0.py (gen2)
# born: 2026-05-29T23:49:10Z

"""
Module fusing hybrid_hybrid_hybrid_semant_hybrid_hybrid_hybrid_m1352_s1.py and 
hybrid_fractional_hdc_hybrid_endpoint_circ_m119_s0.py.
The mathematical bridge between the two structures is the application of 
the semantic neighborhood analysis to the fractional power binding 
in Hyperdimensional Computing (HDC), where the health score from the 
endpoint morphology fusion is used to weight the neighborhood certainty 
matrix in the Bayesian update function.
"""

import numpy as np
import random
import math
import sys
import pathlib
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# In-memory semantic enclave 
# ----------------------------------------------------------------------
_ENCLAVE: dict[str, np.ndarray] = {}

def clear_enclave() -> None:
    """Remove all registered documents."""
    _ENCLAVE.clear()

def register_document(doc_id: str, vector: list[float]) -> None:
    """Store a document vector as a NumPy array for fast linear algebra."""
    _ENCLAVE[doc_id] = np.array(vector, dtype=float)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0.0:
        return 0.0
    return float(np.dot(a, b) / den)

def semantic_neighbors(doc_id: str, k: int = 5) -> list[tuple[str, float]]:
    """Return the k most similar documents (excluding the query)."""
    v = _ENCLAVE[doc_id]
    sims = [(d, _cosine(v, w)) for d, w in _ENCLAVE.items() if d != doc_id]
    sims.sort(key=lambda x: (-x[1], x[0]))
    return sims[:k]

# ----------------------------------------------------------------------
# Bayesian update function 
# ----------------------------------------------------------------------
def bayes_update(prior: np.ndarray, likelihood: np.ndarray, health_score: float) -> np.ndarray:
    """Update prior with likelihood and health score."""
    return prior * likelihood * health_score

# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def random_hv(d=10000, kind="complex", seed=None):
    """Generate a random hypervector of dimension d.

    Parameters
    ----------
    d:
        Dimension of the hypervector.
    kind:
        "complex"  — unit-magnitude complex vector (each component e^{i*theta},
                     theta ~ Uniform[0, 2pi]).  These are the natural carriers
                     for phase-based fractional binding.
        "bipolar"  — real vector with each component in {+1, -1}.
        "real"     — Gaussian sample normalized to unit L2 norm.
    seed:
        Integer seed for reproducibility; None for random.

    Returns
    -------
    np.ndarray
        Shape (d,).  dtype=complex128 for kind="complex", float64 otherwise.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    elif kind == "bipolar":
        return rng.choice([-1, 1], size=d)
    else:
        vec = rng.normal(size=d)
        return vec / np.linalg.norm(vec)

def compute_health_score(hv1: np.ndarray, hv2: np.ndarray) -> float:
    """Compute the health score as the dot product of two hypervectors."""
    return np.dot(hv1, hv2)

def hybrid_operation(doc_id: str, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
    """Perform the hybrid operation by updating the prior with the likelihood 
    and health score, and then computing the semantic neighbors."""
    prior = _ENCLAVE[doc_id]
    likelihood = hv1
    health_score = compute_health_score(hv1, hv2)
    updated_prior = bayes_update(prior, likelihood, health_score)
    return updated_prior

if __name__ == "__main__":
    # Smoke test
    clear_enclave()
    register_document("doc1", [1.0, 2.0, 3.0])
    register_document("doc2", [4.0, 5.0, 6.0])
    hv1 = random_hv(d=3, kind="real")
    hv2 = random_hv(d=3, kind="real")
    updated_prior = hybrid_operation("doc1", hv1, hv2)
    print(updated_prior)