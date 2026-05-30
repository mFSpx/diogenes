# DARWIN HAMMER — match 1795, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_rbf_surrogate_perceptual_dedupe_m57_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_regret_engine_m2_s4.py (gen3)
# born: 2026-05-29T23:39:02Z

"""Hybrid RBF‑Perceptual + Regret‑Decision Engine

Parents
-------
* **Parent A** – ``hybrid_hybrid_rbf_surrogate_perceptual_dedupe_m57_s2.py``  
  Builds an RBF surrogate using a kernel  
  ``K_ij = exp(-ε_e·‖x_i‑x_j‖² - ε_h·(d_H(h_i,h_j)/B)²)``  
  where ``h_i`` are perceptual hashes of the input vectors.

* **Parent B** – ``hybrid_hybrid_hybrid_decisi_hybrid_regret_engine_m2_s4.py``  
  Extracts regex‑based feature counts **c**, forms a utility vector  
  ``u = p·c – n·c`` (positive/negative weight vectors), builds a
  regret‑weighted soft‑max distribution  
  ``π_i = exp(u_i‑max(u))/Σ_j exp(u_j‑max(u))`` and evaluates the Gini
  coefficient of ``π``.

Mathematical Bridge
-------------------
The surrogate of Parent A predicts a scalar *s(x)* for any feature vector
*x*.  This scalar is injected as an **extra additive term** into the utility
construction of Parent B:


u_i = (p·c_i – n·c_i) + α·s(x_i)


Thus the Euclidean‑/Hamming‑based similarity encoded in the hybrid kernel
influences the regret‑weighted decision making.  The rest of the pipeline
(remains unchanged) – soft‑max, Gini, and final scoring – operates on the
augmented utilities.

The module below implements:
* perceptual hash utilities,
* the combined RBF kernel and a lightweight surrogate trainer/predictor,
* regex‑based feature extraction,
* utility construction that fuses surrogate predictions,
* regret‑weighted soft‑max and Gini,
* an end‑to‑end ``hybrid_decision`` function.

Only the Python standard library and NumPy are used.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
import datetime as dt
import re
import hashlib

import numpy as np

Vector = np.ndarray  # alias for readability

# ----------------------------------------------------------------------
# Parent A – perceptual hash utilities
# ----------------------------------------------------------------------
def compute_phash(data: bytes, bits: int = 64) -> int:
    """Return a deterministic *bits*-length perceptual hash of *data*.

    The implementation uses an MD5 digest, truncates/pads to the requested
    length and interprets the result as an unsigned integer.
    """
    digest = hashlib.md5(data).digest()
    # Convert to integer and mask to required bit‑length
    h_int = int.from_bytes(digest, byteorder="big")
    return h_int & ((1 << bits) - 1)


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return bin(a ^ b).count("1")


# ----------------------------------------------------------------------
# Parent A – combined RBF kernel
# ----------------------------------------------------------------------
def combined_kernel(
    X: np.ndarray,
    eps_e: float = 1.0,
    eps_h: float = 1.0,
    hash_bits: int = 64,
) -> np.ndarray:
    """Build the hybrid kernel matrix K for a set of vectors X.

    Parameters
    ----------
    X : (N, D) array
        Input feature vectors.
    eps_e, eps_h : float
        Scaling factors for Euclidean and Hamming contributions.
    hash_bits : int
        Length of the perceptual hash in bits.

    Returns
    -------
    K : (N, N) array
        Symmetric positive‑definite kernel matrix.
    """
    N = X.shape[0]
    # Euclidean part
    sq_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)  # (N,N)

    # Perceptual‑hash part
    hashes = np.array([compute_phash(X[i].tobytes(), bits=hash_bits) for i in range(N)], dtype=np.uint64)
    ham_matrix = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            d = hamming_distance(int(hashes[i]), int(hashes[j]))
            ham_matrix[i, j] = d
            ham_matrix[j, i] = d

    # Normalized Hamming distance (divide by max possible distance = hash_bits)
    ham_norm = ham_matrix / hash_bits

    K = np.exp(-eps_e * sq_dists - eps_h * ham_norm ** 2)
    return K


def fit_hybrid(X: np.ndarray, y: np.ndarray, eps_e: float = 1.0, eps_h: float = 1.0) -> np.ndarray:
    """Solve K·w = y for the weight vector w."""
    K = combined_kernel(X, eps_e, eps_h)
    # Add a small nugget for numerical stability
    K += np.eye(K.shape[0]) * 1e-12
    w = np.linalg.solve(K, y)
    return w


@dataclass
class HybridRBFSurrogate:
    """Trained hybrid RBF surrogate model."""

    X_train: np.ndarray
    w: np.ndarray
    eps_e: float = 1.0
    eps_h: float = 1.0
    hash_bits: int = 64

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict scalar outputs for new inputs."""
        K_test = self._kernel_between(X, self.X_train)
        return K_test @ self.w

    def _kernel_between(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Kernel matrix between two sets of vectors."""
        # Euclidean part
        sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)

        # Hash part
        hashes_A = np.array([compute_phash(a.tobytes(), bits=self.hash_bits) for a in A], dtype=np.uint64)
        hashes_B = np.array([compute_phash(b.tobytes(), bits=self.hash_bits) for b in B], dtype=np.uint64)
        N, M = len(A), len(B)
        ham = np.empty((N, M), dtype=np.float64)
        for i in range(N):
            for j in range(M):
                ham[i, j] = hamming_distance(int(hashes_A[i]), int(hashes_B[j]))
        ham_norm = ham / self.hash_bits

        return np.exp(-self.eps_e * sq - self.eps_h * ham_norm ** 2)


# ----------------------------------------------------------------------
# Parent B – regex feature extraction
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)


def extract_counts(text: str) -> np.ndarray:
    """Return a vector [evidence, planning, delay, support] of regex matches."""
    return np.array(
        [
            len(EVIDENCE_RE.findall(text)),
            len(PLANNING_RE.findall(text)),
            len(DELAY_RE.findall(text)),
            len(SUPPORT_RE.findall(text)),
        ],
        dtype=np.float64,
    )


# ----------------------------------------------------------------------
# Parent B – regret‑weighted soft‑max and Gini
# ----------------------------------------------------------------------
def regret_softmax(u: np.ndarray) -> np.ndarray:
    """Regret‑weighted soft‑max of utility vector *u*."""
    shift = np.max(u)
    exps = np.exp(u - shift)
    return exps / np.sum(exps)


def gini_coefficient(p: np.ndarray) -> float:
    """Gini coefficient of a probability distribution *p* (sum(p)=1)."""
    if p.ndim != 1:
        raise ValueError("p must be a one‑dimensional array")
    sorted_p = np.sort(p)
    n = len(p)
    cum = np.cumsum(sorted_p)
    gini = 1.0 - 2.0 * np.sum(cum) / (n * np.sum(sorted_p)) + (1.0 / n)
    return float(gini)


# ----------------------------------------------------------------------
# Hybrid – utility construction that fuses the surrogate prediction
# ----------------------------------------------------------------------
def compute_utilities(
    counts: np.ndarray,
    pos_weights: np.ndarray,
    neg_weights: np.ndarray,
    surrogate_pred: float,
    alpha: float = 0.5,
) -> np.ndarray:
    """Combine regex‑based utilities with surrogate output.

    Parameters
    ----------
    counts : (K,) array of feature counts.
    pos_weights, neg_weights : (K,) arrays.
    surrogate_pred : scalar prediction from the hybrid RBF surrogate.
    alpha : weighting factor for the surrogate term.

    Returns
    -------
    u : (K,) utility vector.
    """
    base = pos_weights * counts - neg_weights * counts
    # Broadcast surrogate term across all dimensions
    return base + alpha * surrogate_pred


# ----------------------------------------------------------------------
# End‑to‑end decision function
# ----------------------------------------------------------------------
def hybrid_decision(
    text: str,
    surrogate: HybridRBFSurrogate,
    pos_weights: np.ndarray,
    neg_weights: np.ndarray,
    alpha: float = 0.5,
) -> dict:
    """Perform the full hybrid pipeline on *text*.

    Returns a dictionary with intermediate results and the final decision
    score (higher = more favorable).
    """
    # 1. Regex feature counts
    counts = extract_counts(text)

    # 2. Vector representation for surrogate:
    #    Use the same counts vector as a proxy numeric feature.
    X_input = counts.reshape(1, -1).astype(np.float64)

    # 3. Surrogate prediction (scalar)
    s_pred = surrogate.predict(X_input)[0]

    # 4. Utility vector with surrogate injection
    u = compute_utilities(counts, pos_weights, neg_weights, s_pred, alpha)

    # 5. Regret‑weighted probability distribution
    pi = regret_softmax(u)

    # 6. Gini of the distribution
    gini_feat = gini_coefficient(pi)

    # 7. Weekday distribution Gini (static uniform distribution → Gini=0)
    weekday_dist = np.ones(7) / 7.0
    gini_week = gini_coefficient(weekday_dist)

    # 8. Final decision score: combine inverse Gini (more equality = better)
    #    and the surrogate prediction.
    decision_score = (1.0 - gini_feat) * 0.7 + (1.0 - gini_week) * 0.3
    decision_score = decision_score * (1.0 + 0.1 * s_pred)  # slight boost from surrogate

    return {
        "counts": counts,
        "surrogate_prediction": float(s_pred),
        "utilities": u,
        "probabilities": pi,
        "gini_features": gini_feat,
        "gini_weekday": gini_week,
        "decision_score": float(decision_score),
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic training set for the surrogate
    rng = np.random.default_rng(42)
    X_train = rng.random((10, 4))  # 4‑dimensional (matches count vector size)
    y_train = rng.random(10) * 2 - 1  # targets in [-1, 1]

    # Train surrogate
    w = fit_hybrid(X_train, y_train, eps_e=0.8, eps_h=1.2)
    surrogate = HybridRBFSurrogate(X_train, w, eps_e=0.8, eps_h=1.2)

    # Define positive/negative weights for the four regex categories
    pos_w = np.array([1.0, 0.6, 0.2, 0.4])
    neg_w = np.array([0.3, 0.5, 0.7, 0.2])

    sample_text = """
    The evidence was verified and the source is reliable. We have a plan
    with a clear timeline, but we might need to pause tomorrow for a
    review. If you need support, call the team.
    """

    result = hybrid_decision(
        sample_text,
        surrogate,
        pos_weights=pos_w,
        neg_weights=neg_w,
        alpha=0.4,
    )

    print("Hybrid decision result:")
    for k, v in result.items():
        print(f"{k}: {v}")

    sys.exit(0)