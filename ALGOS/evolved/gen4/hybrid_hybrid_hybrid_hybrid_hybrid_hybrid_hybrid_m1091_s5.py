# DARWIN HAMMER — match 1091, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_nlms_omni_cha_m1_s3.py (gen3)
# born: 2026-05-29T23:32:48Z

"""Hybrid Decision‑Hygiene & Bayesian‑NLMS Engine
================================================

This module fuses the two parent algorithms:

* **Parent A** – a hygiene‑scoring system that extracts categorical regex
  counts from a text and builds a feature‑count vector.
* **Parent B** – a Bayesian edge‑belief system that computes expected edge
  lengths in a geometric graph and provides NLMS (Normalized Least‑Mean‑Squares)
  learning utilities.

**Mathematical bridge**

Let  

* :math:`\\mathbf{f}\\in\\mathbb{R}^k` be the feature‑count vector produced by
  Parent A (one entry per regex category).  
* :math:`\\{e_i\\}` be the set of edges of a graph with endpoints
  :math:`(p_i,q_i)` and integer impedance :math:`z_i`.  
* The *prior* belief that an edge is true is :math:`\\pi`.  
* For each edge we define a likelihood proportional to the inverse impedance,
  :math:`\\ell_i = 1/(z_i+\\epsilon)`.

Using Parent B’s Bayesian update we obtain a posterior probability for each
edge:

.. math::
    P(e_i\\mid \\text{data}) = 
    \\frac{\\pi\\,\\ell_i}{\\pi\\,\\ell_i + (1-\\pi)\\,\\beta},

where :math:`\\beta` is a small false‑positive rate (default 0.01).  
The *expected edge length* vector is then

.. math::
    \\mathbf{L}=\\bigl[\\,P(e_i\\mid \\text{data})\\,\\|p_i-q_i\\|\\,\\bigr]_{i=1}^{m}.

The hybrid combines the two structures by weighting the hygiene feature
vector with the normalized expected lengths:

.. math::
    \\mathbf{h}=\\frac{\\mathbf{L}}{\\|\\mathbf{L}\\|_1}\\odot\\mathbf{f},

where :math:`\\odot` denotes element‑wise multiplication.  The resulting
vector ``h`` is fed to an NLMS predictor (Parent B) that yields a scalar
decision score.  A certainty dictionary is then assembled using Parent B’s
``certainty`` helper.

The code below implements this pipeline and provides three public functions
demonstrating the hybrid operation.
"""

import math
import sys
import random
from pathlib import Path
import re
from typing import Dict, List, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regex‑based feature extraction
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
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:scarce|shortage|limited|rare|run out|exhaust|deplete|budget cut|quota|cap|restriction)\b",
    re.I,
)

_REGEXES: List[Tuple[str, re.Pattern]] = [
    ("evidence", EVIDENCE_RE),
    ("planning", PLANNING_RE),
    ("delay", DELAY_RE),
    ("support", SUPPORT_RE),
    ("boundary", BOUNDARY_RE),
    ("outcome", OUTCOME_RE),
    ("impulsive", IMPULSIVE_RE),
    ("scarcity", SCARCITY_RE),
]

def extract_feature_vector(text: str) -> np.ndarray:
    """
    Count matches for each regex category and return a float vector.
    Order follows ``_REGEXES``.
    """
    counts = [len(pattern.findall(text)) for _, pattern in _REGEXES]
    return np.array(counts, dtype=float)


# ----------------------------------------------------------------------
# Parent B – geometry, Bayesian belief, NLMS learning
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]
NodeId = str
EdgeWithImpedance = Tuple[NodeId, NodeId, int]

EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = L·π + FP·(1‑π)"""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = π·L / P(E)"""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Tuple[str, ...] = (),
) -> dict[str, str]:
    """Structure mirroring Parent B's certainty payload."""
    return {
        "label": label,
        "confidence_bps": str(confidence_bps),
        "authority_class": authority_class,
        "rationale": rationale,
        "evidence_refs": evidence_refs,
    }

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot‑product w·x."""
    return float(weights @ x)

def nlms_batch_update(
    weights: np.ndarray,
    X: np.ndarray,
    targets: np.ndarray,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalized LMS batch update.

    Returns
    -------
    new_weights, errors
    """
    errors = targets - X @ weights
    norm_factors = np.linalg.norm(X, axis=1) ** 2 + eps
    delta = mu * (errors / norm_factors)[:, None] * X
    new_weights = weights + delta.sum(axis=0)
    return new_weights, errors


# ----------------------------------------------------------------------
# Hybrid core – mathematical bridge
# ----------------------------------------------------------------------
def edge_posterior_probabilities(
    edges: Sequence[EdgeWithImpedance],
    prior: float = 0.5,
    false_positive: float = 0.01,
) -> np.ndarray:
    """
    Compute posterior probabilities for each edge using a likelihood that
    is inversely proportional to its impedance.

    Returns an array ``p`` with shape (m, ) where ``m`` is the number of edges.
    """
    impedances = np.array([z for _, _, z in edges], dtype=float)
    # Avoid division by zero
    likelihoods = 1.0 / (impedances + 1e-6)
    # Normalise likelihoods to [0,1] (optional but keeps them in probability range)
    likelihoods = likelihoods / likelihoods.max()
    marginals = np.vectorize(bayes_marginal)(prior, likelihoods, false_positive)
    posteriors = np.vectorize(bayes_update)(prior, likelihoods, marginals)
    return posteriors


def expected_edge_lengths(
    points: Dict[NodeId, Point],
    edges: Sequence[EdgeWithImpedance],
    prior: float = 0.5,
    false_positive: float = 0.01,
) -> np.ndarray:
    """
    Expected length of each edge = posterior(edge) * Euclidean length.
    Returns a 1‑D ``np.ndarray`` of shape (m, ).
    """
    post = edge_posterior_probabilities(edges, prior, false_positive)
    lengths = np.array(
        [length(points[u], points[v]) for u, v, _ in edges], dtype=float
    )
    return post * lengths


def hybrid_feature_weighting(
    feature_vec: np.ndarray,
    edge_lengths: np.ndarray,
) -> np.ndarray:
    """
    Weight the hygiene feature vector by the normalised expected edge lengths.

    The edge length vector is first normalised to sum to 1 (L1 norm).  If the
    vector sums to zero we fall back to uniform weights.
    """
    if edge_lengths.size == 0:
        raise ValueError("edge_lengths must contain at least one element")
    total = edge_lengths.sum()
    if total == 0.0:
        weights = np.ones_like(edge_lengths) / edge_lengths.size
    else:
        weights = edge_lengths / total
    # Broadcast or truncate to match feature vector size
    if weights.size != feature_vec.size:
        # Simple strategy: repeat or truncate the weight vector
        repeats = int(np.ceil(feature_vec.size / weights.size))
        weights = np.tile(weights, repeats)[: feature_vec.size]
    return weights * feature_vec


# ----------------------------------------------------------------------
# Public API – three demonstrative functions
# ----------------------------------------------------------------------
def hybrid_predict(
    text: str,
    points: Dict[NodeId, Point],
    edges: Sequence[EdgeWithImpedance],
    prior: float = 0.5,
    false_positive: float = 0.01,
    nlms_weights: np.ndarray | None = None,
) -> dict:
    """
    End‑to‑end hybrid prediction.

    Steps
    -----
    1. Extract hygiene feature vector from ``text``.
    2. Compute expected edge lengths from the geometric graph.
    3. Weight the feature vector (bridge).
    4. Run NLMS prediction (uses supplied ``nlms_weights`` or defaults to ones).
    5. Assemble a certainty payload.

    Returns a dictionary with keys:
        - ``prediction`` : float
        - ``features``   : np.ndarray (raw)
        - ``weighted``   : np.ndarray (after bridge)
        - ``edge_lengths`` : np.ndarray
        - ``certainty``  : dict
    """
    # 1. Feature extraction
    f_vec = extract_feature_vector(text)

    # 2. Expected edge lengths
    exp_lengths = expected_edge_lengths(points, edges, prior, false_positive)

    # 3. Bridge weighting
    h_vec = hybrid_feature_weighting(f_vec, exp_lengths)

    # 4. NLMS prediction
    d = h_vec.shape[0]
    if nlms_weights is None:
        nlms_weights = np.ones(d, dtype=float)
    elif nlms_weights.shape != (d,):
        raise ValueError("nlms_weights shape mismatch")
    pred = nlms_predict(nlms_weights, h_vec)

    # 5. Certainty payload (confidence scaled to basis points)
    confidence_bps = int(max(0, min(10000, pred * 10000)))  # clamp to [0,10000]
    cert = certainty(
        label="HYBRID_SCORE",
        confidence_bps=confidence_bps,
        authority_class="HYBRID_ENGINE",
        rationale="Weighted hygiene features passed through NLMS predictor.",
        evidence_refs=(),
    )

    return {
        "prediction": pred,
        "features": f_vec,
        "weighted": h_vec,
        "edge_lengths": exp_lengths,
        "certainty": cert,
    }


def hybrid_train_batch(
    texts: List[str],
    points: Dict[NodeId, Point],
    edges: Sequence[EdgeWithImpedance],
    targets: np.ndarray,
    prior: float = 0.5,
    false_positive: float = 0.01,
    init_weights: np.ndarray | None = None,
    mu: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train NLMS weights on a batch of examples.

    Parameters
    ----------
    texts : List[str]
        Raw documents.
    points, edges : graph description (shared across the batch).
    targets : np.ndarray of shape (N,) – desired scalar outputs.
    init_weights : optional initial weight vector; if ``None`` a vector of ones
        of appropriate length is used.
    mu : learning rate for NLMS.

    Returns
    -------
    final_weights, errors
    """
    if len(texts) != targets.shape[0]:
        raise ValueError("Number of texts must match number of targets")

    # Build design matrix X where each row is the hybrid‑weighted feature vector
    X_rows = []
    for txt in texts:
        f = extract_feature_vector(txt)
        L = expected_edge_lengths(points, edges, prior, false_positive)
        h = hybrid_feature_weighting(f, L)
        X_rows.append(h)
    X = np.vstack(X_rows)

    d = X.shape[1]
    if init_weights is None:
        init_weights = np.ones(d, dtype=float)

    final_weights, errors = nlms_batch_update(init_weights, X, targets, mu=mu)
    return final_weights, errors


def demo_hybrid_system() -> None:
    """
    Simple demonstration that runs the hybrid on synthetic data.
    """
    # Synthetic graph
    points = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
    }
    edges = [
        ("A", "B", 3),
        ("B", "C", 5),
        ("C", "D", 2),
        ("D", "A", 4),
    ]

    # Sample texts
    texts = [
        "We have verified the source and the plan is ready. No delay.",
        "Immediate action required! Rage and panic, do it now!",
        "The budget is limited, but we have a checklist and audit logs.",
    ]

    # Dummy targets for training (e.g., 0 = low risk, 1 = high risk)
    targets = np.array([0.2, 0.9, 0.4])

    # Train
    weights, err = hybrid_train_batch(texts, points, edges, targets)
    print("Trained NLMS weights:", weights)
    print("Training errors:", err)

    # Predict on a new message
    new_msg = "Please verify the documentation before we proceed. No rush."
    result = hybrid_predict(new_msg, points, edges, nlms_weights=weights)
    print("\nHybrid prediction result:")
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: {v.tolist()}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    demo_hybrid_system()