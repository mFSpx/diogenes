# DARWIN HAMMER — match 1352, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_semantic_neig_hybrid_hybrid_hard_t_m939_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_path_s_m17_s4.py (gen4)
# born: 2026-05-29T23:35:28Z

"""Hybrid Semantic‑Bayesian‑Cue Algorithm
Parents:
- hybrid_hybrid_semantic_neig_hybrid_hybrid_hard_t_m939_s0.py (semantic neighbors + Bayesian update)
- hybrid_hybrid_hybrid_decisi_hybrid_hybrid_path_s_m17_s4.py (cue‑based load/privacy + deterministic pseudo‑random feature extraction)

Mathematical bridge:
The Bayesian‑updated feature vector `u = bayes_update(prior, likelihood)` is compared to
the stored document vectors via cosine similarity.  The raw similarity scores are
modulated by the cue‑derived load‑privacy pair `(ℓ, π)` extracted from each neighbour's
text and by a scalar derived from the deterministic pseudo‑random feature set
`f = extract_full_features(text)`.  The final hybrid score for a neighbour *j* is

    σ_j = cos(u, v_j) · (1 + ℓ_j) · exp(−π_j) · (1 + ‖f_j‖₂)

where `v_j` is the neighbour's vector, `ℓ_j` the load, `π_j` the privacy, and `‖f_j‖₂`
the Euclidean norm of its feature vector.  This fuses the linear‑algebraic core of
Parent A with the cue‑based weighting and deterministic feature generation of Parent B.
"""

import sys
import math
import random
import pathlib
import numpy as np
import re
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# In‑memory semantic enclave (Parent A)
# ----------------------------------------------------------------------
_ENCLAVE: Dict[str, Tuple[np.ndarray, str]] = {}  # doc_id → (vector, raw_text)


def clear_enclave() -> None:
    """Remove all registered documents."""
    _ENCLAVE.clear()


def register_document(doc_id: str, vector: List[float], text: str = "") -> None:
    """Store a document vector together with its raw text for later cue analysis."""
    _ENCLAVE[doc_id] = (np.array(vector, dtype=float), text)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0.0:
        return 0.0
    return float(np.dot(a, b) / den)


def semantic_neighbors(query_id: str, k: int = 5) -> List[Tuple[str, float]]:
    """Return the *k* most similar documents to *query_id* (excluding the query itself)."""
    if query_id not in _ENCLAVE:
        raise KeyError(f"Document {query_id!r} not registered.")
    q_vec, _ = _ENCLAVE[query_id]
    sims = [
        (doc_id, _cosine(q_vec, vec))
        for doc_id, (vec, _) in _ENCLAVE.items()
        if doc_id != query_id
    ]
    sims.sort(key=lambda x: (-x[1], x[0]))
    return sims[:k]


def bayes_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """Bayesian update: posterior ∝ prior × likelihood."""
    prod = prior * likelihood
    s = np.sum(prod)
    if s == 0.0:
        # Avoid division by zero – return a uniform distribution
        return np.full_like(prior, 1.0 / prior.size)
    return prod / s


# ----------------------------------------------------------------------
# Cue extraction & deterministic pseudo‑random features (Parent B)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|delay|postpone|defer)\b", re.I)

W_POS = np.array([1.2, 0.8, 0.5])
W_NEG = np.array([0.3, 0.2, 1.0])


def _count_cues(text: str) -> np.ndarray:
    """Count evidence, planning and delay cues."""
    return np.array(
        [
            len(EVIDENCE_RE.findall(text)),
            len(PLANNING_RE.findall(text)),
            len(DELAY_RE.findall(text)),
        ],
        dtype=float,
    )


def compute_load_privacy(text: str) -> Tuple[float, float]:
    """
    Compute a *load* (positive‑weighted cue sum) and a *privacy* penalty
    (proportional to delay cues) from raw text.
    """
    c = _count_cues(text)
    load = float(c @ (W_POS - W_NEG))
    privacy = float(c[2] * 0.7)  # delay cues increase privacy penalty
    return load, privacy


_FEATURE_KEYS = [
    "operator_visceral_ratio",
    "operator_tech_ratio",
    "operator_legal_osint_ratio",
    "operator_ledger_density",
    "operator_recursion_score",
    "operator_directive_ratio",
    "operator_target_density",
    "psyche_forensic_shield_ratio",
    "psyche_poetic_entropy",
    "psyche_dissociative_index",
    "psyche_wrath_velocity",
    "resilience_bureaucratic_weaponization_index",
    "resilience_resource_exhaustion_metric",
    "resilience_swarm_orchestration_density",
    "resilience_logic_crucifixion_index",
    "resilience_conspiracy_grounding_ratio",
    "resilience_chaotic_good_tax",
]


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Deterministic pseudo‑random feature extraction based on the hash of *text*.
    Returns a mapping from feature name to a float in [0, 1).
    """
    rnd = random.Random(hash(text))
    return {k: rnd.random() for k in _FEATURE_KEYS}


def _feature_norm(features: Dict[str, float]) -> float:
    """Euclidean norm of the feature vector."""
    arr = np.fromiter(features.values(), dtype=float)
    return float(np.linalg.norm(arr))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_update_vector(prior: List[float], likelihood: List[float]) -> np.ndarray:
    """Convenient wrapper that accepts Python lists, returns the Bayesian‑updated vector."""
    prior_arr = np.array(prior, dtype=float)
    likelihood_arr = np.array(likelihood, dtype=float)
    return bayes_update(prior_arr, likelihood_arr)


def hybrid_neighbor_scores(
    query_id: str,
    prior: List[float],
    likelihood: List[float],
    k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Compute hybrid scores for the *k* most semantically similar neighbours of *query_id*.

    Steps:
    1. Bayesian update of the query's prior/likelihood vectors → `u`.
    2. Retrieve the *k* nearest neighbours by raw cosine similarity.
    3. For each neighbour *j*:
       - Load & privacy from its stored text.
       - Deterministic feature norm `‖f_j‖₂`.
       - Final score σ_j = cos(u, v_j)·(1+ℓ_j)·exp(−π_j)·(1+‖f_j‖₂).
    4. Return a list sorted by descending σ_j.
    """
    if query_id not in _ENCLAVE:
        raise KeyError(f"Document {query_id!r} not registered.")
    u = hybrid_update_vector(prior, likelihood)

    # Raw neighbours (cosine similarity to the original query vector)
    raw_neighbours = semantic_neighbors(query_id, k)

    results: List[Tuple[str, float]] = []
    for neigh_id, raw_cos in raw_neighbours:
        v_j, text_j = _ENCLAVE[neigh_id]

        # Cosine between updated vector u and neighbour vector v_j
        cos_uv = _cosine(u, v_j)

        # Cue‑based modifiers
        load_j, privacy_j = compute_load_privacy(text_j)

        # Deterministic feature norm
        feat_j = extract_full_features(text_j)
        norm_f = _feature_norm(feat_j)

        # Hybrid score
        sigma = cos_uv * (1.0 + load_j) * math.exp(-privacy_j) * (1.0 + norm_f)
        results.append((neigh_id, sigma))

    results.sort(key=lambda x: -x[1])
    return results


def hybrid_query(
    query_id: str,
    prior: List[float],
    likelihood: List[float],
    top_n: int = 3,
) -> Tuple[str, float]:
    """
    Return the single best neighbour according to the hybrid scoring scheme.
    """
    scores = hybrid_neighbor_scores(query_id, prior, likelihood, k=top_n)
    if not scores:
        raise RuntimeError("No neighbours found.")
    return scores[0]  # (doc_id, score)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic data
    docs = {
        "alpha": ([1, 0, 0], "Evidence of plan and schedule."),
        "beta": ([0, 1, 0], "Delay and postpone the test."),
        "gamma": ([0, 0, 1], "Verified source and audit log."),
        "delta": ([0.5, 0.5, 0], "Checklist and roadmap prepared."),
        "epsilon": ([0, 0.5, 0.5], "Pause and wait for later verification."),
    }

    # Register documents
    for doc_id, (vec, txt) in docs.items():
        register_document(doc_id, vec, txt)

    # Prior and likelihood vectors (simple categorical distributions)
    prior_vec = [0.33, 0.33, 0.34]
    likelihood_vec = [0.6, 0.2, 0.2]

    # Perform a hybrid query from "alpha"
    best_id, best_score = hybrid_query("alpha", prior_vec, likelihood_vec, top_n=4)

    print(f"Best neighbour for 'alpha': {best_id} with hybrid score {best_score:.4f}")

    # Show full neighbour ranking
    ranking = hybrid_neighbor_scores("alpha", prior_vec, likelihood_vec, k=4)
    print("\nFull ranking (doc_id, hybrid_score):")
    for doc, sc in ranking:
        print(f"  {doc}: {sc:.4f}")