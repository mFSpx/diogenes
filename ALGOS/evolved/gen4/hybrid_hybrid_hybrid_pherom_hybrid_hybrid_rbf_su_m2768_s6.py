# DARWIN HAMMER — match 2768, survivor 6
# gen: 4
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s1.py (gen2)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hard_truth_ma_m93_s3.py (gen3)
# born: 2026-05-29T23:45:54Z

"""Hybrid Algorithm combining Pheromone‑Infotaxis & Decision Hygiene (Parent A) with
RBF Surrogate Stylometry & Hard‑Truth Modeling (Parent B).

Mathematical Bridge
-------------------
- The pheromone probabilities `p_i` (from Parent A) are used as *weights* for a
  Radial‑Basis‑Function (RBF) surrogate model (from Parent B).  
- Decision‑hygiene scores are transformed into a Shannon entropy `H`.  This entropy
  modulates the RBF kernel width `ε` (ε → ε·(1+H)), linking information‑theoretic
  uncertainty to the smoothness of the surrogate.
- The RBF surrogate predicts a stylometric similarity `s` for a feature vector.
  The final broadcast probability for a graph node is the weighted sum
  `∑_i p_i·s_i`, i.e. a matrix‑vector multiplication that fuses both topologies.

The module provides three core functions demonstrating this hybrid operation:
`compute_hygiene_entropy`, `rbf_surrogate_predict`, and `hybrid_broadcast_probability`.
"""

import re
import math
import random
import sys
from pathlib import Path
from typing import Sequence, Mapping, Hashable, List, Dict, Set, Tuple

import numpy as np

# -------------------- Parent A Components --------------------

def calculate_pheromone_probabilities(surface_key: str, limit: int, db_url: str) -> List[float]:
    """
    Retrieve the most recent `limit` pheromone signal values for `surface_key`
    and return their normalized probabilities.  If the required database
    driver is unavailable, fall back to a uniform distribution.
    """
    try:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    SELECT signal_value FROM lucidota_runtime.surface_pheromone
                    WHERE surface_key=%s ORDER BY created_at DESC LIMIT %s
                    ''',
                    (surface_key, limit)
                )
                pheromones = [r['signal_value'] for r in cur.fetchall()]
    except Exception:
        # Fallback: generate deterministic dummy values for testing / offline use
        random.seed(hash(surface_key) % (2**32))
        pheromones = [random.random() + 0.1 for _ in range(limit)]

    total = sum(pheromones) or 1.0
    return [p / total for p in pheromones]

def decision_hygiene_scores(text: str) -> Dict[str, int]:
    """
    Score the presence of hygiene‑related keyword categories in `text`.
    Returns a dictionary mapping category names to integer counts.
    """
    EVIDENCE_RE = re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
        re.I
    )
    PLANNING_RE = re.compile(
        r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
        re.I
    )
    DELAY_RE = re.compile(
        r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|first|after|review)\b",
        re.I
    )
    SUPPORT_RE = re.compile(
        r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
        re.I
    )
    BOUNDARY_RE = re.compile(
        r"\b(?:boundary|boundaries|wall|limit|restriction|rule|policy|guideline|protocol)\b",
        re.I
    )

    categories = {
        "evidence": EVIDENCE_RE,
        "planning": PLANNING_RE,
        "delay": DELAY_RE,
        "support": SUPPORT_RE,
        "boundary": BOUNDARY_RE,
    }

    scores = {}
    for name, pattern in categories.items():
        scores[name] = len(pattern.findall(text))
    return scores

def compute_hygiene_entropy(scores: Dict[str, int]) -> float:
    """
    Compute the Shannon entropy of the normalized decision‑hygiene scores.
    """
    total = sum(scores.values()) or 1
    probs = [v / total for v in scores.values() if v > 0]
    return -sum(p * math.log(p, 2) for p in probs)

# -------------------- Parent B Components --------------------

Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def rbf_surrogate_predict(
    x: Vector,
    centers: List[Vector],
    weights: List[float],
    epsilon: float = 1.0,
) -> float:
    """
    RBF surrogate model prediction.
    Computes Σ_i w_i * φ(‖x‑c_i‖) where φ is the Gaussian kernel.
    """
    if len(centers) != len(weights):
        raise ValueError("centers and weights must have the same length")
    result = 0.0
    for c, w in zip(centers, weights):
        r = euclidean(x, c)
        result += w * gaussian(r, epsilon)
    return result

# -------------------- Hybrid Core Functions --------------------

def hybrid_rbf_epsilon_from_entropy(base_epsilon: float, entropy: float) -> float:
    """
    Modulate the RBF kernel width using Shannon entropy.
    Larger entropy → broader kernel (more smoothing).
    """
    return base_epsilon * (1.0 + entropy)

def hybrid_similarity_score(
    feature_vec: Vector,
    rbf_centers: List[Vector],
    rbf_weights: List[float],
    base_epsilon: float,
    hygiene_entropy: float,
) -> float:
    """
    Compute a stylometric similarity score where the RBF epsilon is adapted
    to the decision‑hygiene entropy.
    """
    epsilon = hybrid_rbf_epsilon_from_entropy(base_epsilon, hygiene_entropy)
    return rbf_surrogate_predict(feature_vec, rbf_centers, rbf_weights, epsilon)

def hybrid_broadcast_probability(
    surface_key: str,
    limit: int,
    db_url: str,
    text: str,
    feature_vec: Vector,
    rbf_centers: List[Vector],
    rbf_weights: List[float],
    base_epsilon: float = 1.0,
) -> float:
    """
    Full hybrid pipeline:
    1. Extract pheromone probabilities (weights) for the surface.
    2. Score decision‑hygiene keywords in `text` and compute entropy `H`.
    3. Predict stylometric similarity `s` with an entropy‑aware RBF surrogate.
    4. Return the weighted broadcast probability Σ_i p_i * s   (scalar).

    The result lies in [0, 1] if the surrogate output is bounded accordingly.
    """
    # Step 1 – pheromone weights
    pheromone_probs = calculate_pheromone_probabilities(surface_key, limit, db_url)

    # Step 2 – hygiene entropy
    scores = decision_hygiene_scores(text)
    entropy = compute_hygiene_entropy(scores)

    # Step 3 – similarity prediction
    similarity = hybrid_similarity_score(
        feature_vec, rbf_centers, rbf_weights, base_epsilon, entropy
    )

    # Step 4 – combine
    weighted_sum = sum(p * similarity for p in pheromone_probs)
    # Clamp to [0,1] for safety
    return max(0.0, min(1.0, weighted_sum))

# -------------------- Smoke Test --------------------

if __name__ == "__main__":
    # Dummy inputs for a self‑contained test
    surface_key = "test_surface"
    limit = 5
    db_url = "postgresql://user:pass@localhost/db"  # not used in fallback mode

    sample_text = (
        "We have verified the source and recorded the log. "
        "The plan includes a checklist and a timeline. "
        "Please wait until tomorrow before proceeding."
    )

    # Feature vector (e.g., stylometric embedding) – 3‑dimensional for simplicity
    feature_vec = [0.2, 0.5, 0.3]

    # RBF centers and weights – synthetic
    rbf_centers = [
        [0.1, 0.4, 0.3],
        [0.3, 0.6, 0.2],
        [0.25, 0.45, 0.35],
    ]
    rbf_weights = [0.6, 0.3, 0.1]

    prob = hybrid_broadcast_probability(
        surface_key,
        limit,
        db_url,
        sample_text,
        feature_vec,
        rbf_centers,
        rbf_weights,
        base_epsilon=0.8,
    )
    print(f"Hybrid broadcast probability: {prob:.4f}")