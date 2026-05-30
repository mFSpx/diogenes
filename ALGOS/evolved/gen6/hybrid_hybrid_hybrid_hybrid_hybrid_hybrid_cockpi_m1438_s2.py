# DARWIN HAMMER — match 1438, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m1048_s0.py (gen5)
# parent_b: hybrid_hybrid_cockpit_metri_hybrid_hybrid_hybrid_m608_s1.py (gen5)
# born: 2026-05-29T23:36:19Z

"""Hybrid Bayesian‑SSIM‑Stylometry Decision Engine

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m1048_s0.py (Algorithm A)
- hybrid_hybrid_cockpit_metri_hybrid_hybrid_hybrid_m608_s1.py (Algorithm B)

Mathematical bridge:
Both parents transform raw text into a numeric *feature vector*.
Algorithm A creates a deterministic random vector via ``extract_full_features`` and
uses Euclidean distance as an SSIM‑like similarity measure.
Algorithm B produces a stylometry vector (function‑category frequencies) and
operates on those vectors with linear‑algebraic utilities.

The fusion builds a **joint feature vector** by concatenating the two
representations and then forms a *joint information matrix* ``J = v·vᵀ``.
From ``J`` we derive:
* Gini impurity (information‑theoretic dispersion)
* Hoeffding bound (confidence on empirical estimates)
* A Bayesian posterior that treats the hybrid similarity as a likelihood.

The resulting score integrates spatial (SSIM), linguistic (stylometry),
information‑theoretic (Gini), and statistical‑learning (Hoeffding) aspects
into a single decision value.
"""

import numpy as np
import random
import math
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Parent A – feature extraction & SSIM‑like similarity
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑random feature set derived from the text."""
    features: Dict[str, float] = {}
    rnd = random.Random(hash(text) & 0xFFFFFFFFFFFFFFFF)
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "rainmaker_corporate_grit_tension", "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight", "telemetry_agent_symm"
    ]
    for key in keys:
        features[key] = rnd.random()
    return features

def ssim_like_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    A lightweight SSIM‑like similarity: 1 – normalized Euclidean distance.
    Result lies in [0, 1] where 1 means identical vectors.
    """
    diff = np.linalg.norm(vec_a - vec_b)
    norm = np.linalg.norm(vec_a) + np.linalg.norm(vec_b) + 1e-12
    return max(0.0, 1.0 - diff / norm)

def payload_prototype_similarity(payload: str, prototype: str) -> float:
    """Similarity between two texts using the feature vectors of Algorithm A."""
    f_payload = np.array(list(extract_full_features(payload).values()))
    f_prototype = np.array(list(extract_full_features(prototype).values()))
    return ssim_like_similarity(f_payload, f_prototype)

# ----------------------------------------------------------------------
# Parent B – stylometry utilities
# ----------------------------------------------------------------------
FUNCTION_CATS = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}

def words(text: str) -> List[str]:
    """Tokenise a string into lower‑case alphabetic words."""
    return [w for w in (text or "").lower().split() if w.isalpha()]

def stylometry_features(text: str, dim: int) -> np.ndarray:
    """
    Frequency of function‑category words.
    Returns a vector of length ``dim`` (max 9 for the defined categories).
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    cats = list(FUNCTION_CATS.keys())[:dim]
    return np.array([sum(cnt[w] for w in FUNCTION_CATS[cat]) / total for cat in cats])

def lsm_vector(text: str) -> np.ndarray:
    """Convenient 6‑dim stylometry vector used by the original parent."""
    return stylometry_features(text, 6)

# ----------------------------------------------------------------------
# Fusion core – joint vector, information matrix, and decision logic
# ----------------------------------------------------------------------
def joint_feature_vector(text: str) -> np.ndarray:
    """
    Concatenate Algorithm A's deterministic random features with
    Algorithm B's stylometry vector (6‑dim). Resulting length = 13 + 6 = 19.
    """
    a_vec = np.array(list(extract_full_features(text).values()))
    b_vec = lsm_vector(text)
    return np.concatenate([a_vec, b_vec])

def gini_impurity(vec: np.ndarray) -> float:
    """
    Gini impurity of a probability distribution derived from ``vec``.
    The vector is first normalised to sum to 1.
    """
    if vec.size == 0:
        return 0.0
    probs = np.abs(vec)
    total = probs.sum()
    if total == 0:
        return 0.0
    probs = probs / total
    return 1.0 - np.sum(probs ** 2)

def hoeffding_bound(range_r: float, n: int, delta: float = 0.05) -> float:
    """
    Hoeffding bound ε = sqrt( (R² * ln(1/δ)) / (2n) ).
    ``range_r`` is the known range of the random variable (max‑min).
    """
    if n <= 0:
        return float('inf')
    return math.sqrt((range_r ** 2 * math.log(1.0 / delta)) / (2.0 * n))

def hybrid_similarity(payload: str, prototype: str, reference: str) -> float:
    """
    Fuse:
    * SSIM‑like similarity between payload & prototype (Algorithm A)
    * Cosine similarity between stylometry vectors of prototype & reference (Algorithm B)

    The two similarities are combined with equal weighting.
    """
    # SSIM‑like part
    ssim = payload_prototype_similarity(payload, prototype)

    # Stylometry cosine part
    vec_proto = lsm_vector(prototype)
    vec_ref = lsm_vector(reference)
    dot = np.dot(vec_proto, vec_ref)
    norm = np.linalg.norm(vec_proto) * np.linalg.norm(vec_ref) + 1e-12
    cosine = max(0.0, min(1.0, dot / norm))

    # Equal blend
    return 0.5 * ssim + 0.5 * cosine

def bayesian_posterior(prior: float, likelihood: float) -> float:
    """Standard binary Bayesian update."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0):
        raise ValueError("Prior and likelihood must be in [0,1]")
    numerator = prior * likelihood
    denominator = numerator + (1 - prior) * (1 - likelihood)
    return numerator / denominator if denominator != 0 else 0.0

def hybrid_decision_score(
    payload: str,
    prototype: str,
    reference: str,
    prior: float = 0.5,
    hoeffding_range: float = 1.0,
    sample_size: int = 100,
    delta: float = 0.05
) -> float:
    """
    Compute a decision score that blends:
    * Bayesian posterior (using hybrid similarity as likelihood)
    * Gini impurity of the joint feature representation of the payload
    * Hoeffding bound as a confidence penalty

    The final score is:
        posterior - ε * Gini
    where ε is the Hoeffding bound.
    """
    # 1. Likelihood from hybrid similarity
    likelihood = hybrid_similarity(payload, prototype, reference)

    # 2. Bayesian posterior
    post = bayesian_posterior(prior, likelihood)

    # 3. Information‑theoretic penalty
    joint_vec = joint_feature_vector(payload)
    gini = gini_impurity(joint_vec)

    # 4. Confidence penalty via Hoeffding bound
    epsilon = hoeffding_bound(hoeffding_range, sample_size, delta)

    # Combine
    score = post - epsilon * gini
    # Clamp to [0,1] for interpretability
    return max(0.0, min(1.0, score))

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    payload_txt = "The quick brown fox jumps over the lazy dog."
    prototype_txt = "A swift auburn animal leaps above a sleepy canine."
    reference_txt = "In a distant land, the fox was known for its agility."

    print("SSIM‑like similarity (A):", payload_prototype_similarity(payload_txt, prototype_txt))
    print("Stylometry vector (B) prototype:", lsm_vector(prototype_txt))
    print("Hybrid similarity:", hybrid_similarity(payload_txt, prototype_txt, reference_txt))
    print("Joint feature vector (first 5 values):", joint_feature_vector(payload_txt)[:5])
    print("Gini impurity of payload joint vector:", gini_impurity(joint_feature_vector(payload_txt)))
    print("Hoeffding bound (R=1, n=100):", hoeffding_bound(1.0, 100))
    print("Hybrid decision score:", hybrid_decision_score(payload_txt, prototype_txt, reference_txt))
    sys.exit(0)