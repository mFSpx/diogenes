# DARWIN HAMMER — match 4275, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2083_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hoeffd_m2531_s2.py (gen6)
# born: 2026-05-29T23:54:35Z

"""Hybrid Stylometric‑Bayesian‑Regret Engine

Parents:
- hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s0.py (stylometry, Bayesian update, haversine)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hoeffd_m2531_s2.py (entropy, risk, Gini, Hoeffding)

Mathematical bridge:
The posterior probability `P(model|text,weekday)` obtained from the Bayesian update in
Parent A is used as a *prior* for the regret‑weighted audit score of Parent B.
Conversely, the audit score `S` (entropy + risk·log(N)·G·sigmoid(U)) is modulated by the
spatial‑privacy penalty `exp(-α·d)` where `d` is the haversine distance between the
request location and the model’s registered location.  The final hybrid score for a
model `m` is

    ℑ(m) = P(m|text,weekday) · exp(-α·d(m)) · S(audit)

Thus the two topologies are fused through multiplication of the Bayesian posterior,
the spatial decay, and the regret‑weighted audit term.  The algorithm selects the model
with maximal `ℑ` to load and evicts the model with minimal `ℑ`."""

import math
import random
import sys
import pathlib
from datetime import datetime as dt
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Stylometry utilities (Parent A)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won".split()),
}

def _text_to_vector(text: str) -> np.ndarray:
    """Count occurrences of each function‑category token in `text`."""
    tokens = text.lower().split()
    vec = []
    for cat in FUNCTION_CATS.values():
        vec.append(sum(tok in cat for tok in tokens))
    return np.array(vec, dtype=float)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great‑circle distance in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def bayesian_posterior(prior: float, likelihood: float) -> float:
    """Simple Bayesian update with binary hypothesis."""
    # posterior ∝ prior * likelihood
    numer = prior * likelihood
    denom = numer + (1 - prior) * (1 - likelihood)
    return numer / denom if denom != 0 else 0.0

# ----------------------------------------------------------------------
# Regret‑Weighted audit utilities (Parent B)
# ----------------------------------------------------------------------
def shannon_entropy(counts: List[int]) -> float:
    """Entropy H = - Σ p_i log p_i."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log(p) for p in probs)

def gini_coefficient(values: List[float]) -> float:
    """Gini = 1 - Σ (v_i / Σv)²  (simplified for non‑negative values)."""
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values]
    return 1.0 - sum(p ** 2 for p in probs)

def hoeffding_uncertainty(N: int, delta: float = 0.05) -> float:
    """Hoeffding bound U = sqrt( (1/(2N)) * ln(2/delta) )."""
    if N <= 0:
        return float('inf')
    return math.sqrt((1.0 / (2 * N)) * math.log(2.0 / delta))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def audit_score(
    hygiene_counts: List[int],
    risk: float,
    token_count: int,
    regret_vals: List[float],
    N: int,
) -> float:
    """
    Implements the Parent B formula:
        S = H + R * log(N) * G * sigmoid(U)
    where
        H – Shannon entropy of hygiene_counts
        R – risk score
        G – Gini of regret_vals
        U – Hoeffding uncertainty (based on N)
    """
    H = shannon_entropy(hygiene_counts)
    G = gini_coefficient(regret_vals)
    U = hoeffding_uncertainty(N)
    return H + risk * math.log(max(token_count, 1)) * G * sigmoid(U)

# ----------------------------------------------------------------------
# Hybrid core (fusion of A & B)
# ----------------------------------------------------------------------
def hybrid_model_score(
    text: str,
    weekday: int,
    model_vec: np.ndarray,
    model_prior: float,
    model_location: Tuple[float, float],
    request_location: Tuple[float, float],
    hygiene_counts: List[int],
    risk: float,
    regret_vals: List[float],
    token_count: int,
    alpha: float = 0.001,
) -> float:
    """
    Compute the fused score ℑ(m) for a single model.

    Steps
    -----
    1. Stylometric similarity → likelihood.
    2. Bayesian posterior using `model_prior` and the likelihood.
    3. Spatial decay exp(-α·d) where d is haversine distance.
    4. Audit score S from Parent B.
    5. Final ℑ = posterior * spatial_decay * S.
    """
    # 1. similarity as likelihood (scaled to [0,1])
    text_vec = _text_to_vector(text)
    similarity = cosine_similarity(text_vec, model_vec)
    likelihood = similarity  # already in [0,1]

    # 2. posterior (weekday influences prior slightly)
    weekday_factor = 0.9 + 0.02 * ((weekday - 3) % 7)  # modest modulation
    prior = min(1.0, max(0.0, model_prior * weekday_factor))
    posterior = bayesian_posterior(prior, likelihood)

    # 3. spatial decay
    d = haversine_distance(
        request_location[0],
        request_location[1],
        model_location[0],
        model_location[1],
    )
    spatial_decay = math.exp(-alpha * d)

    # 4. audit score
    S = audit_score(
        hygiene_counts=hygiene_counts,
        risk=risk,
        token_count=token_count,
        regret_vals=regret_vals,
        N=token_count,
    )

    # 5. fused score
    return posterior * spatial_decay * S

def select_models(
    text: str,
    weekday: int,
    request_location: Tuple[float, float],
    model_pool: Dict[str, Dict],
) -> Tuple[str, str]:
    """
    Evaluate every model in `model_pool` with `hybrid_model_score` and return
    (best_model_id, worst_model_id).
    `model_pool` structure:
        {
            model_id: {
                "vector": np.ndarray,
                "prior": float,
                "location": (lat, lon),
                "hygiene_counts": List[int],
                "risk": float,
                "regret_vals": List[float],
                "token_count": int,
            },
            ...
        }
    """
    scores = {}
    for mid, spec in model_pool.items():
        score = hybrid_model_score(
            text=text,
            weekday=weekday,
            model_vec=spec["vector"],
            model_prior=spec["prior"],
            model_location=spec["location"],
            request_location=request_location,
            hygiene_counts=spec["hygiene_counts"],
            risk=spec["risk"],
            regret_vals=spec["regret_vals"],
            token_count=spec["token_count"],
        )
        scores[mid] = score

    best = max(scores, key=scores.get)
    worst = min(scores, key=scores.get)
    return best, worst

def eviction_policy(model_pool: Dict[str, Dict], retain: int = 3) -> List[str]:
    """
    Return a list of model IDs to keep (size `retain`) based on descending hybrid scores.
    The rest are candidates for eviction.
    """
    dummy_text = "sample text for evaluation"
    weekday = dt.utcnow().weekday()
    request_loc = (0.0, 0.0)  # neutral location for policy computation

    # Compute scores once
    scores = {
        mid: hybrid_model_score(
            text=dummy_text,
            weekday=weekday,
            model_vec=spec["vector"],
            model_prior=spec["prior"],
            model_location=spec["location"],
            request_location=request_loc,
            hygiene_counts=spec["hygiene_counts"],
            risk=spec["risk"],
            regret_vals=spec["regret_vals"],
            token_count=spec["token_count"],
        )
        for mid, spec in model_pool.items()
    }

    # Keep top‑`retain` models
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return sorted_ids[:retain]

# ----------------------------------------------------------------------
# Demo / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Construct a tiny synthetic model pool
    rng = np.random.default_rng(42)
    pool = {}
    for i in range(5):
        mid = f"model_{i}"
        pool[mid] = {
            "vector": rng.integers(0, 5, size=len(FUNCTION_CATS)).astype(float),
            "prior": random.uniform(0.2, 0.8),
            "location": (random.uniform(-90, 90), random.uniform(-180, 180)),
            "hygiene_counts": [random.randint(0, 10) for _ in range(4)],
            "risk": random.uniform(0.0, 1.0),
            "regret_vals": [random.random() for _ in range(5)],
            "token_count": random.randint(5, 100),
        }

    sample_text = "I think we should plan the next steps and verify the evidence."
    today_weekday = dt.utcnow().weekday()
    user_loc = (37.7749, -122.4194)  # San Francisco

    best, worst = select_models(
        text=sample_text,
        weekday=today_weekday,
        request_location=user_loc,
        model_pool=pool,
    )
    print(f"Best model: {best}")
    print(f"Worst model: {worst}")

    keep = eviction_policy(pool, retain=3)
    print(f"Models kept after eviction policy: {keep}")