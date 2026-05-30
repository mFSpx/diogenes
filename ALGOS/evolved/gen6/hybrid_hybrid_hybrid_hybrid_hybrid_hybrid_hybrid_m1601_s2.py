# DARWIN HAMMER — match 1601, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_korpus_text_h_m537_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_decisi_m176_s1.py (gen4)
# born: 2026-05-29T23:37:41Z

"""Hybrid Fusion of DARWIN HAMMER Algorithms A and B
===================================================

Parent A: *Hybrid Endpoint Decision Hygiene* – provides morphology‑based recovery
priority `p`, regret term `R_i`, sigmoid weighting `g(·)`, MinHash Jaccard similarity
`sim(·,·)` and a bounded control signal `dance`.

Parent B: *Hybrid Bayesian‑SSIM‑Curvature Router* – contributes Bayesian posterior
`P(i|data)`, SSIM‑like similarity based on rich textual features, and a spatial
privacy metric derived from the haversine distance between resource locations.

**Mathematical Bridge**

Both parents expose a *score* that multiplies a *priority* with a *similarity* and a
*confidence* term.  The bridge is therefore the product of:


S_i   = p_i                     # morphology priority (A)
       * g(R_i)                # regret sigmoid (A)
       * (1 + sim_minhash)     # MinHash similarity (A)
       * dance_i               # bounded control (A)
       * P_i                   # Bayesian posterior (B)
       * (1 + ssim_i)          # SSIM‑like feature similarity (B)
       * (1 / (1 + d_i))       # spatial privacy factor, d_i = haversine distance (B)


The resulting hybrid score `S_i` is fed to a softmax to obtain a policy
distribution and to a LinUCB‑style confidence bound for exploration.

The module implements this fused computation together with three public
functions that illustrate the hybrid workflow.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology & Regret utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


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
    """Morphology‑based recovery priority (p)."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def sigmoid(x: float) -> float:
    """Standard logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))


def minhash_jaccard(sig_a: Tuple[int, ...], sig_b: Tuple[int, ...]) -> float:
    """Very lightweight MinHash Jaccard estimator using integer signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must be of equal length")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def _minhash_signature(text: str, n_hashes: int = 64) -> Tuple[int, ...]:
    """Create a deterministic MinHash signature from token hashes."""
    tokens = set(text.lower().split())
    rng = random.Random(hash(text) & 0xFFFFFFFFFFFFFFFF)
    # Simple deterministic pseudo‑random hash per token
    token_hashes = [abs(hash(tok)) for tok in tokens]
    signature = []
    for i in range(n_hashes):
        # Random coefficients for a linear hash family
        a = rng.randint(1, 2 ** 31 - 1)
        b = rng.randint(0, 2 ** 31 - 1)
        min_val = min(((a * h + b) % (2 ** 31 - 1)) for h in token_hashes)
        signature.append(min_val)
    return tuple(signature)


def dance_factor(seed: int = None) -> float:
    """Bounded control signal in [0.5, 1.5]."""
    rng = random.Random(seed)
    return 0.5 + rng.random()


# ----------------------------------------------------------------------
# Parent B – Feature extraction, SSIM‑like similarity, Bayesian update,
# and spatial privacy
# ----------------------------------------------------------------------


def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic random feature vector for a given text."""
    features: Dict[str, float] = {}
    rng = random.Random(hash(text) & 0xFFFFFFFFFFFFFFFF)
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
        features[key] = rng.random()
    return features


def ssim_like_similarity(a: str, b: str) -> float:
    """SSIM‑like similarity based on Euclidean distance of feature vectors."""
    fa = np.array(list(extract_full_features(a).values()))
    fb = np.array(list(extract_full_features(b).values()))
    dist = np.linalg.norm(fa - fb)
    # Convert distance to similarity in [0,1]
    return 1.0 / (1.0 + dist)


def bayesian_posterior(prior: float, likelihood: float) -> float:
    """Simple Bayesian update (unnormalized) and normalization."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0):
        raise ValueError("prior and likelihood must be probabilities")
    unnorm = prior * likelihood
    # For a binary hypothesis we can normalize against the complement
    complement = (1 - prior) * (1 - likelihood)
    return unnorm / (unnorm + complement) if (unnorm + complement) > 0 else 0.0


def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """Great‑circle distance in kilometers."""
    R = 6371.0  # Earth radius
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def spatial_privacy_factor(dist_km: float, scale: float = 1000.0) -> float:
    """Decay factor that penalises large distances (privacy cost)."""
    return 1.0 / (1.0 + dist_km / scale)


# ----------------------------------------------------------------------
# Hybrid core: combine both parents
# ----------------------------------------------------------------------


def compute_hybrid_score(
    morph: Morphology,
    expected: float,
    cost: float,
    risk: float,
    counterfactual: float,
    text_a: str,
    text_b: str,
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
    prior: float = 0.5,
    seed: int = None,
) -> float:
    """
    Compute the fused hybrid score S_i.

    Parameters
    ----------
    morph : Morphology
        Physical description used for recovery priority `p`.
    expected, cost, risk, counterfactual : float
        Components of the regret term `R_i`.
    text_a, text_b : str
        Textual payloads for MinHash and SSIM‑like similarity.
    lat_a, lon_a, lat_b, lon_b : float
        Geographic coordinates for the spatial privacy term.
    prior : float, optional
        Prior probability for Bayesian update (default 0.5).
    seed : int, optional
        Seed for the stochastic `dance` factor.

    Returns
    -------
    float
        The hybrid score `S_i`.
    """
    # ---- Parent A components ----
    p = righting_time_index(morph)
    R = expected - cost - risk + counterfactual
    g = sigmoid(R)

    sig_a = _minhash_signature(text_a)
    sig_b = _minhash_signature(text_b)
    sim_minhash = minhash_jaccard(sig_a, sig_b)

    dance = dance_factor(seed)

    # ---- Parent B components ----
    ssim = ssim_like_similarity(text_a, text_b)

    # Bayesian posterior using SSIM as likelihood proxy
    posterior = bayesian_posterior(prior, ssim)

    dist = haversine_distance(lat_a, lon_a, lat_b, lon_b)
    spatial_factor = spatial_privacy_factor(dist)

    # ---- Fusion ----
    score = (
        p *
        g *
        (1.0 + sim_minhash) *
        dance *
        posterior *
        (1.0 + ssim) *
        spatial_factor
    )
    return score


def softmax_policy(scores: Iterable[float]) -> List[float]:
    """Convert raw scores to a probability distribution via softmax."""
    arr = np.array(list(scores), dtype=float)
    max_val = np.max(arr)
    exp_vals = np.exp(arr - max_val)  # numerical stability
    probs = exp_vals / np.sum(exp_vals)
    return probs.tolist()


def linucb_confidence(
    score: float,
    feature_vector: np.ndarray,
    alpha: float = 1.0,
    A_inv: np.ndarray = None,
    b: np.ndarray = None,
) -> float:
    """
    Compute a LinUCB‑style upper confidence bound.

    Parameters
    ----------
    score : float
        Expected reward (the hybrid score).
    feature_vector : np.ndarray
        Contextual feature vector.
    alpha : float
        Exploration parameter.
    A_inv : np.ndarray, optional
        Inverse of the design matrix (defaults to identity).
    b : np.ndarray, optional
        Accumulated reward vector (defaults to zeros).

    Returns
    -------
    float
        Upper confidence bound.
    """
    d = feature_vector.shape[0]
    if A_inv is None:
        A_inv = np.eye(d)
    if b is None:
        b = np.zeros(d)

    theta = A_inv @ b
    p = score + alpha * math.sqrt(feature_vector @ A_inv @ feature_vector)
    return p


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Sample morphology
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # Regret components
    expected = 10.0
    cost = 2.5
    risk = 1.0
    counterfactual = 0.5

    # Textual payloads
    txt1 = "The quick brown fox jumps over the lazy dog."
    txt2 = "A swift auburn fox leaped above a sleepy canine."

    # Geographic locations (lat, lon)
    lat1, lon1 = 37.7749, -122.4194   # San Francisco
    lat2, lon2 = 34.0522, -118.2437   # Los Angeles

    # Compute hybrid score
    score = compute_hybrid_score(
        morph=morph,
        expected=expected,
        cost=cost,
        risk=risk,
        counterfactual=counterfactual,
        text_a=txt1,
        text_b=txt2,
        lat_a=lat1,
        lon_a=lon1,
        lat_b=lat2,
        lon_b=lon2,
        prior=0.6,
        seed=42,
    )
    print(f"Hybrid score: {score:.6f}")

    # Policy over a small set of alternative scores
    alt_scores = [score,
                  score * 0.9,
                  score * 1.1,
                  score * 0.8]
    policy = softmax_policy(alt_scores)
    print("Softmax policy:", policy)

    # LinUCB confidence bound demonstration
    feat_vec = np.array([1.0, 0.5, -0.2])
    ub = linucb_confidence(score, feat_vec, alpha=0.7)
    print(f"LinUCB upper confidence bound: {ub:.6f}")

    sys.exit(0)