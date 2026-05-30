# DARWIN HAMMER — match 1971, survivor 4
# gen: 4
# parent_a: hybrid_model_pool_hybrid_hybrid_worksh_m319_s1.py (gen3)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_infotaxis_min_m178_s1.py (gen2)
# born: 2026-05-29T23:40:09Z

"""
Hybrid Algorithm: Curvature‑Weighted MinHash Bayesian Selector

Parents:
- hybrid_model_pool_hybrid_hybrid_worksh_m319_s1.py (curvature matrix from a 24‑dimensional
  feature vector, deterministic doomsday scaling, model‑load weighting)
- hybrid_hybrid_minimum_cost__hybrid_infotaxis_min_m178_s1.py (minimum‑cost tree edge
  weights interpreted as a probability distribution, Shannon entropy, MinHash signatures,
  Bayesian update, expected‑entropy minimisation)

Mathematical Bridge:
The curvature matrix `C = v·vᵀ` (parent A) yields per‑model curvature weights
`w_i = v_i²`.  These weights are treated as a discrete probability distribution over
models and are used as *edge priors* in a complete graph whose edge weight between
models `i` and `j` is the product of the average curvature weight and the
MinHash‑based similarity of their token signatures (parent B).  Normalising all
edge weights produces a probability distribution `p_ij`.  Its Shannon entropy
measures the uncertainty of the current model pool.  A Bayesian update, with the
curvature weight as prior and the similarity‑derived likelihood, yields a posterior
distribution over models.  The expected posterior entropy is computed for each
candidate model, and the model that minimises this expectation is selected – a
fusion of curvature‑driven allocation and minimum‑cost/entropy‑driven decision
logic.
"""

import math
import random
import sys
import hashlib
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set

import numpy as np

# ---------------------------------------------------------------------------
# Shared deterministic utilities
# ---------------------------------------------------------------------------

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """
    Return the weekday index used by the original doomsday calendar:
    Monday → 1, …, Sunday → 0 (mod 7).
    """
    return (date(year, month, day).weekday() + 1) % 7

def _rng_from_text(text: str) -> random.Random:
    """Deterministic RNG seeded from a stable SHA‑256 hash of *text*."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)

# ---------------------------------------------------------------------------
# Parent‑A inspired curvature computation
# ---------------------------------------------------------------------------

FEATURE_DIM = 24  # dimensionality of the deterministic feature vector

def compute_feature_vector(text: str) -> np.ndarray:
    """
    Produce a deterministic 24‑dimensional feature vector from *text*.
    The vector is L2‑normalised.
    """
    rng = _rng_from_text(text)
    vec = np.array([rng.random() for _ in range(FEATURE_DIM)], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def compute_curvature_weights(v: np.ndarray, model_names: Tuple[str, ...]) -> np.ndarray:
    """
    From the normalised feature vector *v*, compute per‑model curvature weights.
    For a model indexed *i* the weight is w_i = v_i² (the diagonal of C = v·vᵀ).
    The resulting vector is normalised to sum to 1.
    """
    if len(v) < len(model_names):
        raise ValueError("Feature vector length must be >= number of models")
    diag = v[: len(model_names)] ** 2
    total = diag.sum()
    if total == 0:
        return np.full_like(diag, 1.0 / len(diag))
    return diag / total

# ---------------------------------------------------------------------------
# Parent‑B inspired MinHash and Bayesian machinery
# ---------------------------------------------------------------------------

def _hash(seed: int, token: str) -> int:
    """BLAKE2b hashing function returning a 64‑bit integer."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def minhash_signature(tokens: Iterable[str], k: int = 64, seed: int = 0) -> np.ndarray:
    """
    Compute a MinHash signature of length *k* for *tokens*.
    Deterministic because the hash seed is fixed.
    """
    token_set: Set[str] = {t for t in tokens if t}
    if not token_set:
        return np.full(k, 2**63 - 1, dtype=np.int64)
    sig = np.full(k, 2**63 - 1, dtype=np.int64)
    for i in range(k):
        min_hash = min(_hash(seed + i, t) for t in token_set)
        sig[i] = min_hash
    return sig

def signature_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Proportion of equal entries between two MinHash signatures."""
    if sig1.shape != sig2.shape:
        raise ValueError("Signature shapes must match")
    equal = np.sum(sig1 == sig2)
    return equal / sig1.size

def shannon_entropy(probs: np.ndarray) -> float:
    """Standard Shannon entropy (base e) for a probability vector."""
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute marginal probability for Bayesian update."""
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform Bayesian update on the prior probability."""
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

# ---------------------------------------------------------------------------
# Hybrid core: combine curvature, MinHash similarity and Bayesian entropy
# ---------------------------------------------------------------------------

def build_edge_distribution(
    curvature_weights: np.ndarray,
    signatures: Dict[str, np.ndarray],
    model_names: Tuple[str, ...],
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Construct a normalised probability distribution over all unordered model pairs.
    Edge weight for pair (i, j) = avg_curvature * similarity(i, j).
    Returns the probability vector and the corresponding list of model‑pair keys.
    """
    pair_weights = []
    pair_keys = []
    n = len(model_names)
    for a in range(n):
        for b in range(a + 1, n):
            name_a, name_b = model_names[a], model_names[b]
            avg_curv = (curvature_weights[a] + curvature_weights[b]) / 2.0
            sim = signature_similarity(signatures[name_a], signatures[name_b])
            w = avg_curv * sim
            pair_weights.append(w)
            pair_keys.append((name_a, name_b))
    pair_weights = np.array(pair_weights, dtype=np.float64)
    total = pair_weights.sum()
    if total == 0:
        probs = np.full_like(pair_weights, 1.0 / len(pair_weights))
    else:
        probs = pair_weights / total
    return probs, pair_keys

def expected_posterior_entropy(
    candidate_idx: int,
    curvature_weights: np.ndarray,
    signatures: Dict[str, np.ndarray],
    model_names: Tuple[str, ...],
    false_positive: float = 0.01,
) -> float:
    """
    Compute the expected Shannon entropy of the posterior distribution if
    *candidate_idx* were selected.  The likelihood for each model is the average
    MinHash similarity with the candidate.
    """
    n = len(model_names)
    candidate_name = model_names[candidate_idx]
    cand_sig = signatures[candidate_name]

    # likelihood_i = similarity(model_i, candidate)
    likelihoods = np.array(
        [
            signature_similarity(signatures[name], cand_sig)
            for name in model_names
        ],
        dtype=np.float64,
    )

    # Bayesian update for each model
    priors = curvature_weights
    marginals = np.array(
        [
            bayes_marginal(priors[i], likelihoods[i], false_positive)
            for i in range(n)
        ],
        dtype=np.float64,
    )
    posteriors = np.array(
        [
            bayes_update(priors[i], likelihoods[i], marginals[i])
            for i in range(n)
        ],
        dtype=np.float64,
    )
    # Normalise posterior (numerical safety)
    post_sum = posteriors.sum()
    if post_sum == 0:
        posteriors = np.full_like(posteriors, 1.0 / n)
    else:
        posteriors = posteriors / post_sum

    return shannon_entropy(posteriors)

def hybrid_select_model(
    text: str,
    model_tokens: Dict[str, List[str]],
    doomsday_date: Tuple[int, int, int] = (2026, 5, 29),
) -> str:
    """
    Full hybrid decision routine.
    1. Extract a deterministic 24‑D feature vector from *text*.
    2. Compute curvature weights, scaled by the doomsday weekday (deterministic factor).
    3. Build MinHash signatures for each model.
    4. Form the edge probability distribution and compute its entropy (diagnostic).
    5. For each model, evaluate the expected posterior entropy and pick the minimiser.
    Returns the selected model name.
    """
    # ---- Step 1: feature vector ------------------------------------------------
    v = compute_feature_vector(text)

    # ---- Step 2: curvature weights with doomsday scaling -----------------------
    model_names = tuple(model_tokens.keys())
    curv_weights = compute_curvature_weights(v, model_names)

    # deterministic scaling factor from doomsday weekday (0‑6)
    dw = doomsday(*doomsday_date)
    scale = (dw + 1) / 7.0  # map to (0,1]
    curv_weights = curv_weights * scale
    curv_weights = curv_weights / curv_weights.sum()  # re‑normalise

    # ---- Step 3: MinHash signatures -------------------------------------------
    signatures = {
        name: minhash_signature(tokens, k=64, seed=dw) for name, tokens in model_tokens.items()
    }

    # ---- Step 4: edge distribution (diagnostic only) ---------------------------
    edge_probs, edge_keys = build_edge_distribution(curv_weights, signatures, model_names)
    edge_entropy = shannon_entropy(edge_probs)

    # ---- Step 5: expected posterior entropy per candidate ----------------------
    exp_entropies = [
        expected_posterior_entropy(idx, curv_weights, signatures, model_names)
        for idx in range(len(model_names))
    ]

    # Choose model with minimal expected posterior entropy; break ties by curvature weight.
    min_entropy = min(exp_entropies)
    candidates = [i for i, e in enumerate(exp_entropies) if e == min_entropy]
    if len(candidates) > 1:
        # pick the one with highest curvature weight
        best_idx = max(candidates, key=lambda i: curv_weights[i])
    else:
        best_idx = candidates[0]

    # Optional: expose diagnostic info via a summary dict
    hybrid_summary = {
        "feature_vector_norm": float(np.linalg.norm(v)),
        "doomsday_weekday": dw,
        "curvature_weights": curv_weights.tolist(),
        "edge_entropy": _pct(edge_entropy),
        "expected_posterior_entropies": [_pct(e) for e in exp_entropies],
        "selected_model": model_names[best_idx],
    }
    # Store for external inspection (could be returned or logged)
    hybrid_summary["edge_distribution"] = {
        f"{a}<->{b}": _pct(p) for p, (a, b) in zip(edge_probs, edge_keys)
    }

    # Attach summary to function attribute for quick access in tests
    hybrid_select_model.last_summary = hybrid_summary

    return model_names[best_idx]

def load_model_with_curvature(model_name: str, text: str) -> None:
    """
    Mock function demonstrating how a model would be “loaded” using curvature‑derived
    allocation.  It prints the allocated weight for the model.
    """
    v = compute_feature_vector(text)
    dummy_models = ("modelA", "modelB", "modelC", "modelD")
    curv_weights = compute_curvature_weights(v, dummy_models)
    idx = dummy_models.index(model_name)
    weight = _pct(curv_weights[idx] * 100)  # percentage
    print(f"Loading '{model_name}' with curvature allocation: {weight}% of RAM budget")

def hybrid_summary() -> Dict:
    """
    Return the diagnostic summary from the most recent call to
    :func:`hybrid_select_model`.  If the function has not been called yet,
    an empty dictionary is returned.
    """
    return getattr(hybrid_select_model, "last_summary", {})

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Define a tiny pool of mock models with token sets
    model_pool = {
        "modelA": ["alpha", "beta", "gamma", "delta"],
        "modelB": ["epsilon", "zeta", "eta", "theta"],
        "modelC": ["iota", "kappa", "lambda", "mu"],
        "modelD": ["nu", "xi", "omicron", "pi"],
    }

    sample_text = "The quick brown fox jumps over the lazy dog."
    chosen = hybrid_select_model(sample_text, model_pool)
    print(f"Selected model: {chosen}")

    # Demonstrate the load‑model mock
    load_model_with_curvature(chosen, sample_text)

    # Print the diagnostic summary
    import json
    print("Hybrid summary:")
    print(json.dumps(hybrid_summary(), indent=2))