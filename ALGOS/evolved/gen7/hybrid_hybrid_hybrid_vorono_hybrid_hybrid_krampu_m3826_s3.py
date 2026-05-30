# DARWIN HAMMER — match 3826, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_geomet_m1462_s0.py (gen6)
# parent_b: hybrid_hybrid_krampus_brain_fractional_hdc_m240_s0.py (gen2)
# born: 2026-05-29T23:51:58Z

import math
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 format with trailing Z."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def shannon_entropy(values: np.ndarray) -> float:
    """Compute Shannon entropy (base‑2) of a non‑negative vector."""
    if values.size == 0:
        return 0.0
    total = values.sum()
    if total == 0.0:
        return 0.0
    probs = values / total
    mask = probs > 0
    return -float(np.sum(probs[mask] * np.log2(probs[mask])))


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


# ----------------------------------------------------------------------
# Deterministic pseudo‑random generator seeded from text
# ----------------------------------------------------------------------


def _rng_from_text(text: str) -> random.Random:
    """Create a deterministic RNG from a SHA‑256 hash of *text*."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Mix length to avoid collisions for very short strings
    seed_bytes = h[:8]
    seed = int.from_bytes(seed_bytes, "big") ^ len(text)
    return random.Random(seed)


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------


_FEATURE_KEYS: List[str] = [
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
    "rainmaker_corporate_grit_tension",
    "rainmaker_countdown_density",
    "rainmaker_asset_structuring_weight",
    "rainmaker_pitch_formatting_ratio",
    "telemetry_agent_symmetry_ratio",
    "telemetry_protocol_discipline",
    "telemetry_manic_velocity",
]


def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑features for demonstration purposes."""
    rng = _rng_from_text(text)
    return {k: rng.random() * 10.0 for k in _FEATURE_KEYS}


# ----------------------------------------------------------------------
# Core mathematical components
# ----------------------------------------------------------------------


def _clamp_ratio(ratio: float) -> float:
    """Clamp a ratio to the interval [0, 1] to avoid negative powers."""
    return max(0.0, min(1.0, ratio))


def hybrid_health_distance_score(
    endpoint: Dict[str, Any],
    point: np.ndarray,
    max_dist: float,
    feature_weights: np.ndarray,
) -> float:
    """
    Weighted geometric mean of endpoint attributes and distance term.
    The feature_weights vector (summing to 1) replaces the static
    ``weights`` list, allowing deeper integration of extracted features.
    """
    # unpack endpoint
    R = max(endpoint["reliability"], 0.0)
    P = max(endpoint["recovery_priority"], 0.0)
    sigma = max(endpoint["sphericity"], 1e-6)   # avoid zero
    phi = max(endpoint["flatness"], 1e-6)       # avoid division by zero

    # distance term
    d = np.linalg.norm(point - np.array(endpoint["seed"]))
    distance_ratio = _clamp_ratio(1.0 - d / max_dist)

    # unpack feature weights
    w_r, w_d, w_p, w_s, w_f = feature_weights

    # geometric mean (log‑space for stability)
    log_score = (
        w_r * math.log(R + 1e-12)
        + w_d * math.log(distance_ratio + 1e-12)
        + w_p * math.log(P + 1e-12)
        + w_s * math.log(sigma + 1e-12)
        + w_f * math.log(1.0 / phi + 1e-12)
    )
    return math.exp(log_score)


def effective_learning_rates(
    shannon_entropy_value: float,
    surrogate_prediction: float,
) -> Tuple[float, float]:
    """
    Produce two distinct learning rates.
    * eta_w* is boosted by entropy (exploration) and shrunk by the
      surrogate confidence (closer to 0 or 1 → lower learning).
    * eta_r* is the complement, encouraging stability when confidence is high.
    """
    base_lr = 0.05
    # map surrogate_prediction ∈ [0,1] to a confidence factor ∈ [0.5,1]
    confidence = 0.5 + 0.5 * sigmoid(10 * (surrogate_prediction - 0.5))
    eta_w = base_lr * (1.0 + shannon_entropy_value) * (1.0 - confidence)
    eta_r = base_lr * (1.0 + shannon_entropy_value) * confidence
    return eta_w, eta_r


def hybrid_update(
    endpoint: Dict[str, Any],
    point: np.ndarray,
    max_dist: float,
    feature_weights: np.ndarray,
    shannon_entropy_value: float,
    surrogate_prediction: float,
) -> Dict[str, Any]:
    """Perform a single update of the endpoint using the hybrid score."""
    eta_w, eta_r = effective_learning_rates(shannon_entropy_value, surrogate_prediction)
    score = hybrid_health_distance_score(endpoint, point, max_dist, feature_weights)

    # exponential moving average style update
    endpoint["reliability"] += eta_w * (score - endpoint["reliability"])
    endpoint["recovery_priority"] += eta_r * (score - endpoint["recovery_priority"])
    return endpoint


# ----------------------------------------------------------------------
# High‑level API
# ----------------------------------------------------------------------


def _normalize_features(values: np.ndarray) -> np.ndarray:
    """Scale a vector to sum to 1 (simple probability simplex)."""
    total = values.sum()
    if total == 0.0:
        # fallback to uniform distribution
        return np.full_like(values, 1.0 / values.size)
    return values / total


def extract_master_vector(text: str) -> Dict[str, float]:
    """
    Produce a 24‑dimensional master vector that fuses feature extraction
    with the hybrid Voronoi‑Geometric‑Decision engine.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Feature extraction
    # ------------------------------------------------------------------
    features = extract_full_features(text)

    # ------------------------------------------------------------------
    # 2️⃣ Build a high‑dimensional point from *all* numeric features
    # ------------------------------------------------------------------
    feature_array = np.array([features[k] for k in _FEATURE_KEYS], dtype=float)

    # Normalise to unit hyper‑cube to keep distances bounded
    point = feature_array / 10.0

    # ------------------------------------------------------------------
    # 3️⃣ Derive dynamic weights from a subset of semantic features
    # ------------------------------------------------------------------
    weight_source_keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "operator_ledger_density",
        "operator_recursion_score",
    ]
    raw_weights = np.array([features[k] for k in weight_source_keys], dtype=float)
    feature_weights = _normalize_features(raw_weights)

    # ------------------------------------------------------------------
    # 4️⃣ Compute entropy and surrogate prediction (placeholder model)
    # ------------------------------------------------------------------
    entropy_vals = np.array(
        [
            features["psyche_forensic_shield_ratio"],
            features["psyche_poetic_entropy"],
            features["psyche_dissociative_index"],
        ],
        dtype=float,
    )
    shannon_val = shannon_entropy(entropy_vals)

    # In a real system this would be the output of a learned surrogate;
    # here we map the mean of three resilience metrics into [0,1].
    resilience_keys = [
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
    ]
    surrogate_prediction = np.mean([features[k] for k in resilience_keys]) / 10.0

    # ------------------------------------------------------------------
    # 5️⃣ Initialise endpoint and run a single update
    # ------------------------------------------------------------------
    endpoint = {
        "reliability": 0.5,
        "recovery_priority": 0.5,
        "sphericity": 0.5,
        "flatness": 0.5,
        "seed": np.zeros_like(point),  # origin in the normalized space
    }

    max_dist = math.sqrt(point.size)  # diagonal of the unit hyper‑cube
    updated_endpoint = hybrid_update(
        endpoint,
        point,
        max_dist,
        feature_weights,
        shannon_val,
        surrogate_prediction,
    )

    # ------------------------------------------------------------------
    # 6️⃣ Assemble the master vector (includes updated endpoint params)
    # ------------------------------------------------------------------
    master_vector = {
        "reliability": updated_endpoint["reliability"],
        "recovery_priority": updated_endpoint["recovery_priority"],
        "sphericity": updated_endpoint["sphericity"],
        "flatness": updated_endpoint["flatness"],
        "shannon_entropy": shannon_val,
        "surrogate_prediction": surrogate_prediction,
    }
    # Append the original raw features for downstream interpretability
    master_vector.update(features)

    return master_vector


def main() -> None:
    sample_text = "This is a sample text for the improved hybrid algorithm."
    vec = extract_master_vector(sample_text)
    # pretty‑print a subset to avoid overwhelming the console
    for k in sorted(vec)[:10]:
        print(f"{k}: {vec[k]:.4f}")
    # optionally dump the full vector to a JSON file
    # Path("master_vector.json").write_text(json.dumps(vec, indent=2))


if __name__ == "__main__":
    main()