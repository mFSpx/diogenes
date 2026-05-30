# DARWIN HAMMER — match 5101, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_privac_m1448_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hdc_se_hybrid_infotaxis_hyb_m1784_s0.py (gen4)
# born: 2026-05-29T23:59:47Z

"""
Hybrid Algorithm: Fusion of hybrid_hybrid_hybrid_privac_m1448_s1.py (A) and
hybrid_hybrid_hybrid_hdc_se_hybrid_infotaxis_hyb_m1784_s0.py (B).

Mathematical Bridge
-------------------
Both parents define a *Morphology* dataclass and transform it into a high‑dimensional
vector representation:

* A – `lead_lag_transform` extracts linear and quadratic statistics from an
  arbitrary numeric matrix `X`.
* B – `morphology_vector` builds a random hyper‑dimensional vector seeded by the
  morphology’s physical parameters.

The fusion uses the **Kullback‑Leibler (KL) divergence** between the normalized
feature vector from A and the normalized morphology vector from B as a similarity
measure.  This KL term modulates the **Shannon entropy** of a set of actions,
while a **differential‑privacy (DP) reconstruction risk score** derived from the
morphology provides a recovery‑priority scaling factor `p ∈ [0,1]`.

The final hybrid affinity is

    h = p * (E + KL),

where `E` is the entropy of the action distribution and `KL` is the divergence
between the two vector spaces.  This single scalar can be used to rank
recovery candidates, guide decision hygiene, or drive circuit‑breaker logic.
"""

import sys
import math
import random
import hashlib
import pathlib
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Parent A utilities (trimmed to core needed functions)
# ----------------------------------------------------------------------
def lead_lag_transform(X: np.ndarray) -> np.ndarray:
    """
    Extract linear (sum) and quadratic (sum of squares) statistics per row.
    Returns a 1‑D vector of length 2 * n_rows.
    """
    linear_features = np.sum(X, axis=1)
    quadratic_features = np.sum(X ** 2, axis=1)
    return np.concatenate((linear_features, quadratic_features))


def reconstruction_risk_score(unique_quasi_identifiers: int,
                             total_records: int) -> float:
    """
    Simple DP‑inspired risk score: proportion of unique identifiers.
    Clipped to [0, 1].
    """
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_aggregate(values: List[float],
                 epsilon: float = 1.0) -> float:
    """
    Laplace mechanism for sum aggregation.
    """
    sensitivity = max(abs(v) for v in values) if values else 0.0
    scale = sensitivity / epsilon if epsilon > 0 else 0.0
    noise = random.laplacevariate(0.0, scale) if scale > 0 else 0.0
    return sum(values) + noise


# ----------------------------------------------------------------------
# Parent B utilities (core functions)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|"
    r"receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|"
    r"check|checked|audit)\b",
    re.I,
)


def random_vector(dim: int = 10000,
                  seed: str | int | None = None) -> List[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def morphology_vector(m: Morphology,
                      dim: int = 10000) -> np.ndarray:
    """
    Hyper‑dimensional vector seeded by the morphology’s physical parameters.
    The vector is scaled element‑wise by the morphology dimensions.
    """
    seed_bytes = hashlib.sha256(
        f"{m.length}{m.width}{m.height}{m.mass}".encode("utf-8")
    ).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    vec = np.array(random_vector(dim, seed))
    scale_factors = np.array(
        [m.length, m.width, m.height, m.mass] * (dim // 4 + 1)
    )[:dim]
    return vec * scale_factors


def sphericity_index(length: float, width: float, height: float) -> float:
    """
    Dimension‑agnostic shape metric: cubic root of volume divided by length.
    """
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """
    Ratio of smallest to largest dimension.
    """
    dims = sorted([length, width, height])
    return dims[0] / dims[-1] if dims[-1] != 0 else 0.0


# ----------------------------------------------------------------------
# Hybrid core functions (the mathematical fusion)
# ----------------------------------------------------------------------
def hybrid_feature_vector(morph: Morphology,
                          X: np.ndarray,
                          dim: int = 10000) -> np.ndarray:
    """
    Combine A's lead‑lag statistics with B's morphology hyper‑vector.
    The two vectors are L2‑normalized before concatenation.
    """
    # A‑side feature
    a_feat = lead_lag_transform(X)
    a_norm = a_feat / np.linalg.norm(a_feat) if np.linalg.norm(a_feat) != 0 else a_feat

    # B‑side feature
    b_feat = morphology_vector(morph, dim=dim)
    b_norm = b_feat / np.linalg.norm(b_feat) if np.linalg.norm(b_feat) != 0 else b_feat

    # Concatenate into a single hybrid representation
    return np.concatenate((a_norm, b_norm))


def action_entropy(actions: List[MathAction]) -> float:
    """
    Compute Shannon entropy of a probability distribution derived from the
    expected values of actions. Expected values are shifted to be non‑negative
    and then normalized.
    """
    if not actions:
        return 0.0
    vals = np.array([a.expected_value for a in actions], dtype=float)
    # Shift to non‑negative
    vals -= vals.min()
    total = vals.sum()
    if total == 0:
        probs = np.full_like(vals, 1.0 / len(vals))
    else:
        probs = vals / total
    # Shannon entropy
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(entropy)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL(p‖q) where p and q are probability vectors.
    Both vectors are assumed to be non‑negative and sum to 1.
    """
    eps = 1e-12
    p_safe = p + eps
    q_safe = q + eps
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def hybrid_affinity(actions: List[MathAction],
                    morph: Morphology,
                    X: np.ndarray,
                    dim: int = 10000) -> float:
    """
    Full hybrid affinity:
        h = p * (E + KL)

    * `p` – reconstruction risk score derived from morphology surface metrics.
    * `E` – Shannon entropy of the action distribution.
    * `KL` – divergence between the normalized lead‑lag vector and the normalized
      morphology hyper‑vector.
    """
    # 1️⃣ Recovery priority `p`
    # Use surface‑area‑like proxy as unique quasi‑identifiers count.
    uq = int(
        (morph.length * morph.width + morph.width * morph.height +
         morph.height * morph.length) * 10
    )
    total_records = int((morph.length + morph.width + morph.height + morph.mass) * 10)
    p = reconstruction_risk_score(uq, total_records)

    # 2️⃣ Entropy `E`
    E = action_entropy(actions)

    # 3️⃣ KL divergence between A‑side and B‑side normalized vectors
    hybrid_vec = hybrid_feature_vector(morph, X, dim=dim)
    # Split back into the two halves (they were concatenated after normalization)
    split = len(hybrid_vec) // 2
    p_vec = hybrid_vec[:split]
    q_vec = hybrid_vec[split:]
    KL = kl_divergence(p_vec, q_vec)

    # Final affinity
    return p * (E + KL)


def prioritize_actions(actions: List[MathAction],
                       morph: Morphology,
                       X: np.ndarray,
                       dim: int = 10000) -> List[Tuple[MathAction, float]]:
    """
    Rank actions by descending hybrid affinity. Returns a list of tuples
    (action, score).
    """
    base_affinity = hybrid_affinity(actions, morph, X, dim=dim)
    # For demonstration we modulate each action's score by its individual cost
    # and risk to obtain a per‑action hybrid score.
    ranked = []
    for act in actions:
        individual_factor = 1.0
        if act.cost > 0:
            individual_factor /= (1.0 + act.cost)
        if act.risk > 0:
            individual_factor /= (1.0 + act.risk)
        score = base_affinity * individual_factor
        ranked.append((act, score))
    ranked.sort(key=lambda tup: tup[1], reverse=True)
    return ranked


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create dummy morphology
    morph = Morphology(length=2.5, width=1.8, height=0.9, mass=3.2)

    # Dummy numeric matrix X (e.g., sensor readings)
    X = np.array([[0.2, 0.5, 0.3],
                  [0.1, 0.4, 0.6],
                  [0.7, 0.2, 0.1]])

    # Define a few actions
    actions = [
        MathAction(id="A1", expected_value=10.0, cost=1.2, risk=0.3),
        MathAction(id="A2", expected_value=5.0, cost=0.5, risk=0.1),
        MathAction(id="A3", expected_value=8.0, cost=0.9, risk=0.2),
    ]

    # Compute hybrid affinity
    affinity = hybrid_affinity(actions, morph, X)
    print(f"Hybrid affinity (scalar): {affinity:.6f}")

    # Rank actions
    ranked = prioritize_actions(actions, morph, X)
    print("\nRanked actions (action id, hybrid score):")
    for act, score in ranked:
        print(f"  {act.id}: {score:.6f}")

    # Verify that KL divergence is finite and non‑negative
    hybrid_vec = hybrid_feature_vector(morph, X)
    split = len(hybrid_vec) // 2
    kl_val = kl_divergence(hybrid_vec[:split], hybrid_vec[split:])
    assert kl_val >= 0.0, "KL divergence should be non‑negative"
    print(f"\nKL divergence between feature halves: {kl_val:.6f}")

    sys.exit(0)