# DARWIN HAMMER — match 3750, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m1227_s3.py (gen5)
# born: 2026-05-29T23:51:26Z

"""Hybrid Path‑Signature Regret Bandit (Hybrid A + B)

This module fuses the feature‑extraction / master‑vector logic of *parent algorithm A*
with the Fisher‑score‑adjusted regret / SSIM similarity machinery of *parent algorithm B*.

Mathematical bridge:
- The master vector is interpreted as a point in ℝⁿ and, over a synthetic time axis,
  as a piece‑wise linear *path* 𝑝(t).  The iterated‑integral (signature) of 𝑝 up to
  order 2 yields a feature tensor S(𝑝) = (∫𝑑𝑝, ∫𝑑𝑝⊗𝑑𝑝) ∈ ℝⁿ×ℝⁿˣⁿ.
- Historical signatures are stored.  Similarity between the current signature and a
  historical one is measured with the Structural‑Similarity Index (SSIM) applied to
  their flattened vector representations.
- The SSIM value (θ) is fed to a Fisher‑score kernel (Gaussian‑beam derivative) to
  obtain an *information‑weight* w = FisherScore(θ; μ=0.5, σ=0.1).  This weight scales
  the regret (negative expected reward) of each candidate BanditAction, producing a
  Fisher‑adjusted propensity used for selection.

The resulting system can rank actions based on how well the current feature‑path
matches past successful signatures, while automatically modulating confidence
through the Fisher information of the similarity measure.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Parent A – feature extraction (kept unchanged)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> dict:
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_recursion_score", "operator_directive_ratio",
        "operator_target_density", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "psyche_wrath_velocity", "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric", "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index", "resilience_conspiracy_grounding_ratio",
        "resilience_chaotic_good_tax", "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density", "rainmaker_asset_structuring_weight",
        "rainmaker_pitch_formatting_ratio", "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline", "telemetry_manic_velocity",
    ]
    return {k: rnd.random() * 10 for k in keys}

def extract_master_vector(text: str) -> dict:
    if not text.strip():
        return {}
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "ledger_density": f.get("operator_ledger_density", 0.0),
        "recursion_score": f.get("operator_recursion_score", 0.0)
    }

# ----------------------------------------------------------------------
# Parent B – Gaussian beam, Fisher score, SSIM, Bandit structures
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx ** 2 + my ** 2 + c1) *
                                                    (vx + vy + c2))

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # probability‑like weight after Fisher adjustment
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridPSRegret"

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def _vector_to_array(vec: Dict[str, float]) -> np.ndarray:
    """Deterministic ordering of master‑vector keys to a 1‑D NumPy array."""
    ordered_keys = ["visceral_ratio", "tech_ratio",
                    "legal_osint_ratio", "ledger_density",
                    "recursion_score"]
    return np.array([vec.get(k, 0.0) for k in ordered_keys], dtype=float)

def generate_path_from_vector(base_vec: Dict[str, float],
                              steps: int = 10,
                              noise_scale: float = 0.05) -> List[np.ndarray]:
    """
    Produce a synthetic piece‑wise linear path by adding small Gaussian noise
    to the base master vector at each time step.
    Returns a list of points (arrays) of length `steps`.
    """
    base_arr = _vector_to_array(base_vec)
    rng = np.random.default_rng(seed=hash(tuple(base_arr)))
    path = []
    for i in range(steps):
        noise = rng.normal(scale=noise_scale, size=base_arr.shape)
        point = base_arr + noise
        path.append(point)
    return path

def compute_path_signature(path: List[np.ndarray]) -> np.ndarray:
    """
    Compute a simple level‑2 signature for a piece‑wise linear path.
    - Level‑1: cumulative sum of increments (Δp)
    - Level‑2: cumulative sum of outer products of increments (Δp ⊗ Δp)
    The result is a flattened 1‑D array: [lvl1, lvl2_flat]
    """
    if len(path) < 2:
        raise ValueError("Path must contain at least two points")
    increments = [path[i+1] - path[i] for i in range(len(path)-1)]
    level1 = np.sum(increments, axis=0)                     # shape (n,)
    level2 = np.sum([np.outer(d, d) for d in increments], axis=0)  # shape (n,n)
    return np.concatenate([level1.ravel(), level2.ravel()])

def select_bandit_action(current_signature: np.ndarray,
                         history_signatures: List[np.ndarray],
                         candidate_actions: List[MathAction],
                         fisher_center: float = 0.5,
                         fisher_width: float = 0.1) -> List[BanditAction]:
    """
    For each candidate action:
    1. Compute SSIM similarity between the current signature and the *most similar*
       historical signature (max SSIM).
    2. Convert similarity (θ ∈ [0,1]) to a Fisher information weight.
    3. Combine the weight with the raw expected value of the action to obtain a
       propensity (softmax‑like scaling) and a confidence bound.
    Returns a list of BanditAction objects.
    """
    if not history_signatures:
        raise ValueError("History signatures list cannot be empty")
    # Pre‑compute SSIM between current and each historical signature
    sims = [ssim(current_signature, h) for h in history_signatures]
    max_sim = max(sims)  # the best match informs overall confidence
    # Fisher weight for the best similarity
    fisher_w = fisher_score(max_sim, fisher_center, fisher_width)

    bandit_actions = []
    # Normalise raw expected values to avoid overflow in softmax
    raw_vals = np.array([a.expected_value for a in candidate_actions], dtype=float)
    max_raw = np.max(raw_vals) if raw_vals.size else 0.0
    exp_vals = np.exp(raw_vals - max_raw)  # stable softmax numerator
    softmax = exp_vals / (exp_vals.sum() + 1e-12)

    for idx, action in enumerate(candidate_actions):
        propensity = softmax[idx] * fisher_w
        # confidence bound loosely modelled as inverse of variance of similarity
        conf_bound = 1.0 / (np.var(sims) + 1e-6)
        bandit_actions.append(
            BanditAction(
                action_id=action.id,
                propensity=propensity,
                expected_reward=action.expected_value,
                confidence_bound=conf_bound,
                algorithm="HybridPSRegret"
            )
        )
    return bandit_actions

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample input text
    sample_text = "The quick brown fox jumps over the lazy dog."
    master_vec = extract_master_vector(sample_text)

    # Build a synthetic path and its signature
    path = generate_path_from_vector(master_vec, steps=12, noise_scale=0.02)
    cur_sig = compute_path_signature(path)

    # Mock historical signatures (perturbed versions of current signature)
    rng = np.random.default_rng(42)
    history = [cur_sig + rng.normal(scale=0.1, size=cur_sig.shape) for _ in range(5)]

    # Define candidate actions
    actions = [
        MathAction(id="A1", expected_value=1.2, cost=0.1, risk=0.05),
        MathAction(id="A2", expected_value=0.8, cost=0.2, risk=0.02),
        MathAction(id="A3", expected_value=1.5, cost=0.15, risk=0.07)
    ]

    # Run hybrid bandit selection
    selected = select_bandit_action(cur_sig, history, actions)

    # Print results
    for ba in selected:
        print(f"Action {ba.action_id}: propensity={ba.propensity:.4f}, "
              f"expected_reward={ba.expected_reward:.2f}, "
              f"confidence_bound={ba.confidence_bound:.4f}")