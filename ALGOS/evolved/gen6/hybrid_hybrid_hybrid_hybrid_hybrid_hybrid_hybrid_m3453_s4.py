# DARWIN HAMMER — match 3453, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2409_s1.py (gen5)
# born: 2026-05-29T23:50:11Z

"""Hybrid Algorithm integrating State‑Space Similarity (SSIM) & Morphology from
parent A (hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s2) with
Bandit decision‑making and Radial‑Basis‑Function (RBF) similarity from
parent B (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2409_s1).

Mathematical bridge
-------------------
* Parent A provides a *state representation* (morphology‑driven indices) and a
  structural‑similarity metric `ssim(x, y)`.  
* Parent B models *feature similarity* with a Gaussian RBF
  `g(r) = exp(-(ε·r)²)` and uses a contextual bandit whose propensity
  `π` is adapted by the Shannon entropy of the feature distribution.

The fusion treats the SSIM value as a *distance‑like* quantity:

r = 1 - ssim(x, y)        # 0 ≤ r ≤ 1

and feeds it into the Gaussian RBF. The resulting kernel weight modulates
the bandit propensity, while the morphology‑derived righting‑time index
scales the learning‑rate used in the bandit update. This yields a single
unified decision score that simultaneously accounts for physical morphology,
image‑style similarity, and information‑theoretic confidence.

The module implements three core hybrid operations:
1. `state_similarity_kernel` – SSIM → RBF kernel.
2. `hybrid_bandit_score` – combines bandit propensity, kernel weight and
   Shannon entropy into a decision score.
3. `hybrid_bandit_update` – updates the bandit action using a learning rate
   derived from morphology (right‑time index).
"""

import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Structures from Parent A
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


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

    def as_dict(self) -> Dict[str, any]:
        d = asdict(self)
        d["morphology"] = asdict(self.morphology)
        return d


# ----------------------------------------------------------------------
# Structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # prior probability of selection
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


def c_to_k(celsius: float) -> float:
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Simplified Schoolfield model – returns a temperature‑scaled activity factor."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    # Using a very simple Arrhenius‑like scaling for demonstration
    ea = 50000.0  # activation energy (J/mol) – placeholder
    return params.rho_25 * math.exp(-ea / (params.r_cal * temp_k))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def shannon_entropy(p: np.ndarray) -> float:
    """Entropy H(p) = - Σ p_i log2 p_i . p must sum to 1."""
    if np.any(p < 0):
        raise ValueError("probabilities must be non‑negative")
    p = p[p > 0]  # ignore zero entries to avoid log(0)
    return -np.sum(p * np.log2(p))


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def ssim(x: List[float], y: List[float], dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Simplified SSIM for 1‑D signals."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("inputs must have the same shape")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / denominator)


def state_similarity_kernel(state_a: List[float],
                           state_b: List[float],
                           epsilon: float = 1.0) -> float:
    """
    Compute the hybrid kernel:
        r = 1 - SSIM(state_a, state_b)
        k = Gaussian(r, epsilon)
    The kernel lies in [0, 1] and quantifies morphological‑state similarity.
    """
    s = ssim(state_a, state_b)
    r = 1.0 - s  # distance‑like quantity
    return gaussian(r, epsilon)


def hybrid_bandit_score(context_features: List[float],
                        actions: List[BanditAction],
                        temperature_c: float = 25.0,
                        epsilon: float = 1.0) -> List[Tuple[BanditAction, float]]:
    """
    Produce a scored list of actions.
    Score_i = propensity_i * kernel_i * (1 + entropy_factor)

    where:
        kernel_i = state_similarity_kernel(context_features, action_feature_i)
        entropy_factor = shannon_entropy(normalized(context_features))
    """
    # Normalise context features to a probability distribution
    ctx_arr = np.asarray(context_features, dtype=float)
    if np.allclose(ctx_arr, 0):
        prob_ctx = np.ones_like(ctx_arr) / ctx_arr.size
    else:
        prob_ctx = ctx_arr / ctx_arr.sum()
    entropy = shannon_entropy(prob_ctx)

    temp_k = c_to_k(temperature_c)
    dev_rate = developmental_rate(temp_k)

    scored = []
    for act in actions:
        # For illustration we treat the action_id string as a seed to generate a deterministic feature vector
        random.seed(act.action_id)
        act_feat = [random.random() for _ in range(len(context_features))]
        kernel = state_similarity_kernel(context_features, act_feat, epsilon)
        # Combine propensity, kernel, entropy, and temperature scaling
        score = act.propensity * kernel * (1.0 + entropy) * dev_rate
        scored.append((act, score))
    # Sort descending by score
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def hybrid_bandit_update(action: BanditAction,
                         update: BanditUpdate,
                         morphology: Morphology,
                         learning_rate_base: float = 0.1) -> BanditAction:
    """
    Update the expected reward of a BanditAction using a learning rate that
    is modulated by the morphology‑derived righting‑time index.

    lr = base_lr * recovery_priority(morphology)
    new_reward = old_reward + lr * (reward - old_reward)
    """
    priority = recovery_priority(morphology)
    lr = learning_rate_base * priority
    delta = update.reward - action.expected_reward
    new_reward = action.expected_reward + lr * delta

    # Confidence bound shrinks with more information (simple heuristic)
    new_conf = max(0.0, action.confidence_bound - lr * 0.05)

    return BanditAction(
        action_id=action.action_id,
        propensity=action.propensity,          # unchanged in this simple update
        expected_reward=new_reward,
        confidence_bound=new_conf,
        algorithm=action.algorithm
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy morphology
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # Engine endpoint (unused in this test but validates dataclass)
    endpoint = EngineEndpoint(
        engine_id="eng01",
        channel="alpha",
        residency="local",
        runtime="python3.11",
        resource_class="standard",
        always_on=True,
        endpoint="http://localhost",
        capabilities=["compute", "store"],
        morphology=morph,
    )
    print("Endpoint dict:", endpoint.as_dict())

    # Define a context feature vector (e.g., sensor readings)
    context = [0.2, 0.5, 0.7, 0.1]

    # Define a few bandit actions
    actions = [
        BanditAction(action_id="A", propensity=0.4, expected_reward=0.5,
                     confidence_bound=0.2, algorithm="hybrid"),
        BanditAction(action_id="B", propensity=0.3, expected_reward=0.6,
                     confidence_bound=0.25, algorithm="hybrid"),
        BanditAction(action_id="C", propensity=0.3, expected_reward=0.4,
                     confidence_bound=0.15, algorithm="hybrid"),
    ]

    # Compute hybrid scores
    scored_actions = hybrid_bandit_score(context, actions, temperature_c=22.0, epsilon=1.5)
    print("\nHybrid action scores (sorted):")
    for act, score in scored_actions:
        print(f"  Action {act.action_id}: score={score:.4f}, exp_reward={act.expected_reward:.3f}")

    # Simulate receiving a reward for the top‑scoring action
    top_action, _ = scored_actions[0]
    reward = random.uniform(0, 1)  # mock reward
    update = BanditUpdate(context_id="ctx1", action_id=top_action.action_id,
                          reward=reward, propensity=top_action.propensity)

    # Update the action
    updated_action = hybrid_bandit_update(top_action, update, morphology=morph)
    print(f"\nUpdated action {updated_action.action_id}:")
    print(f"  old reward={top_action.expected_reward:.3f}, new reward={updated_action.expected_reward:.3f}")
    print(f"  old confidence={top_action.confidence_bound:.3f}, new confidence={updated_action.confidence_bound:.3f}")

    sys.exit(0)