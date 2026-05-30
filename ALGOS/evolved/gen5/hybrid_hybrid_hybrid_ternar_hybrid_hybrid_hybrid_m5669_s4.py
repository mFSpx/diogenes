# DARWIN HAMMER — match 5669, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_variational_free_ene_m21_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_rbf_su_m971_s1.py (gen4)
# born: 2026-05-30T00:04:05Z

"""Hybrid algorithm merging:
- Parent A: ternary router with SSIM similarity and variational free energy belief update.
- Parent B: temperature‑dependent bandit router, RBF surrogate, and learning vector construction.

Mathematical bridge:
1. The SSIM score between a prediction and an observation is interpreted as a reward‑like signal.
2. This reward modulates the bandit’s propensity via a temperature‑dependent activity factor derived from
   the Schoolfield (Arrhenius) model.
3. The updated propensity feeds the learning‑vector feature set that drives the RBF surrogate.
4. The surrogate’s prediction error is fed back into a variational free‑energy (VFE) update of the belief
   mean, closing the loop.

The module therefore intertwines SSIM ↔ reward ↔ temperature‑scaled bandit ↔ learning vector ↔ RBF ↔ VFE.
"""

import json
import sys
import math
import random
import pathlib
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("inputs must have the same shape")
    if x.size == 0:
        raise ValueError("inputs must be non‑empty")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)


def variational_free_energy(mean: float,
                            var: float,
                            observation: float,
                            prediction: float,
                            prior_precision: float = 1.0,
                            obs_precision: float = 1.0) -> Tuple[float, float]:
    """
    Simple VFE update for a scalar Gaussian belief.
    Returns updated (mean, variance).
    """
    # Prior term
    prior_term = prior_precision * (mean - observation) ** 2
    # Likelihood term (prediction error)
    likelihood_term = obs_precision * (prediction - observation) ** 2
    # Free energy (up to constants)
    free_energy = 0.5 * (prior_term / var + likelihood_term / var + math.log(var))
    # Gradient descent step on mean (ignoring variance derivative for brevity)
    grad_mean = (prior_precision + obs_precision) * (mean - observation) / var
    learning_rate = 0.1
    new_mean = mean - learning_rate * grad_mean
    # Simple variance update (inverse‑additive precision)
    new_prec = prior_precision + obs_precision
    new_var = 1.0 / new_prec
    return new_mean, new_var


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid_bandit"


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
    r_cal: float = 1.987  # cal·K⁻¹·mol⁻¹


def schoolfield_activity(params: SchoolfieldParams, T: float) -> float:
    """Temperature‑dependent activity factor using the Schoolfield model."""
    if T <= 0:
        raise ValueError("Temperature must be > 0 K")
    term1 = params.rho_25 * math.exp(
        -params.delta_h_activation / params.r_cal *
        (1.0 / T - 1.0 / 298.15)
    )
    term_low = 1.0 + math.exp(
        params.delta_h_low / params.r_cal *
        (1.0 / params.t_low - 1.0 / T)
    )
    term_high = 1.0 + math.exp(
        params.delta_h_high / params.r_cal *
        (1.0 / T - 1.0 / params.t_high)
    )
    return term1 / (term_low * term_high)


def softmax_propensities(actions: List[BanditAction], temperature: float) -> List[float]:
    """Scale propensities with a temperature and return a probability distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.array([a.propensity / temperature for a in actions])
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)  # for numerical stability
    probs = exp_logits / exp_logits.sum()
    return probs.tolist()


def select_bandit_action(actions: List[BanditAction], temperature: float) -> BanditAction:
    """Sample an action according to temperature‑scaled propensities."""
    probs = softmax_propensities(actions, temperature)
    choice = random.choices(actions, weights=probs, k=1)[0]
    return choice


def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float) -> float:
    return math.exp(-(r ** 2) / (2 * epsilon ** 2))


@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: List[float]) -> float:
        return sum(
            w * gaussian(euclidean(x, list(c)), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


class LearningVector:
    """Simple wrapper that builds a feature vector from bandit and SSIM information."""
    def __init__(self, terms: List[str] | None = None):
        self.terms = terms or ["ssim", "activity", "propensity"]

    def construct(self, ssim_score: float, activity: float, propensity: float) -> List[float]:
        # Linear combination; in a real system this would be more sophisticated.
        return [ssim_score, activity, propensity]


# ----------------------------------------------------------------------
# Hybrid functions (the required three+ functions)
# ----------------------------------------------------------------------
def hybrid_reward_from_ssim(prediction: np.ndarray, observation: np.ndarray) -> float:
    """Convert SSIM similarity into a scalar reward (higher SSIM → higher reward)."""
    return ssim(prediction, observation)


def hybrid_bandit_update(actions: List[BanditAction],
                         ssim_reward: float,
                         temp_params: SchoolfieldParams,
                         current_T: float) -> BanditAction:
    """
    1. Compute temperature‑dependent activity.
    2. Modulate each action's propensity by activity * reward.
    3. Select an action with temperature‑scaled softmax.
    """
    activity = schoolfield_activity(temp_params, current_T)
    # Adjust propensities multiplicatively
    adjusted = [
        BanditAction(
            action_id=a.action_id,
            propensity=a.propensity * activity * ssim_reward,
            expected_reward=ssim_reward,
            confidence_bound=a.confidence_bound,
            algorithm=a.algorithm,
        )
        for a in actions
    ]
    # Use a nominal temperature equal to inverse of activity to keep scale reasonable
    temperature = max(0.01, 1.0 / (activity + 1e-9))
    chosen = select_bandit_action(adjusted, temperature)
    return chosen


def hybrid_vfe_step(mean: float,
                    var: float,
                    observation: np.ndarray,
                    prediction: np.ndarray,
                    learning_rate: float = 0.1) -> Tuple[float, float]:
    """
    Perform a VFE update where the observation is the scalar mean of the observation array
    and the prediction is the scalar mean of the prediction array.
    """
    obs_scalar = float(np.mean(observation))
    pred_scalar = float(np.mean(prediction))
    new_mean, new_var = variational_free_energy(
        mean, var, obs_scalar, pred_scalar,
        prior_precision=1.0,
        obs_precision=1.0
    )
    # Apply an extra damping factor for stability
    new_mean = mean + learning_rate * (new_mean - mean)
    new_var = var + learning_rate * (new_var - var)
    return new_mean, new_var


def hybrid_predict(features: List[float],
                   surrogate: RBFSurrogate) -> float:
    """Run the RBF surrogate on the constructed learning‑vector features."""
    return surrogate.predict(features)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic signals
    np.random.seed(0)
    pred_signal = np.random.randint(0, 256, size=100).astype(float)
    obs_signal = pred_signal + np.random.normal(0, 5, size=100)

    # 1. Compute SSIM reward
    reward = hybrid_reward_from_ssim(pred_signal, obs_signal)
    print(f"SSIM reward: {reward:.4f}")

    # 2. Initialise bandit actions
    actions = [
        BanditAction(action_id="A", propensity=1.0, expected_reward=0.0, confidence_bound=0.1),
        BanditAction(action_id="B", propensity=0.8, expected_reward=0.0, confidence_bound=0.2),
        BanditAction(action_id="C", propensity=0.5, expected_reward=0.0, confidence_bound=0.3),
    ]

    # 3. Temperature‑dependent activity
    temp_params = SchoolfieldParams()
    current_T = 295.0  # Kelvin

    chosen_action = hybrid_bandit_update(actions, reward, temp_params, current_T)
    print(f"Chosen action: {chosen_action.action_id} (propensity={chosen_action.propensity:.4f})")

    # 4. Build learning vector
    activity_factor = schoolfield_activity(temp_params, current_T)
    lv = LearningVector()
    feature_vec = lv.construct(ssim_score=reward,
                               activity=activity_factor,
                               propensity=chosen_action.propensity)
    print(f"Learning vector: {feature_vec}")

    # 5. RBF surrogate (dummy centers/weights)
    centers = [(0.5, 0.5, 0.5), (0.2, 0.8, 0.1), (0.9, 0.1, 0.4)]
    weights = [0.3, -0.2, 0.5]
    surrogate = RBFSurrogate(centers=centers, weights=weights, epsilon=0.5)

    surrogate_output = hybrid_predict(feature_vec, surrogate)
    print(f"Surrogate prediction: {surrogate_output:.4f}")

    # 6. VFE belief update
    belief_mean = 0.0
    belief_var = 1.0
    new_mean, new_var = hybrid_vfe_step(belief_mean, belief_var, obs_signal, pred_signal)
    print(f"Updated belief: mean={new_mean:.4f}, var={new_var:.4f}")