# DARWIN HAMMER — match 4634, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_percep_hybrid_bandit_router_m1637_s0.py (gen5)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m278_s1.py (gen4)
# born: 2026-05-29T23:57:10Z

import math
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple, List

Vector = Sequence[float]

def gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    """Standard Gaussian probability density function."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Beam intensity modeled as a Gaussian pdf (normalized to its peak)."""
    intensity = gaussian_pdf(theta, center, width)
    # Normalise by the peak value (sigma -> 0 gives peak = 1/(sigma*sqrt(2π)))
    peak = gaussian_pdf(center, center, width)
    return intensity / peak if peak > 0 else 0.0

def fisher_information(sigma: float) -> float:
    """Fisher information for a Gaussian with known mean and variance sigma^2."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return 1.0 / (sigma ** 2)

@dataclass
class Entity:
    spatial_load: float
    cognitive_risk: float

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(np.dot(weights, x))

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
    signal_intensity: float = 1.0,
    risk_factor: float = 1.0,
) -> np.ndarray:
    """
    Normalised LMS update with adaptive step size.
    The step size is attenuated by the cognitive risk (higher risk → smaller updates)
    and scaled by the signal intensity from the Fisher localisation module.
    """
    pred = nlms_predict(weights, x)
    error = target - pred
    # Adaptive learning rate
    adaptive_mu = mu * signal_intensity / (risk_factor + eps)
    denom = eps + np.dot(x, x)
    delta = adaptive_mu * error * x / denom
    return weights + delta

def hybrid_fusion(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    theta: float,
    center: float,
    width: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, Entity]:
    """
    Fuse NLMS with Fisher localisation.
    Returns updated weights and an Entity containing the spatial load (signal intensity)
    and cognitive risk (inverse Fisher information).
    """
    intensity = gaussian_beam(theta, center, width)
    risk = fisher_information(width)  # higher width → lower information → higher risk
    updated_weights = nlms_update(
        weights,
        x,
        target,
        mu=mu,
        eps=eps,
        signal_intensity=intensity,
        risk_factor=risk,
    )
    entity = Entity(spatial_load=intensity, cognitive_risk=risk)
    return updated_weights, entity

def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    shifted = values - np.max(values)
    exp_vals = np.exp(shifted / temperature)
    return exp_vals / np.sum(exp_vals)

def bandit_select_action(propensities: np.ndarray, temperature: float = 0.1) -> int:
    """Select an action using a softmax policy."""
    probs = softmax(propensities, temperature)
    return int(np.random.choice(len(propensities), p=probs))

def hybrid_bandit_fusion(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    theta: float,
    center: float,
    width: float,
    propensities: np.ndarray,
    mu: float = 0.5,
    eps: float = 1e-9,
    eta: float = 0.1,
    temperature: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, Entity, int]:
    """
    Combine NLMS, Fisher localisation and a contextual bandit.
    The cognitive risk modulates both the NLMS step size and the bandit learning rate.
    """
    intensity = gaussian_beam(theta, center, width)
    risk = fisher_information(width)

    # NLMS update with risk‑aware step size
    updated_weights = nlms_update(
        weights,
        x,
        target,
        mu=mu,
        eps=eps,
        signal_intensity=intensity,
        risk_factor=risk,
    )

    # Bandit reward based on prediction error (negative MSE)
    pred = nlms_predict(weights, x)
    error = target - pred
    reward = -error ** 2

    # Risk‑aware exponential weighting update
    propensities = propensities * np.exp(eta * reward / (risk + eps))

    # Action selection
    action = bandit_select_action(propensities, temperature)

    entity = Entity(spatial_load=intensity, cognitive_risk=risk)
    return updated_weights, propensities, entity, action

if __name__ == "__main__":
    np.random.seed(0)

    # Example dimensions
    dim = 5
    weights = np.random.rand(dim)
    x = np.random.rand(dim)
    target = 10.0
    theta = 0.5
    center = 0.0
    width = 1.0
    propensities = np.random.rand(dim)

    # Hybrid NLMS‑Fisher fusion
    w_fused, entity_fused = hybrid_fusion(
        weights, x, target, theta, center, width
    )
    print("Updated Weights (fusion):", w_fused)
    print("Entity (fusion):", entity_fused)

    # Hybrid bandit fusion
    w_bandit, prop_updated, entity_bandit, act = hybrid_bandit_fusion(
        weights,
        x,
        target,
        theta,
        center,
        width,
        propensities,
    )
    print("\nUpdated Weights (bandit):", w_bandit)
    print("Updated Propensities:", prop_updated)
    print("Entity (bandit):", entity_bandit)
    print("Selected Action:", act)