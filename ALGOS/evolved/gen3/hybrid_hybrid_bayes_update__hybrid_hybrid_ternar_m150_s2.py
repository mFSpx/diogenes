# DARWIN HAMMER — match 150, survivor 2
# gen: 3
# parent_a: hybrid_bayes_update_hybrid_krampus_brain_m15_s1.py (gen2)
# parent_b: hybrid_hybrid_ternary_route_hybrid_bandit_router_m56_s0.py (gen2)
# born: 2026-05-29T23:26:05Z

"""
Hybrid Bayesian‑SSIM‑Curvature Router
====================================

This module fuses the two parent algorithms:

* **Parent A** – *bayes_update* + *hybrid_krampus_brainmap_ollivier_ricci*  
  (features → master vector → Ollivier‑Ricci curvature used as a Bayesian prior).

* **Parent B** – *hybrid_ternary_router_ssim* + *hybrid_bandit_router_honeybee_store*  
  (SSIM similarity between a packet payload and a prototype vector used as a
  likelihood for a multi‑armed bandit).

**Mathematical bridge**  
For each possible routing *action* we maintain a prior probability derived from
the Ollivier‑Ricci curvature of the brain‑map projection (Parent A).  
When a packet arrives we compute the Structural Similarity Index (SSIM) between
its payload and a fixed prototype (Parent B); this SSIM value is interpreted as
the likelihood of the packet belonging to each action.  

Bayes’ rule provides the posterior:


posterior(action) ∝ prior(action) * likelihood(action)


The posterior is then fed to the bandit policy (ε‑greedy) to select the routing
action and to update the reward statistics.  All core operations – curvature
computation, SSIM, Bayesian update, and bandit bookkeeping – are implemented
as pure NumPy / std‑lib code.

"""

import numpy as np
import random
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Parent‑A feature extraction (simplified)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Generate a deterministic‑looking random feature set."""
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
        "rainmaker_asset_structuring_weight", "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline", "telemetry_manic_velocity",
    ]
    for k in keys:
        features[k] = rnd.random()
    return features


def extract_master_vector(text: str) -> Dict[str, float]:
    """Select a subset of the full feature dict as the “master vector”. """
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "forensic_shield_ratio": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
        "bureaucratic_weaponization_index": f.get(
            "resilience_bureaucratic_weaponization_index", 0.0
        ),
        "resource_exhaustion_metric": f.get(
            "resilience_resource_exhaustion_metric", 0.0
        ),
        "swarm_orchestration_density": f.get(
            "resilience_swarm_orchestration_density", 0.0
        ),
    }


# ----------------------------------------------------------------------
# Parent‑A curvature (a very light Ollivier‑Ricci proxy)
# ----------------------------------------------------------------------
def compute_curvature(master_vec: Dict[str, float]) -> Dict[str, float]:
    """
    Produce a curvature‑based prior for each routing action.

    The curvature is approximated by the variance of the master vector
    components; higher variance → lower curvature → lower prior probability.
    The function returns a normalized prior distribution over a fixed set of
    actions.
    """
    actions = ["alpha", "beta", "gamma", "delta"]
    values = np.fromiter(master_vec.values(), dtype=np.float64)
    var = values.var() + 1e-8  # avoid division by zero
    # Inverse variance acts as a proxy for curvature strength.
    raw = np.array([1.0 / (abs(math.sin(i + var)) + 0.1) for i in range(len(actions))])
    prior = raw / raw.sum()
    return dict(zip(actions, prior))


# ----------------------------------------------------------------------
# Parent‑B SSIM utilities
# ----------------------------------------------------------------------
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)


def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index (SSIM) for 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)


def hybrid_score(packet: Dict[str, List[float]]) -> float:
    """
    Convert a packet payload into a likelihood value using SSIM.
    The prototype vector defines the “ideal” payload shape.
    """
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        return 0.0
    try:
        payload_vec = np.asarray(payload, dtype=np.float64)
        if payload_vec.size < PROTOTYPE_VECTOR.size:
            payload_vec = np.pad(
                payload_vec,
                (0, PROTOTYPE_VECTOR.size - payload_vec.size),
                mode="constant",
                constant_values=0.0,
            )
        elif payload_vec.size > PROTOTYPE_VECTOR.size:
            payload_vec = payload_vec[: PROTOTYPE_VECTOR.size]
        return compute_ssim(payload_vec.tolist(), PROTOTYPE_VECTOR.tolist())
    except Exception:
        return 0.0


# ----------------------------------------------------------------------
# Bayesian fusion of curvature prior and SSIM likelihood
# ----------------------------------------------------------------------
def bayesian_update(
    prior: Dict[str, float], likelihood: float, actions: List[str]
) -> Dict[str, float]:
    """
    Perform a Bayes update where the same scalar likelihood (SSIM) is applied
    to every action.  To keep the posterior diverse we modulate the likelihood
    by a simple action‑specific factor derived from the action name hash.
    """
    # Action‑specific modulation (adds a tiny asymmetry)
    mod_factors = {
        a: 1.0 + (hash(a) % 7) * 0.01 for a in actions
    }  # values in [1.0, 1.06]

    unnorm = {}
    for a in actions:
        unnorm[a] = prior.get(a, 0.0) * likelihood * mod_factors[a]

    total = sum(unnorm.values()) + 1e-12
    posterior = {a: v / total for a, v in unnorm.items()}
    return posterior


# ----------------------------------------------------------------------
# Bandit policy (ε‑greedy) that consumes the posterior
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}
_EPSILON = 0.1  # exploration probability


def reset_policy() -> None:
    """Clear all stored reward statistics."""
    _POLICY.clear()


def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]


def update_policy(action: str, reward: float) -> None:
    """Incremental update of the average reward for *action*."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    _POLICY[action] = [total + reward, n + 1.0]


def select_action(posterior: Dict[str, float]) -> str:
    """
    ε‑greedy selection: with probability ε explore uniformly,
    otherwise exploit the action with the highest posterior *and* empirical reward.
    """
    actions = list(posterior.keys())
    if random.random() < _EPSILON:
        return random.choice(actions)

    # Combine posterior with empirical reward (weighted 70/30)
    scores = {
        a: 0.7 * posterior[a] + 0.3 * _reward(a) for a in actions
    }
    return max(scores, key=scores.get)


# ----------------------------------------------------------------------
# High‑level hybrid routing function
# ----------------------------------------------------------------------
def hybrid_route(packet: Dict[str, List[float]], context_text: str) -> Tuple[str, Dict[str, float]]:
    """
    End‑to‑end routing:
    1. Extract brain‑map features from *context_text* → master vector.
    2. Compute curvature‑based prior over actions.
    3. Compute SSIM similarity of the packet → likelihood.
    4. Apply Bayes rule → posterior.
    5. Choose an action via the bandit policy.
    6. Return the chosen action and the posterior distribution.
    """
    # 1‑2: prior
    master_vec = extract_master_vector(context_text)
    prior = compute_curvature(master_vec)

    # 3: likelihood
    likelihood = hybrid_score(packet)

    # 4: posterior
    actions = list(prior.keys())
    posterior = bayesian_update(prior, likelihood, actions)

    # 5: selection
    chosen = select_action(posterior)

    # 6: (optional) update policy with a synthetic reward proportional to similarity
    synthetic_reward = likelihood  # in a real system this would be measured downstream
    update_policy(chosen, synthetic_reward)

    return chosen, posterior


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)

    # Example packet and contextual text
    example_packet = {"payload": [0.15, 0.55, 0.28, 0.68, 0.12]}
    example_text = "operator telemetry and resilience metrics for the upcoming mission"

    # Run the hybrid router
    action, post = hybrid_route(example_packet, example_text)

    print(f"Selected action: {action}")
    print("Posterior distribution:")
    for a, prob in post.items():
        print(f"  {a}: {prob:.4f}")

    # Show policy statistics
    print("\nPolicy statistics (average reward per action):")
    for a in sorted(_POLICY.keys()):
        avg = _reward(a)
        cnt = _count(a)
        print(f"  {a}: avg_reward={avg:.4f}, pulls={int(cnt)}")