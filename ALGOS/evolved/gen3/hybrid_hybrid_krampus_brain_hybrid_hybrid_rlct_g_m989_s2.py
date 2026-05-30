# DARWIN HAMMER — match 989, survivor 2
# gen: 3
# parent_a: hybrid_krampus_brainmap_bandit_router_m129_s0.py (gen1)
# parent_b: hybrid_hybrid_rlct_grokking_hybrid_pheromone_inf_m208_s2.py (gen2)
# born: 2026-05-29T23:32:06Z

"""Hybrid Krampus‑Bandit‑RLCT‑Pheromone System
Parents:
- hybrid_krampus_brainmap_bandit_router_m129_s0.py (feature extraction + contextual bandit)
- hybrid_hybrid_rlct_grokking_hybrid_pheromone_inf_m208_s2.py (RLCT estimation + pheromone infotaxis)

Mathematical Bridge:
Both parents operate on high‑dimensional vectors and employ information‑theoretic quantities.
The bridge is built by treating the feature vector from the Krampus brain‑map as a
context c∈ℝ^d, the pheromone signal ϕ∈ℝ⁺ as an entropy‑like weight, and the Real Log
Canonical Threshold λ̂ (estimated from loss trajectories) as a regularisation exponent.
A hybrid action score S_i is defined as

    S_i = μ_i + β·σ_i + γ·ϕ·(‖c‖₂)^{‑λ̂}

where μ_i is the empirical mean reward, σ_i the confidence bound, β and γ are
tunable scalars. The term ϕ·‖c‖^{‑λ̂} injects the pheromone‑guided entropy
modulated by the RLCT exponent, thus fusing the two parent topologies into a single
decision‑making rule.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Bandit core (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridKrampusBanditRLCT"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # [total_reward, count]

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(updates: list[BanditUpdate]) -> None:
    """Accumulate rewards for each action."""
    for u in updates:
        s = _POLICY[u.action_id]
        s[0] += float(u.reward)
        s[1] += 1.0

def _mean_reward(action_id: str) -> float:
    total, n = _POLICY[action_id]
    return total / n if n else 0.0

def _confidence_bound(action_id: str) -> float:
    """Simple UCB‑like confidence bound."""
    _, n = _POLICY[action_id]
    if n == 0:
        return float('inf')
    return math.sqrt(2 * math.log(max(1, sum(cnt for _, cnt in _POLICY.values()))) / n)

# ----------------------------------------------------------------------
# Feature extraction (Krampus brain‑map, Parent A)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> dict[str, float]:
    """Generate a high‑dimensional random feature vector from text."""
    # In a real system this would be a learned embedding; we use random numbers for demo.
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    keys = [
        'operator_visceral_ratio', 'operator_tech_ratio',
        'operator_legal_osint_ratio', 'operator_ledger_density',
        'operator_recursion_score', 'operator_directive_ratio',
        'operator_target_density', 'psyche_forensic_shield_ratio',
        'psyche_poetic_entropy', 'psyche_dissociative_index',
        'psyche_wrath_velocity', 'resilience_bureaucratic_weaponization_index',
        'resilience_resource_exhaustion_metric', 'resilience_swarm_orchestration_density',
        'resilience_logic_crucifixion_index'
    ]
    return {k: float(rng.random()) for k in keys}

# ----------------------------------------------------------------------
# Pheromone system (Parent B)
# ----------------------------------------------------------------------
class PheromoneSystem:
    def __init__(self):
        self.pheromone_signals: dict[str, dict[str, float]] = {}

    def calculate_pheromone_signal(self, surface_key: str, signal_kind: str,
                                   signal_value: float, half_life_seconds: float) -> float:
        """Return decayed pheromone strength; elapsed_time is mocked as 0 for simplicity."""
        if surface_key not in self.pheromone_signals:
            self.pheromone_signals[surface_key] = {}
        if signal_kind not in self.pheromone_signals[surface_key]:
            self.pheromone_signals[surface_key][signal_kind] = signal_value
        # No real time tracking – assume immediate use.
        elapsed_time = 0.0
        base = self.pheromone_signals[surface_key][signal_kind]
        return base * math.pow(0.5, elapsed_time / half_life_seconds)

    def update_pheromone_signal(self, surface_key: str, signal_kind: str,
                                signal_value: float, half_life_seconds: float) -> None:
        """Set or replace a pheromone signal."""
        if surface_key not in self.pheromone_signals:
            self.pheromone_signals[surface_key] = {}
        self.pheromone_signals[surface_key][signal_kind] = signal_value

# ----------------------------------------------------------------------
# RLCT estimation (Parent B – simplified)
# ----------------------------------------------------------------------
def estimate_rlct_from_losses(train_losses_per_n: list[float],
                              n_values: list[int]) -> float:
    """
    Estimate the Real Log Canonical Threshold λ̂ by fitting a power‑law
    loss ≈ C·n^{‑λ} on log‑log scale using ordinary least squares.
    """
    if len(train_losses_per_n) != len(n_values) or len(train_losses_per_n) < 2:
        raise ValueError("Need at least two (loss, n) pairs.")
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)

    # Guard against non‑positive values.
    if np.any(losses <= 0) or np.any(ns <= 0):
        raise ValueError("Losses and n values must be positive.")

    log_losses = np.log(losses)
    log_ns = np.log(ns)

    # Linear regression: log(loss) = log(C) - λ·log(n)
    A = np.vstack([np.ones_like(log_ns), -log_ns]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_losses, rcond=None)
    _, lambda_est = coeffs
    return max(lambda_est, 0.0)  # RLCT is non‑negative

# ----------------------------------------------------------------------
# Hybrid scoring utilities
# ----------------------------------------------------------------------
def _vector_norm(features: dict[str, float]) -> float:
    """L2 norm of the feature dictionary."""
    vec = np.fromiter(features.values(), dtype=np.float64)
    return float(np.linalg.norm(vec, ord=2))

def compute_entropy_weight(pheromone: float, rlct: float, norm_feat: float) -> float:
    """
    Entropy‑like weight w = ϕ * (‖c‖)^{‑λ̂}
    where ϕ is pheromone strength, λ̂ is RLCT estimate, and ‖c‖ is feature norm.
    """
    if norm_feat == 0:
        return 0.0
    return pheromone * (norm_feat ** (-rlct))

def hybrid_score_action(action_id: str, features: dict[str, float],
                       pheromone: float, rlct: float,
                       beta: float = 1.0, gamma: float = 1.0) -> float:
    """
    Compute hybrid score S_i = μ_i + β·σ_i + γ·w,
    where w = pheromone * (‖features‖)^{‑λ̂}.
    """
    mu = _mean_reward(action_id)
    sigma = _confidence_bound(action_id)
    norm_feat = _vector_norm(features)
    w = compute_entropy_weight(pheromone, rlct, norm_feat)
    return mu + beta * sigma + gamma * w

def hybrid_select_action(text: str, candidate_action_ids: list[str],
                         pheromone_system: PheromoneSystem,
                         half_life_seconds: float = 60.0,
                         beta: float = 1.0, gamma: float = 1.0) -> BanditAction:
    """
    End‑to‑end selection:
    1. Extract features from text.
    2. Obtain a pheromone signal keyed by the text hash.
    3. Estimate RLCT from a synthetic loss trajectory (for demo purposes).
    4. Score each candidate action with the hybrid formula.
    5. Return the best action wrapped in BanditAction.
    """
    # 1. Feature extraction
    features = extract_full_features(text)

    # 2. Pheromone signal (use a deterministic pseudo‑signal for reproducibility)
    surface_key = f"ctx_{abs(hash(text)) % (2**16)}"
    signal_kind = "entropy"
    # Create a pseudo‑signal based on text length
    signal_value = (len(text) % 10) + 1.0
    pheromone = pheromone_system.calculate_pheromone_signal(
        surface_key, signal_kind, signal_value, half_life_seconds
    )

    # 3. RLCT estimate – in a real system this would use training history.
    # Here we fabricate a simple decreasing loss curve.
    n_vals = list(range(10, 110, 10))
    losses = [1.0 / (n ** 0.5) + random.uniform(-0.01, 0.01) for n in n_vals]
    rlct_est = estimate_rlct_from_losses(losses, n_vals)

    # 4. Score actions
    scores = {}
    for aid in candidate_action_ids:
        scores[aid] = hybrid_score_action(
            aid, features, pheromone, rlct_est, beta=beta, gamma=gamma
        )

    # 5. Choose best
    best_id = max(scores, key=scores.get)
    best_score = scores[best_id]
    propensity = math.exp(best_score) / sum(math.exp(v) for v in scores.values())
    expected_reward = _mean_reward(best_id)
    confidence = _confidence_bound(best_id)

    return BanditAction(
        action_id=best_id,
        propensity=propensity,
        expected_reward=expected_reward,
        confidence_bound=confidence
    )

# ----------------------------------------------------------------------
# Demonstration / Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset any prior state
    reset_policy()
    pheromone_system = PheromoneSystem()

    # Define a tiny action set
    actions = ["click", "scroll", "type", "idle"]

    # Simulate a few rounds of interaction
    for round_idx in range(5):
        sample_text = f"User input #{round_idx} with random seed {random.random()}"
        selected = hybrid_select_action(
            sample_text,
            actions,
            pheromone_system,
            half_life_seconds=120.0,
            beta=0.5,
            gamma=0.8
        )
        # Mock reward: give +1 if action matches a simple rule, else 0
        reward = 1.0 if selected.action_id == "click" else 0.0
        update_policy([
            BanditUpdate(
                context_id=f"ctx_{round_idx}",
                action_id=selected.action_id,
                reward=reward,
                propensity=selected.propensity
            )
        ])
        # Update pheromone to reflect the selected action
        pheromone_system.update_pheromone_signal(
            surface_key=f"ctx_{abs(hash(sample_text)) % (2**16)}",
            signal_kind="entropy",
            signal_value=selected.propensity * 10.0,
            half_life_seconds=120.0
        )
        print(f"Round {round_idx}: selected {selected.action_id}, reward {reward}")

    # Final policy snapshot
    print("\nFinal policy statistics:")
    for aid in actions:
        total, cnt = _POLICY[aid]
        print(f"  {aid}: total_reward={total:.2f}, count={cnt:.0f}, mean={_mean_reward(aid):.3f}")