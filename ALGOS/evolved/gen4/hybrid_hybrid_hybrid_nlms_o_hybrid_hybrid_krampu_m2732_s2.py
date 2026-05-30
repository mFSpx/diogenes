# DARWIN HAMMER — match 2732, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s7.py (gen2)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_hybrid_rlct_g_m989_s0.py (gen3)
# born: 2026-05-29T23:45:10Z

"""Hybrid NLMS‑Bandit‑Pheromone System
===================================

This module fuses the core of **Parent A** (the Normalised Least‑Mean‑Squares
adaptive filter) with the decision‑making and entropy‑based exploration
components of **Parent B** (bandit actions with a pheromone system).

Mathematical bridge
------------------
* The NLMS error `e = target – w·x` is interpreted as a *reward signal* for the
  bandit layer.
* The bandit layer maintains a probability distribution `p_i` over actions.
  Its Shannon entropy `H = – Σ p_i log p_i` quantifies exploration.
* The entropy is fed back to the NLMS step‑size `μ`:
  `μ_eff = μ * (1 + α·H)` where `α` controls how much exploration inflates the
  learning rate.
* Conversely, the bandit propensities are updated using the NLMS error as a
  reward, and the pheromone system provides a decay‑based memory of past
  rewards.

The resulting hybrid algorithm updates both the linear weight vector and the
bandit policy in a single mathematically coupled step.

"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# NLMS core (Parent A)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS adaptation step.

    Returns:
        (new_weights, error) where error = target - y.
    """
    y = nlms_predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    new_weights = weights + (mu * error / power) * x
    return new_weights, error


# ----------------------------------------------------------------------
# Simple zero‑shot span extractor (Parent B simplified)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str


def extract_spans(text: str) -> List[Span]:
    """Each whitespace‑separated token becomes a span with a random score."""
    spans: List[Span] = []
    for match in __import__("re").finditer(r"\S+", text):
        start, end = match.start(), match.end()
        token = match.group()
        score = random.uniform(0.5, 1.0)
        spans.append(
            Span(
                start=start,
                end=end,
                text=token,
                label="TOKEN",
                score=score,
                backend="hybrid_extractor",
            )
        )
    return spans


# ----------------------------------------------------------------------
# Bandit & pheromone system (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


_POLICY: dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])  # [sum_rewards, count]


class PheromoneSystem:
    """Very light pheromone decay model."""

    def __init__(self):
        self.pheromone_signals: dict[str, dict[str, float]] = {}

    def calculate_pheromone_signal(
        self, surface_key: str, signal_kind: str, half_life_seconds: float
    ) -> float:
        """Return decayed signal; if missing, treat as zero."""
        if surface_key not in self.pheromone_signals:
            return 0.0
        if signal_kind not in self.pheromone_signals[surface_key]:
            return 0.0
        # No real time tracking – assume one unit of elapsed time per call.
        elapsed = 1.0
        base = self.pheromone_signals[surface_key][signal_kind]
        return base * math.pow(0.5, elapsed / half_life_seconds)

    def update_pheromone_signal(
        self, surface_key: str, signal_kind: str, signal_value: float
    ) -> None:
        self.pheromone_signals.setdefault(surface_key, {})[signal_kind] = signal_value


def reset_policy() -> None:
    _POLICY.clear()


# ----------------------------------------------------------------------
# Entropy utilities (bridge)
# ----------------------------------------------------------------------
def softmax(propensities: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    max_val = np.max(propensities)
    exp_shifted = np.exp(propensities - max_val)
    return exp_shifted / np.sum(exp_shifted)


def shannon_entropy(prob_dist: np.ndarray) -> float:
    """H = - Σ p log p  (base e). Zero probabilities are ignored."""
    mask = prob_dist > 0
    return -float(np.sum(prob_dist[mask] * np.log(prob_dist[mask])))


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_predict(
    weights: np.ndarray,
    x: np.ndarray,
    actions: List[BanditAction],
) -> Tuple[float, np.ndarray]:
    """
    Combine NLMS linear prediction with a weighted sum of bandit propensities.

    Returns:
        (prediction, action_distribution) where action_distribution is the
        softmax‑normalized propensity vector.
    """
    linear_pred = nlms_predict(weights, x)

    prop_vec = np.array([a.propensity for a in actions], dtype=float)
    if prop_vec.size == 0:
        return linear_pred, np.array([])

    action_dist = softmax(prop_vec)
    # Blend the two sources: simple convex combination with equal weight.
    hybrid_pred = 0.5 * linear_pred + 0.5 * float(np.dot(action_dist, prop_vec))
    return hybrid_pred, action_dist


def hybrid_bandit_update(
    actions: List[BanditAction],
    reward: float,
    pheromone: PheromoneSystem,
    surface_key: str,
    half_life: float = 30.0,
) -> List[BanditAction]:
    """
    Update bandit propensities using the received reward and a decayed pheromone
    signal. The update follows a simple additive rule:

        new_propensity = old_propensity + η * (reward + pheromone)

    where η is a small learning rate.
    """
    eta = 0.1
    pheromone_signal = pheromone.calculate_pheromone_signal(
        surface_key, "reward", half_life_seconds=half_life
    )
    updated: List[BanditAction] = []
    for a in actions:
        delta = eta * (reward + pheromone_signal)
        new_propensity = max(a.propensity + delta, 0.0)  # keep non‑negative
        updated.append(
            BanditAction(
                action_id=a.action_id,
                propensity=new_propensity,
                expected_reward=a.expected_reward,  # unchanged for simplicity
                confidence_bound=a.confidence_bound,
                algorithm=a.algorithm,
            )
        )
    # Store the latest pheromone value (simple moving average)
    pheromone.update_pheromone_signal(surface_key, "reward", reward)
    return updated


def hybrid_nlms_bandit_step(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    actions: List[BanditAction],
    pheromone: PheromoneSystem,
    surface_key: str,
    mu: float = 0.5,
    eps: float = 1e-9,
    entropy_coeff: float = 0.3,
) -> Tuple[np.ndarray, List[BanditAction], float]:
    """
    Perform a single hybrid adaptation step:

    1. Compute NLMS error.
    2. Derive entropy of current action distribution.
    3. Modulate NLMS step‑size with entropy.
    4. Update weights with the entropy‑scaled μ.
    5. Treat the NLMS error (negated) as a reward for the bandit layer.
    6. Update bandit propensities using the pheromone system.

    Returns:
        (new_weights, new_actions, error)
    """
    # ---- 1. NLMS error ----
    _, error = nlms_update(weights, x, target, mu=mu, eps=eps)

    # ---- 2. Entropy of action distribution ----
    prop_vec = np.array([a.propensity for a in actions], dtype=float)
    if prop_vec.size == 0:
        entropy = 0.0
        action_dist = np.array([])
    else:
        action_dist = softmax(prop_vec)
        entropy = shannon_entropy(action_dist)

    # ---- 3. Entropy‑scaled step size ----
    mu_eff = mu * (1.0 + entropy_coeff * entropy)

    # ---- 4. Weight update with scaled μ ----
    new_weights, _ = nlms_update(weights, x, target, mu=mu_eff, eps=eps)

    # ---- 5. Reward for bandit (higher error → lower reward) ----
    reward = -abs(error)  # negative absolute error as scalar reward

    # ---- 6. Bandit & pheromone update ----
    new_actions = hybrid_bandit_update(
        actions, reward, pheromone, surface_key=surface_key
    )

    return new_weights, new_actions, error


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Initialise a small NLMS filter
    dim = 5
    w = np.zeros(dim, dtype=float)
    x = np.random.randn(dim)

    # Target is a noisy linear function of x
    true_w = np.arange(1, dim + 1, dtype=float)
    target_val = float(np.dot(true_w, x) + np.random.randn() * 0.1)

    # Initialise a tiny bandit set
    actions = [
        BanditAction(
            action_id=f"act{i}",
            propensity=random.uniform(0.1, 1.0),
            expected_reward=0.0,
            confidence_bound=0.0,
            algorithm="hybrid",
        )
        for i in range(3)
    ]

    pher = PheromoneSystem()
    surface = "demo_surface"

    # Run a few hybrid steps
    for step in range(10):
        w, actions, err = hybrid_nlms_bandit_step(
            weights=w,
            x=x,
            target=target_val,
            actions=actions,
            pheromone=pher,
            surface_key=surface,
            mu=0.4,
            entropy_coeff=0.2,
        )
        pred, dist = hybrid_predict(w, x, actions)
        print(
            f"Step {step:02d} | Error {err: .4f} | Entropy {shannon_entropy(dist): .4f} | "
            f"Pred {pred: .4f} | Propensities {[a.propensity for a in actions]}"
        )

    # Demonstrate span extraction
    sample_text = "Hybrid NLMS‑Bandit system merges adaptive filtering with exploration."
    spans = extract_spans(sample_text)
    print("\nExtracted spans:")
    for s in spans[:5]:
        print(f"{s.start}-{s.end}: '{s.text}' (score={s.score:.2f})")