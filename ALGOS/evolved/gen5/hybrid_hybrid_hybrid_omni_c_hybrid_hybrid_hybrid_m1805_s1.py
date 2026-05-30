# DARWIN HAMMER — match 1805, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m474_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s0.py (gen4)
# born: 2026-05-29T23:38:52Z

"""Hybrid Algorithm: Fusion of Chaotic Omni-Front Synthesis (Parent A) and Distributed Minimum‑Cost Decision Tree (Parent B)

Mathematical Bridge
-------------------
Both parent algorithms rely on *probabilistic decision making*:

* Parent A evaluates the quality of a prediction with a similarity metric (SSIM‑like) that can be interpreted as a **reward** for a bandit router.
* Parent B decides whether to accept a new hypothesis (leader, split, or action) using Hoeffding bounds and temperature‑scaled acceptance probabilities.

The hybrid therefore maps the similarity reward `ρ` from Parent A onto the energy‑based acceptance probability of Parent B:


ΔE = -log(ρ + ε)                     # energy derived from similarity (higher ρ → lower ΔE)
p_accept = exp(‑ΔE / T)             # temperature‑scaled acceptance (Parent B)


The Hoeffding bound is then used as a statistical confidence test on the observed reward gap,
guiding the ternary bandit router to either keep the current route or split the decision node.
All core equations from both parents are retained and combined in a single workflow.

"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import Tuple, List

__all__ = [
    "encode_vector",
    "ssim_reward",
    "bandit_acceptance",
    "should_split",
    "hybrid_algorithm",
]

# ----------------------------------------------------------------------
# Parent A – encoding & similarity (SSIM‑like)
# ----------------------------------------------------------------------
def encode_vector(x: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit L2‑norm (fixed encoder from Parent A)."""
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


def ssim_reward(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute a lightweight SSIM‑like reward between two 1‑D signals.
    The formulation mirrors the classic SSIM numerator/denominator structure
    and yields a value in [0, 1] where 1 means perfect similarity.
    """
    C1 = 1e-4
    C2 = 9e-4

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    return float(numerator / denominator) if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Parent B – probabilistic decision tools
# ----------------------------------------------------------------------
def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Parent B: probability of accepting a move with energy change ΔE at temperature T."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Exponential cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Statistical bound used in Parent B."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def should_split(best_gain: float, second_best_gain: float,
                 r: float, delta: float, n: int,
                 tie_threshold: float = 0.05) -> bool:
    """Decision rule for node splitting (Parent B)."""
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    return gap > eps or eps < tie_threshold


# ----------------------------------------------------------------------
# Hybrid Core – mapping similarity → bandit energy & split logic
# ----------------------------------------------------------------------
def bandit_acceptance(similarity: float, step: int,
                      t0: float = 1.0, alpha: float = 0.95,
                      eps: float = 1e-12) -> Tuple[bool, float]:
    """
    Convert a similarity reward into an energy change ΔE and evaluate acceptance.

    Returns (accepted, current_temperature).
    """
    # Transform similarity (∈[0,1]) into a pseudo‑energy; higher similarity → lower energy.
    delta_e = -math.log(max(similarity, eps))
    temperature = cooling_temperature(step, t0, alpha)
    prob = acceptance_probability(delta_e, temperature)
    accept = random.random() < prob
    return accept, temperature


def hybrid_decision(gain_vector: List[float], r: float, delta: float,
                    n_observations: int) -> bool:
    """
    Apply the Hoeffding‑based split test on a list of gains.
    The first element is treated as the best gain, the second as the runner‑up.
    """
    if len(gain_vector) < 2:
        raise ValueError("need at least two gain values")
    best, second = gain_vector[0], gain_vector[1]
    return should_split(best, second, r, delta, n_observations)


def hybrid_algorithm(input_signal: np.ndarray,
                     target_signal: np.ndarray,
                     max_steps: int = 10) -> dict:
    """
    End‑to‑end hybrid routine:

    1. Encode the input (Parent A).
    2. Generate a mock prediction (here we add small Gaussian noise).
    3. Evaluate similarity → reward.
    4. Map reward to bandit acceptance probability using temperature annealing.
    5. If accepted, update a fictitious gain list and decide whether to split a node
       using the Hoeffding bound (Parent B).

    Returns a dictionary with the evolution of key quantities.
    """
    history = {
        "step": [],
        "similarity": [],
        "accepted": [],
        "temperature": [],
        "split_decision": [],
    }

    encoded = encode_vector(input_signal)

    for step in range(max_steps):
        # 2. Mock prediction: perturb encoded vector
        noise = np.random.normal(scale=0.05, size=encoded.shape)
        prediction = encoded + noise

        # 3. Similarity reward
        sim = ssim_reward(prediction, target_signal)

        # 4. Bandit acceptance
        accepted, temp = bandit_acceptance(sim, step)

        # 5. Split decision (only when accepted)
        if accepted:
            # fabricate gains: current similarity as best gain,
            # a slightly degraded version as second best.
            best_gain = sim
            second_gain = max(0.0, sim - random.uniform(0.0, 0.1))
            split = hybrid_decision([best_gain, second_gain],
                                    r=1.0, delta=0.05, n_observations=step + 1)
        else:
            split = False

        # Record
        history["step"].append(step)
        history["similarity"].append(sim)
        history["accepted"].append(accepted)
        history["temperature"].append(temp)
        history["split_decision"].append(split)

    return history


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Dummy signals (1‑D arrays)
    input_sig = np.linspace(0, 1, 100)
    target_sig = np.sin(2 * np.pi * input_sig) * 0.5 + 0.5  # scaled sine wave in [0,1]

    result = hybrid_algorithm(input_sig, target_sig, max_steps=15)

    # Simple sanity check: print final step summary
    final_step = result["step"][-1]
    print(f"Step {final_step}: similarity={result['similarity'][-1]:.4f}, "
          f"accepted={result['accepted'][-1]}, "
          f"temperature={result['temperature'][-1]:.4f}, "
          f"split={result['split_decision'][-1]}")
    sys.exit(0)