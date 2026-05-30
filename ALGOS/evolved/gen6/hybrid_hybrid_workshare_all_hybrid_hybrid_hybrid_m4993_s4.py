# DARWIN HAMMER — match 4993, survivor 4
# gen: 6
# parent_a: hybrid_workshare_allocator_hybrid_hybrid_hybrid_m1490_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1325_s1.py (gen5)
# born: 2026-05-29T23:59:10Z

"""Hybrid Workshare‑Bandit‑Bayesian Algorithm.

Parents:
- hybrid_workshare_allocator_hybrid_hybrid_m1490_s2.py (deterministic workshare allocation + exponential store equation)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1325_s1.py (feature extraction, Bayesian posterior, ternary routing, epistemic flags)

Mathematical bridge:
Both parents manipulate *probability‑like* quantities that sum to one.
Parent A scales a deterministic “propensity” (the probability of choosing an action) with a
store value S(t)=Δₘₐₓ·exp(−α·t/tₘₐₓ)·U, where U are the work‑share units.
Parent B produces a feature‑derived prior vector **p**∈Δⁿ (simplex) and updates it with a Bayesian
posterior **π** using observed rewards.
The hybrid algorithm treats **p** as the propensity for the store equation,
i.e. S·π becomes the *effective allocation* for each action.  This unifies the
exponential decay schedule with the Bayesian update and the deterministic
workshare allocator, while the ternary routing maps the posterior magnitude to
epistemic certainty flags (LOW, MEDIUM, HIGH)."""

import math
import random
import sys
import pathlib
import hashlib
from typing import Dict, List, Tuple, Any
import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def store_equation(t: int, t_max: int, delta_max: float = 1.0,
                   alpha: float = 3.0, units: float = 100.0) -> float:
    """Exponential decay schedule for the store value."""
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0 or units <= 0:
        raise ValueError("invalid store schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max) * units


def workshare_modulate(ratio: float, deterministic_target_pct: float = 90.0) -> float:
    """Scale a propensity by the deterministic target percentage."""
    return ratio * (deterministic_target_pct / 100.0)


def allocate_workshare(total_units: float,
                       deterministic_target_pct: float = 90.0,
                       n_actions: int = 1) -> np.ndarray:
    """
    Simple deterministic allocator: split ``total_units`` proportionally
    according to a target percentage and return an allocation vector of length
    ``n_actions`` that sums to ``total_units``.
    """
    target = total_units * (deterministic_target_pct / 100.0)
    # Uniform split for the placeholder implementation
    base = target / n_actions
    remainder = total_units - base * n_actions
    alloc = np.full(n_actions, base, dtype=float)
    # Distribute any remainder to the first action (deterministic tie‑break)
    if remainder > 0:
        alloc[0] += remainder
    return alloc

# ----------------------------------------------------------------------
# Parent‑B building blocks (feature extraction & Bayesian update)
# ----------------------------------------------------------------------
_EPISTEMIC_FLAGS = ("LOW", "MEDIUM", "HIGH")


def _deterministic_hash(text: str) -> int:
    """Stable 64‑bit hash using SHA‑256."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Produce a reproducible pseudo‑random feature vector from *text*.
    The same input always yields the same output across runs.
    """
    seed = _deterministic_hash(text) % (2 ** 32)
    rnd = random.Random(seed)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "psyche_forensic_shield_ratio",
        "network_latency_ratio",
        "resource_utilisation_ratio",
    ]
    return {k: rnd.random() for k in keys}


def features_to_distribution(features: Dict[str, float]) -> np.ndarray:
    """Convert a feature dict to a probability simplex via L1 normalisation."""
    vec = np.fromiter(features.values(), dtype=float)
    if vec.sum() == 0:
        return np.full_like(vec, 1.0 / vec.size)
    return vec / vec.sum()


# Simple Beta‑Bernoulli Bayesian model for binary rewards
class BayesianBeta:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, reward: float) -> None:
        """Assume reward ∈ {0,1}."""
        if reward not in (0, 1):
            raise ValueError("reward must be 0 or 1 for Beta update")
        self.alpha += reward
        self.beta += 1 - reward


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_compute_allocation(context_id: str,
                              textual_context: str,
                              total_units: float,
                              t: int,
                              t_max: int = 100,
                              deterministic_target_pct: float = 90.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    1. Extract a deterministic feature distribution **p** from the textual context.
    2. Compute a store value S(t) using the exponential schedule.
    3. Modulate **p** with S(t) to obtain an *effective allocation* vector.
    4. Apply deterministic work‑share allocation to respect the target percentage.

    Returns:
        allocation   – units assigned to each action (size = len(p))
        posterior    – Bayesian posterior means for each action (initialized to prior)
    """
    # Step‑1: feature → prior probability simplex
    feats = extract_full_features(textual_context)
    prior = features_to_distribution(feats)

    # Step‑2: exponential store value
    store_val = store_equation(t, t_max, units=total_units)

    # Step‑3: modulation (propensity scaling)
    modulated = prior * workshare_modulate(store_val / total_units, deterministic_target_pct)

    # Normalise to keep the vector on the simplex (optional but numerically stable)
    if modulated.sum() == 0:
        modulated = np.full_like(modulated, 1.0 / modulated.size)
    else:
        modulated = modulated / modulated.sum() * total_units

    # Step‑4: deterministic allocation respecting the target percentage
    allocation = allocate_workshare(total_units=total_units,
                                    deterministic_target_pct=deterministic_target_pct,
                                    n_actions=modulated.size)

    # Use the modulated vector as the *desired* allocation; blend with deterministic split
    final_alloc = 0.6 * allocation + 0.4 * modulated  # convex blend

    # Initialise a flat Bayesian posterior (same for every action)
    posterior = np.full(modulated.shape, BayesianBeta().posterior_mean())

    return final_alloc, posterior


def hybrid_update(context_id: str,
                  action_index: int,
                  reward: float,
                  allocation: np.ndarray,
                  posterior: np.ndarray,
                  total_units: float,
                  deterministic_target_pct: float = 90.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update both the store‑based allocation and the Bayesian posterior for a
    chosen action.

    The allocation vector is re‑scaled by the store equation (reflecting decay)
    and by the updated posterior mean (reflecting learning from the reward).

    Returns updated (allocation, posterior) vectors.
    """
    # Update Bayesian model for the selected action
    beta = BayesianBeta(alpha=posterior[action_index] * 10,
                        beta=(1 - posterior[action_index]) * 10)
    beta.update(reward)
    posterior[action_index] = beta.posterior_mean()

    # Re‑compute store value at the next time‑step (t+1)
    # For simplicity we treat the current total units as the time index
    t = int(allocation.sum())  # rough proxy for elapsed work
    store_val = store_equation(t, 100, units=total_units)

    # Modulate each allocation entry by its posterior mean
    modulation = posterior * workshare_modulate(store_val / total_units,
                                                deterministic_target_pct)
    new_alloc = allocation * modulation
    if new_alloc.sum() == 0:
        new_alloc = allocate_workshare(total_units, deterministic_target_pct,
                                       n_actions=allocation.size)
    else:
        new_alloc = new_alloc / new_alloc.sum() * total_units
    return new_alloc, posterior


def ternary_epistemic_flag(posterior_mean: float,
                           low_thr: float = 0.33,
                           high_thr: float = 0.66) -> str:
    """
    Map a posterior probability to an epistemic certainty flag.
    Returns one of "LOW", "MEDIUM", "HIGH".
    """
    if posterior_mean < low_thr:
        return "LOW"
    if posterior_mean > high_thr:
        return "HIGH"
    return "MEDIUM"


def min_cost_route(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Greedy minimum‑cost assignment (O(n²)).  Returns a list of (row, col) pairs.
    Not optimal like the Hungarian algorithm but satisfies the requirement
    without external dependencies.
    """
    rows, cols = cost_matrix.shape
    assigned_rows = set()
    assigned_cols = set()
    assignments: List[Tuple[int, int]] = []
    while len(assignments) < min(rows, cols):
        # Find the smallest unassigned cost
        min_val = math.inf
        min_pos = (-1, -1)
        for i in range(rows):
            if i in assigned_rows:
                continue
            for j in range(cols):
                if j in assigned_cols:
                    continue
                if cost_matrix[i, j] < min_val:
                    min_val = cost_matrix[i, j]
                    min_pos = (i, j)
        if min_pos == (-1, -1):
            break
        i, j = min_pos
        assignments.append((i, j))
        assigned_rows.add(i)
        assigned_cols.add(j)
    return assignments

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ctx = "example_context"
    text = "The system must allocate resources based on observed latency and load."
    total_units = 120.0
    t = 5

    # Initial hybrid allocation
    alloc, post = hybrid_compute_allocation(ctx, text, total_units, t)
    print("Initial allocation:", alloc)
    print("Initial posterior means:", post)

    # Simulate a reward for action 0
    reward = 1  # binary reward
    alloc, post = hybrid_update(ctx, action_index=0, reward=reward,
                               allocation=alloc, posterior=post,
                               total_units=total_units)
    print("\nAfter update allocation:", alloc)
    print("After update posterior means:", post)

    # Epistemic flags
    flags = [ternary_epistemic_flag(p) for p in post]
    print("\nEpistemic flags per action:", flags)

    # Minimum‑cost routing demo
    cost_mat = np.array([[9, 2, 7],
                         [6, 4, 3],
                         [5, 8, 1]], dtype=float)
    route = min_cost_route(cost_mat)
    print("\nGreedy min‑cost route (row→col):", route)