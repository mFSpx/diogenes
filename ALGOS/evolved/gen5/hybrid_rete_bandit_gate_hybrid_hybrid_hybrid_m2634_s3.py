# DARWIN HAMMER — match 2634, survivor 3
# gen: 5
# parent_a: rete_bandit_gate.py (gen0)
# parent_b: hybrid_hybrid_hybrid_hard_t_workshare_allocator_m250_s0.py (gen4)
# born: 2026-05-29T23:43:12Z

"""Hybrid Rete‑Bandit‑Workshare Allocator

This module fuses the core mathematics of **rete_bandit_gate.py** (a deterministic
pruning + multi‑armed bandit / regret routing system) with the **workshare
allocator** logic from *hybrid_hybrid_hard_t_workshare_allocator_m250_s0.py*.

Mathematical bridge
-------------------
* In the bandit side each *group* (e.g. "codex", "groq", …) is an arm *i* with
  an estimated value `Q_i` and a selection probability `p_i` derived from a
  regret‑weighted softmax:
  
  
  w_i = exp( -η * R_i )          # regret‑weighted factor
  p_i = w_i / Σ_j w_j
  

* In the workshare side the system must allocate a total amount of compute
  `U_total` across the same groups, respecting a deterministic target share
  `τ` (default 90 %).  The allocation for arm *i* is a convex combination of
  the bandit‑derived probability and a baseline equal‑share vector `b_i = 1/G`:

  
  a_i = U_total * ( τ * p_i + (1‑τ) * b_i )
  

Thus the bandit probabilities become the *direction* for the workshare
allocation, while the deterministic target enforces a lower bound on fairness.
The three functions below implement the update‑select‑allocate cycle, expose
the underlying matrices, and provide a simple smoke‑test.

Author: OpenAI‑ChatGPT
Date: 2026‑05‑29
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
G = len(GROUPS)

@dataclass
class BanditArm:
    """State of a single bandit arm."""
    name: str
    pulls: int = 0
    reward_sum: float = 0.0
    q_est: float = 0.0   # empirical mean reward
    regret: float = 0.0  # cumulative regret

@dataclass
class WorkshareLane:
    """Resulting allocation for a group."""
    group: str
    llm_units: float
    llm_share_pct: float
    proof_required: bool = False

# ----------------------------------------------------------------------
# Bandit mechanics (softmax with regret weighting)
# ----------------------------------------------------------------------
def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    z = x / max(temperature, 1e-12)
    z = z - np.max(z)
    e_z = np.exp(z)
    return e_z / e_z.sum()

def compute_regret_weighted_strategy(arms: List[BanditArm],
                                    eta: float = 0.5) -> np.ndarray:
    """
    Compute a probability vector over arms using regret‑weighted softmax.

    Parameters
    ----------
    arms: list of BanditArm
        Current arm statistics.
    eta: float
        Regret temperature (higher → more exploration).

    Returns
    -------
    probs: np.ndarray, shape (G,)
        Probabilities that sum to 1.
    """
    regrets = np.array([arm.regret for arm in arms], dtype=float)
    # Convert regrets to positive weights; larger regret → lower weight
    weights = np.exp(-eta * regrets)
    probs = weights / weights.sum()
    return probs

def select_action(probs: np.ndarray) -> int:
    """Sample an arm index according to the given probability vector."""
    return int(np.random.choice(len(probs), p=probs))

def update_bandit(arms: List[BanditArm],
                  chosen_idx: int,
                  reward: float,
                  optimal_reward_est: float) -> None:
    """
    Perform a standard bandit update and accumulate regret.

    Parameters
    ----------
    arms: list of BanditArm
        Mutable arm list.
    chosen_idx: int
        Index of the arm that was pulled.
    reward: float
        Observed reward for the chosen arm.
    optimal_reward_est: float
        Estimate of the best possible reward (used for regret).
    """
    arm = arms[chosen_idx]
    arm.pulls += 1
    arm.reward_sum += reward
    arm.q_est = arm.reward_sum / arm.pulls
    # Regret is the difference between optimal and obtained reward
    arm.regret += max(0.0, optimal_reward_est - reward)

# ----------------------------------------------------------------------
# Workshare allocation that consumes the bandit probabilities
# ----------------------------------------------------------------------
def allocate_workshare(total_units: float,
                       deterministic_target_pct: float = 90.0,
                       probs: np.ndarray = None) -> Dict[str, WorkshareLane]:
    """
    Allocate compute units to each group.

    The allocation blends the bandit probability vector `probs` with an
    equal‑share baseline.  If `probs` is None, a uniform distribution is used.

    Returns a dict mapping group name → WorkshareLane.
    """
    tau = deterministic_target_pct / 100.0
    if probs is None:
        probs = np.full(G, 1.0 / G)

    baseline = np.full(G, 1.0 / G)
    blended = tau * probs + (1.0 - tau) * baseline
    blended = blended / blended.sum()  # re‑normalize for safety

    lanes = {}
    for i, group in enumerate(GROUPS):
        units = total_units * blended[i]
        pct = blended[i] * 100.0
        lanes[group] = WorkshareLane(group=group,
                                     llm_units=units,
                                     llm_share_pct=_pct(pct),
                                     proof_required=False)
    return lanes

def _pct(value: float) -> float:
    """Round a percentage to six decimal places."""
    return round(float(value), 6)

# ----------------------------------------------------------------------
# Hybrid operation: bandit update → probability → allocation
# ----------------------------------------------------------------------
def hybrid_cycle(arms: List[BanditArm],
                 total_units: float,
                 deterministic_target_pct: float = 90.0,
                 eta: float = 0.5,
                 reward_fn = None) -> Tuple[Dict[str, WorkshareLane], np.ndarray]:
    """
    Execute one hybrid iteration:
      1. Compute regret‑weighted probabilities.
      2. Sample an arm, obtain a reward (via `reward_fn`), update bandit.
      3. Allocate workshare based on the *new* probability vector.

    Parameters
    ----------
    arms: list of BanditArm
        Current bandit state (will be mutated).
    total_units: float
        Total compute units to distribute.
    deterministic_target_pct: float
        τ in the allocation formula.
    eta: float
        Regret temperature for the softmax.
    reward_fn: callable(idx) → (reward, optimal_est)
        Function that returns a stochastic reward for the chosen arm and an
        estimate of the optimal possible reward for regret calculation.
        If None, a synthetic Gaussian reward is used.

    Returns
    -------
    allocation: dict[group → WorkshareLane]
    probs: np.ndarray of shape (G,) – the probability vector *after* the update.
    """
    # 1. Compute probabilities from current regrets
    probs = compute_regret_weighted_strategy(arms, eta=eta)

    # 2. Choose an arm and obtain reward
    chosen = select_action(probs)

    if reward_fn is None:
        # Synthetic reward: true mean = sin(idx) + 1, plus Gaussian noise
        true_mean = math.sin(chosen) + 1.0
        reward = random.gauss(true_mean, 0.2)
        optimal_est = max(math.sin(i) + 1.0 for i in range(G))
    else:
        reward, optimal_est = reward_fn(chosen)

    # 3. Update bandit statistics
    update_bandit(arms, chosen, reward, optimal_est)

    # 4. Re‑compute probabilities after the update for allocation
    probs_post = compute_regret_weighted_strategy(arms, eta=eta)

    # 5. Allocate workshare using the post‑update probabilities
    allocation = allocate_workshare(total_units,
                                    deterministic_target_pct,
                                    probs=probs_post)
    return allocation, probs_post

# ----------------------------------------------------------------------
# Helper to initialise a fresh bandit ensemble
# ----------------------------------------------------------------------
def init_bandit_arms() -> List[BanditArm]:
    """Create a list of BanditArm objects, one per group."""
    return [BanditArm(name=g) for g in GROUPS]

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise
    arms = init_bandit_arms()
    TOTAL_UNITS = 1000.0
    TARGET_PCT = 90.0

    # Run a few hybrid cycles
    for step in range(5):
        alloc, probs = hybrid_cycle(arms,
                                    total_units=TOTAL_UNITS,
                                    deterministic_target_pct=TARGET_PCT,
                                    eta=0.7)
        print(f"Step {step+1}")
        print("  Probabilities:", {g: round(p, 4) for g, p in zip(GROUPS, probs)})
        for lane in alloc.values():
            print(f"  {lane.group:12s} → {lane.llm_units:8.2f} units "
                  f"({lane.llm_share_pct:5.2f}%)")
        print("-" * 60)

    # Final bandit state dump
    print("\nFinal bandit arm statistics:")
    for arm in arms:
        print(asdict(arm))