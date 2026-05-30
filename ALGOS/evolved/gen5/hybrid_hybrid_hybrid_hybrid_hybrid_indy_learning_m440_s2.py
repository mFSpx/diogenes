# DARWIN HAMMER — match 440, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s1.py (gen3)
# parent_b: hybrid_indy_learning_vector_hybrid_fold_change_d_m38_s2.py (gen4)
# born: 2026-05-29T23:28:58Z

"""Hybrid Fractional‑LTC‑Bandit Allocation Module
================================================

Parents
-------
* **hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s1.py** – provides a
  Liquid‑Time‑Constant (LTC) recurrent update `τ(t)` and a Caputo fractional
  kernel `w_k(α)` used to weight a deterministic LLM allocation.
* **hybrid_indy_learning_vector_hybrid_fold_change_d_m38_s2.py** – supplies a
  contextual bandit with policy bookkeeping, tokenisation‑based context vectors
  and log‑count statistics.

Mathematical Bridge
-------------------
Both parents operate on a discrete time axis `t = 0,1,…,T`.  
The hybrid treats the LTC state `τ(t)` as a *temporal modulation* of the
bandit’s action propensities, while the Caputo kernel supplies a *fractional
memory* that weights past rewards when estimating the expected return of an
action.

For each step `t` we compute


τ(t)   = LTC( τ(t‑1), I(t) )                         # liquid‑time‑constant update
w_k    = CaputoWeight(k, α)  for k = 0…t           # fractional kernel
γ(t)   = (τ(t) / τ_max) * w_t(α)                    # scalar modulation factor
π_a(t)= propensity_a * γ(t)                        # modulated propensity


The bandit selects the action `a* = argmax_a π_a(t)`.  
After receiving a reward `r`, the policy is updated using the usual
incremental average, but the reward contribution is also filtered through the
same fractional kernel to give the *fractional‑averaged* reward used for future
propensity estimates.

The three core functions below implement this fused dynamics."""

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Gamma function (Lanczos approximation) – from Parent A
# ---------------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function for real z > 0."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, len(_LANCZOS_C)):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

# ---------------------------------------------------------------------------
# Caputo fractional kernel – from Parent A
# ---------------------------------------------------------------------------
def caputo_weights(alpha: float, length: int) -> np.ndarray:
    """Return Caputo kernel w_k(α) for k = 0 … length‑1."""
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    coeff = 1.0 / gamma_lanczos(2 - alpha)
    ks = np.arange(length, dtype=float)
    w = (np.power(ks + 1, 1 - alpha) - np.power(ks, 1 - alpha)) * coeff
    return w

# ---------------------------------------------------------------------------
# Liquid‑Time‑Constant (LTC) recurrent update – from Parent A
# ---------------------------------------------------------------------------
def ltc_update(tau_prev: float, I: float, dt: float = 1.0,
               a: float = 1.0, b: float = 1.0) -> float:
    """
    Simple discrete LTC: τ̇ = -a·τ + b·σ(I)
    where σ is a sigmoid. Returns τ(t+dt).
    """
    sigma = 1.0 / (1.0 + math.exp(-I))
    d_tau = -a * tau_prev + b * sigma
    return tau_prev + dt * d_tau

# ---------------------------------------------------------------------------
# Tokenisation – from Parent B
# ---------------------------------------------------------------------------
_WORD_RE = None  # lazily compiled
def _compile_word_re():
    import re
    global _WORD_RE
    _WORD_RE = re.compile(r"\S+")

def tokenize(text: str) -> List[Dict[str, Any]]:
    """Split text into tokens preserving original string."""
    if _WORD_RE is None:
        _compile_word_re()
    return [{"token": m.group(0)} for m in _WORD_RE.finditer(text)]

# ---------------------------------------------------------------------------
# Bandit policy bookkeeping – from Parent B (simplified)
# ---------------------------------------------------------------------------
_POLICY: Dict[str, Tuple[float, int]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _policy_get(action: str) -> Tuple[float, int]:
    return _POLICY.get(action, (0.0, 0))

def _policy_set(action: str, total: float, count: int) -> None:
    _POLICY[action] = (total, count)

def record_reward(action: str, reward: float) -> None:
    total, cnt = _policy_get(action)
    _policy_set(action, total + reward, cnt + 1)

def expected_reward(action: str) -> float:
    total, cnt = _policy_get(action)
    return total / cnt if cnt else 0.0

# ---------------------------------------------------------------------------
# Core Hybrid Functions
# ---------------------------------------------------------------------------
def compute_ltc_sequence(inputs: List[float],
                         tau0: float = 0.5,
                         dt: float = 1.0) -> List[float]:
    """Run LTC over a list of external inputs I(t)."""
    tau = tau0
    seq = []
    for I in inputs:
        tau = ltc_update(tau, I, dt=dt)
        seq.append(tau)
    return seq

def fractional_allocation(tau_seq: List[float],
                          llm_base: float,
                          alpha: float) -> List[float]:
    """
    Compute the LLM allocation for each time step using the LTC sequence and
    Caputo fractional weights.
    """
    tau_max = max(tau_seq) if tau_seq else 1.0
    w = caputo_weights(alpha, len(tau_seq))
    alloc = [
        llm_base * (tau / tau_max) * wk
        for tau, wk in zip(tau_seq, w)
    ]
    return alloc

def bandit_step(context_id: str,
                actions: List[str],
                modulation: float,
                alpha: float) -> str:
    """
    Choose an action using the bandit propensities modulated by the current
    fractional‑LTC factor `modulation`. The expected reward is also filtered
    through a fractional kernel of order `alpha` to provide a memory‑aware
    estimate.
    """
    # Compute fractional‑weighted expected rewards for all actions
    rewards = np.array([expected_reward(a) for a in actions], dtype=float)
    # Apply exponential smoothing with fractional factor (simple proxy)
    weighted = rewards * (modulation ** alpha)
    # Base propensities (uniform if not yet visited)
    base_prop = np.ones_like(weighted) / len(weighted)
    scores = base_prop * (1.0 + weighted)  # additive boost from reward
    chosen_idx = int(np.argmax(scores))
    chosen_action = actions[chosen_idx]
    # Simulated reward (for demo): random between 0 and 1
    reward = random.random()
    # Update policy with fractional influence
    record_reward(chosen_action, reward * modulation)
    return chosen_action

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulated week: day‑of‑week scaled to [0,1] (Mon=0, Sun=1)
    inputs = [i / 6.0 for i in range(7)]

    # 1️⃣ LTC dynamics
    tau_seq = compute_ltc_sequence(inputs, tau0=0.3)

    # 2️⃣ Fractional LLM allocation (llm_base = 100 units, α = 0.7)
    llm_alloc = fractional_allocation(tau_seq, llm_base=100.0, alpha=0.7)

    # 3️⃣ Bandit actions
    actions = ["search", "summarize", "translate"]
    reset_policy()

    print("Day | τ(t) | LLM alloc | Chosen action | Reward")
    for day, (tau, alloc) in enumerate(zip(tau_seq, llm_alloc)):
        modulation = (tau / max(tau_seq)) * alloc / 100.0  # normalized factor
        chosen = bandit_step(f"day{day}", actions, modulation, alpha=0.7)
        reward = _policy_get(chosen)[0] / max(1, _policy_get(chosen)[1])
        print(f"{day:3d} | {tau:5.3f} | {alloc:9.3f} | {chosen:13s} | {reward:6.3f}")

    # Final policy snapshot
    print("\nFinal policy (action → avg reward):")
    for a in actions:
        print(f"{a:10s} -> {expected_reward(a):.3f}")