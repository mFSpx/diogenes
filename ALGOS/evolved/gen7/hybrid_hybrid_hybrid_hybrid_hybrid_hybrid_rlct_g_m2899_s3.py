# DARWIN HAMMER — match 2899, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_cockpit_metri_m1403_s2.py (gen6)
# parent_b: hybrid_hybrid_rlct_grokking_hybrid_pheromone_inf_m208_s3.py (gen2)
# born: 2026-05-29T23:46:33Z

"""Hybrid Algorithm integrating epistemic certainty pruning (Parent A) with RLCT‑infotaxis neuronal dynamics (Parent B).

Mathematical Bridge
-------------------
- The **certainty confidence** (in basis points) from Parent A scales the influence of the
  pheromone field on neuronal conductances (`g_Na`, `g_K`).  Higher confidence amplifies
  the modulation, providing a direct coupling between epistemic certainty and the
  biophysical model.
- The **pruning probability** derived from certainty flags and an honesty metric determines
  a *sample‑size reduction factor* that is applied to the RLCT term in the hybrid objective.
- The hybrid objective combines neuronal ionic energy, the RLCT correction, and the
  expected entropy from the infotaxis pheromone distribution:

    J = E_neuron(V,m,h,n; g_Na, g_K) – λ·RLCT·loglog(N·(1‑p_prune)) + μ·H_expected

  where `p_prune` is the pruning probability, `N` the original sample size, and
  `H_expected` the entropy of the pheromone field.

The module implements three public functions that showcase this integrated dynamics.
"""

import math
import random
import sys
import pathlib
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, Union

# ----------------------------------------------------------------------
# Parent A – Epistemic certainty & pruning utilities
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # basis points, 0..10000
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")
        if not self.generated_at:
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

    def as_dict(self) -> Dict[str, Union[str, int, Tuple[str, ...]]]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }

def hybrid_pruning_probability(
    flag: CertaintyFlag,
    anti_slop_ratio: float,
    honesty_score: float,
) -> float:
    """
    Compute a pruning probability `p ∈ [0,1]` that blends epistemic certainty,
    an anti‑slop ratio (tuning how tolerant we are to uncertainty), and an
    external honesty metric (e.g., from a cockpit honesty score).

    The formula is:
        p = (1 - c) * (1 - anti_slop_ratio) * (1 - honesty_score)

    where `c` is the normalized confidence (basis points → [0,1]).
    """
    if not 0.0 <= anti_slop_ratio <= 1.0:
        raise ValueError("anti_slop_ratio must be in [0,1]")
    if not 0.0 <= honesty_score <= 1.0:
        raise ValueError("honesty_score must be in [0,1]")

    confidence = flag.confidence_bps / 10_000.0
    p = (1.0 - confidence) * (1.0 - anti_slop_ratio) * (1.0 - honesty_score)
    return min(max(p, 0.0), 1.0)

# ----------------------------------------------------------------------
# Parent B – RLCT estimation, pheromone infotaxis, neuronal energy
# ----------------------------------------------------------------------
def estimate_rlct_from_losses(train_losses_per_n: Sequence[float],
                              n_values: Sequence[float]) -> float:
    """
    Simple linear fit of log(loss) vs log(log(n)) to obtain an RLCT estimate.
    """
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)

    if np.any(ns <= math.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have the same length")

    x = np.log(np.log(ns))
    y = np.log(np.maximum(losses, 1e-12))

    # Least‑squares slope = -RLCT
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    rlct = -slope
    return max(rlct, 0.0)

def pheromone_field(grid_shape: Tuple[int, int],
                    sources: Sequence[Tuple[int, int]],
                    decay: float = 0.9) -> np.ndarray:
    """
    Initialise a pheromone concentration grid. Each source deposits 1.0 unit;
    diffusion is approximated by iteratively applying a decay‑weighted average.
    """
    field = np.zeros(grid_shape, dtype=np.float64)
    for i, j in sources:
        if 0 <= i < grid_shape[0] and 0 <= j < grid_shape[1]:
            field[i, j] = 1.0

    # Simple diffusion: 5‑point stencil, repeated few times
    for _ in range(10):
        rolled = (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
        )
        field = decay * 0.25 * rolled + (1 - decay) * field
    return field

def expected_entropy_from_pheromone(field: np.ndarray) -> float:
    """
    Compute Shannon entropy of the normalized pheromone distribution.
    """
    total = field.sum()
    if total == 0:
        return 0.0
    prob = field.ravel() / total
    prob = prob[prob > 0]
    return -float(np.sum(prob * np.log(prob)))

def modulate_conductances(g_Na_base: float,
                          g_K_base: float,
                          pheromone: np.ndarray,
                          confidence_factor: float) -> Tuple[float, float]:
    """
    Scale base conductances by a factor derived from the pheromone field
    and the epistemic confidence.

    The modulation factor is:
        m = 1 + confidence_factor * (mean(pheromone) - 0.5)

    where `confidence_factor` ∈ [0,1] (confidence_bps / 10_000).
    """
    mean_pher = float(np.mean(pheromone))
    mod_factor = 1.0 + confidence_factor * (mean_pher - 0.5)
    return g_Na_base * mod_factor, g_K_base * mod_factor

def neuronal_energy(V: float, m: float, h: float, n: float,
                    g_Na: float, g_K: float,
                    E_Na: float = 50.0, E_K: float = -77.0,
                    g_L: float = 0.003, E_L: float = -54.387) -> float:
    """
    Simplified Hodgkin‑Huxley ionic energy (quadratic form).
    """
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n ** 4) * (V - E_K)
    I_L = g_L * (V - E_L)
    # Energy as absolute power integrated over 1 ms (arbitrary scaling)
    return abs(I_Na + I_K + I_L) * 1e-3

def hybrid_objective(V: float, m: float, h: float, n: float,
                     g_Na: float, g_K: float,
                     rlct: float, sample_size: int,
                     H_expected: float,
                     pruning_prob: float,
                     lam: float = 1.0,
                     mu: float = 1.0) -> float:
    """
    Combined objective:

        J = E_neuron - λ·RLCT·loglog(N·(1‑p_prune)) + μ·H_expected

    The term `loglog` uses natural logarithm twice; a small epsilon guards
    against domain errors.
    """
    eps = 1e-12
    effective_N = max(sample_size * (1.0 - pruning_prob), 2.0)  # at least 2
    loglog = math.log(math.log(effective_N + eps) + eps)

    E = neuronal_energy(V, m, h, n, g_Na, g_K)
    J = E - lam * rlct * loglog + mu * H_expected
    return J

# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def hybrid_step(flag: CertaintyFlag,
                anti_slop_ratio: float,
                honesty_score: float,
                train_losses: Sequence[float],
                ns: Sequence[float],
                grid_shape: Tuple[int, int],
                source_positions: Sequence[Tuple[int, int]],
                base_conductances: Tuple[float, float],
                neuron_state: Tuple[float, float, float, float],
                sample_size: int) -> Dict[str, Union[float, Tuple[float, float]]]:
    """
    Perform one hybrid iteration:
      1. Compute pruning probability.
      2. Estimate RLCT from loss trajectory.
      3. Generate pheromone field and compute expected entropy.
      4. Modulate conductances using confidence and pheromone.
      5. Evaluate hybrid objective J.
    Returns a dict with intermediate values for inspection.
    """
    # 1. Pruning
    p_prune = hybrid_pruning_probability(flag, anti_slop_ratio, honesty_score)

    # 2. RLCT
    rlct = estimate_rlct_from_losses(train_losses, ns)

    # 3. Pheromone & entropy
    pher = pheromone_field(grid_shape, source_positions)
    H_exp = expected_entropy_from_pheromone(pher)

    # 4. Conductance modulation
    confidence_factor = flag.confidence_bps / 10_000.0
    g_Na_mod, g_K_mod = modulate_conductances(
        base_conductances[0], base_conductances[1], pher, confidence_factor
    )

    # 5. Objective
    V, m, h, n = neuron_state
    J = hybrid_objective(V, m, h, n,
                         g_Na_mod, g_K_mod,
                         rlct, sample_size,
                         H_exp, p_prune)

    return {
        "pruning_prob": p_prune,
        "rlct": rlct,
        "entropy": H_exp,
        "g_Na": g_Na_mod,
        "g_K": g_K_mod,
        "objective_J": J,
    }

def run_hybrid_simulation(steps: int = 5) -> None:
    """
    Simple simulation loop exercising the hybrid dynamics.
    """
    # Fixed mock data
    flag = CertaintyFlag(
        label="FACT",
        confidence_bps=8500,
        authority_class="A",
        rationale="demonstration",
    )
    anti_slop = 0.2
    honesty = 0.9
    base_g_Na, base_g_K = 120.0, 36.0
    neuron_state = ( -65.0, 0.05, 0.6, 0.32 )  # V,m,h,n typical HH rest
    sample_N = 1024

    # synthetic loss curve (decreasing)
    ns = np.arange(100, 1100, 200)
    train_losses = 0.5 / np.sqrt(ns) + 0.05 * np.random.rand(len(ns))

    for step in range(steps):
        result = hybrid_step(
            flag=flag,
            anti_slop_ratio=anti_slop,
            honesty_score=honesty,
            train_losses=train_losses,
            ns=ns,
            grid_shape=(20, 20),
            source_positions=[(random.randint(0,19), random.randint(0,19)) for _ in range(3)],
            base_conductances=(base_g_Na, base_g_K),
            neuron_state=neuron_state,
            sample_size=sample_N,
        )
        # Update mock neuron state slightly to keep dynamics alive
        V, m, h, n = neuron_state
        neuron_state = (V + random.uniform(-0.5, 0.5),
                        min(max(m + random.uniform(-0.01, 0.01), 0.0), 1.0),
                        min(max(h + random.uniform(-0.01, 0.01), 0.0), 1.0),
                        min(max(n + random.uniform(-0.01, 0.01), 0.0), 1.0))
        print(f"Step {step+1}: J={result['objective_J']:.6f}, p_prune={result['pruning_prob']:.3f}")

if __name__ == "__main__":
    run_hybrid_simulation()