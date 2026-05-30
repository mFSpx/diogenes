# DARWIN HAMMER — match 2070, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_rlct_grokking_hybrid_pheromone_inf_m208_s3.py (gen2)
# parent_b: hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s1.py (gen4)
# born: 2026-05-29T23:40:41Z

"""Hybrid RLCT–Grokking, Dendritic Compartment, Infotaxis & Epistemic Morphology
=============================================================================

Parents:
- **hybrid_hybrid_rlct_grokking_hybrid_pheromone_inf_m208_s3.py** – RLCT
  estimation, Hodgkin‑Huxley neuronal energy and expected entropy from a
  pheromone‑based infotaxis module.
- **hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s1.py** – geometric
  morphology indices (sphericity, flatness, righting‑time) and epistemic
  certainty flags.

Mathematical bridge:
Both parents optimise scalar quantities that stem from information theory:
the infotaxis module supplies an *expected entropy*  H, while the epistemic
flags provide a *certainty weight* C derived from morphology.  The RLCT term
(R) corrects the neuronal free‑energy E.  We fuse them into a single objective

    J = E(V,g_Na,g_K) – λ·R·log log N + μ·H + ν·C

and let the combined information signal (H, C) modulate the ionic conductances
g_Na and g_K.  This creates a true hybrid system where geometry, learning‑
complexity and search‑entropy jointly drive neuronal dynamics.

The module below implements the fused dynamics and provides three public
functions that demonstrate the integrated operation.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List

# ----------------------------------------------------------------------
# Epistemic‑morphology utilities (Parent B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
FLAG_WEIGHTS = np.linspace(0.0, 1.0, len(EPISTEMIC_FLAGS))


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """(length·width·height)^{1/3} / length  – dimensionless."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """(length+width) / (2·height) – dimensionless."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Epistemic‑certainty proxy based on morphology."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def morphology_uncertainty(m: Morphology) -> Tuple[float, str]:
    """
    Returns a numeric certainty weight C∈[0,1] and the corresponding epistemic flag.
    The weight is derived from the sphericity index (scaled to [0,1]) and mapped
    onto the discrete flag set.
    """
    sph = sphericity_index(m.length, m.width, m.height)
    norm = max(0.0, min(1.0, sph))                     # clamp to [0,1]
    idx = int(round(norm * (len(EPISTEMIC_FLAGS) - 1)))
    flag = EPISTEMIC_FLAGS[idx]
    weight = FLAG_WEIGHTS[idx]
    return weight, flag


# ----------------------------------------------------------------------
# RLCT & neuronal energy utilities (Parent A)
# ----------------------------------------------------------------------
def estimate_rlct_from_losses(train_losses_per_n: List[float],
                              n_values: List[int]) -> float:
    """
    Linear regression of log(loss) versus log(log(n)) yields a slope = -RLCT.
    Returns a positive RLCT estimate.
    """
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)

    if np.any(ns <= math.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have the same length")

    y = np.log(np.maximum(losses, 1e-12))
    x = np.log(np.log(ns))

    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return -slope  # RLCT is positive when loss decays with n


def compute_neuronal_energy(V: float, g_Na: float, g_K: float) -> float:
    """
    Simple quadratic surrogate for Hodgkin‑Huxley ionic energy.
    """
    return 0.5 * (g_Na * (V - 50.0) ** 2 + g_K * (V + 77.0) ** 2)


# ----------------------------------------------------------------------
# Infotaxis (pheromone) utilities (Parent A)
# ----------------------------------------------------------------------
def expected_entropy(pheromone_grid: Dict[Tuple[int, int], float]) -> float:
    """
    Computes -Σ p_i log p_i where p_i are normalized pheromone concentrations.
    """
    total = sum(pheromone_grid.values())
    if total == 0.0:
        return 0.0
    probs = np.array(list(pheromone_grid.values())) / total
    return -np.sum(probs * np.log(probs + 1e-12))


# ----------------------------------------------------------------------
# Hybrid dynamics
# ----------------------------------------------------------------------
def update_conductances(state: Dict[str, float],
                        morphology: Morphology,
                        pheromones: Dict[Tuple[int, int], float],
                        rlct: float,
                        entropy: float) -> Dict[str, float]:
    """
    Modulates Na and K conductances using:
    - average pheromone level (search pressure)
    - sphericity index (geometric pressure)
    - RLCT (learning‑complexity pressure)
    - entropy (information‑pressure)
    """
    avg_phero = np.mean(list(pheromones.values())) if pheromones else 0.0
    sph = sphericity_index(morphology.length, morphology.width, morphology.height)

    # Base scaling factor from geometry & pheromones
    geom_factor = 1.0 + 0.2 * sph + 0.1 * avg_phero

    # RLCT and entropy provide fine‑grained multiplicative tweaks
    g_Na_new = state["g_Na"] * geom_factor * (1.0 + 0.05 * rlct)
    g_K_new = state["g_K"] * geom_factor * (1.0 - 0.03 * entropy)

    state_updated = state.copy()
    state_updated["g_Na"] = max(g_Na_new, 0.0)
    state_updated["g_K"] = max(g_K_new, 0.0)
    return state_updated


def hybrid_objective(state: Dict[str, float],
                     morphology: Morphology,
                     pheromones: Dict[Tuple[int, int], float],
                     train_losses: List[float],
                     n_values: List[int],
                     lambda_rlct: float = 0.1,
                     mu_entropy: float = 0.5,
                     nu_uncert: float = 0.3) -> float:
    """
    Computes the unified scalar objective
        J = E – λ·RLCT·log log N + μ·H + ν·C
    where:
        E – neuronal energy,
        RLCT – Real Log Canonical Threshold,
        N – largest sample size,
        H – expected entropy from pheromones,
        C – epistemic certainty weight from morphology.
    """
    E = compute_neuronal_energy(state["V"], state["g_Na"], state["g_K"])
    rlct = estimate_rlct_from_losses(train_losses, n_values)
    N = max(n_values) if n_values else 1
    H = expected_entropy(pheromones)
    C, _ = morphology_uncertainty(morphology)

    J = (E
         - lambda_rlct * rlct * math.log(math.log(N + math.e))
         + mu_entropy * H
         + nu_uncert * C)
    return J


def hybrid_step(state: Dict[str, float],
                morphology: Morphology,
                pheromones: Dict[Tuple[int, int], float],
                train_losses: List[float],
                n_values: List[int]) -> Tuple[Dict[str, float], float]:
    """
    Executes one hybrid iteration:
    1. Estimate RLCT.
    2. Compute entropy.
    3. Update conductances.
    4. Return updated state and the current objective value J.
    """
    rlct = estimate_rlct_from_losses(train_losses, n_values)
    entropy = expected_entropy(pheromones)
    state_upd = update_conductances(state, morphology, pheromones, rlct, entropy)
    J = hybrid_objective(state_upd, morphology, pheromones,
                         train_losses, n_values)
    return state_upd, J


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy neuronal state
    state = {
        "V": -65.0,          # membrane potential (mV)
        "g_Na": 120.0,       # Na conductance (mS/cm^2)
        "g_K": 36.0,         # K conductance (mS/cm^2)
    }

    # Simple morphology
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=0.8)

    # Random pheromone field on a 5×5 lattice
    pheromones = {(i, j): random.random() for i in range(5) for j in range(5)}

    # Synthetic training loss curve
    n_vals = [100, 200, 400, 800, 1600]
    losses = [0.9 / math.sqrt(n) + 0.02 * random.random() for n in n_vals]

    # Run a few hybrid steps
    for step in range(3):
        state, J = hybrid_step(state, morph, pheromones, losses, n_vals)
        print(f"Step {step+1}: J={J:.4f}, g_Na={state['g_Na']:.2f}, g_K={state['g_K']:.2f}")

    print("Hybrid module executed successfully.")