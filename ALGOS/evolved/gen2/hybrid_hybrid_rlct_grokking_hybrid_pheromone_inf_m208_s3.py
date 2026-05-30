# DARWIN HAMMER — match 208, survivor 3
# gen: 2
# parent_a: hybrid_rlct_grokking_dendritic_compartmen_m61_s0.py (gen1)
# parent_b: hybrid_pheromone_infotaxis_m3_s2.py (gen1)
# born: 2026-05-29T23:27:35Z

"""Hybrid RLCT–Grokking, Dendritic Compartment, and Infotaxis Model
=================================================================

Parent algorithms:
- **hybrid_rlct_grokking_dendritic_compartmen_m61_s0.py** – estimates a Real Log
  Canonical Threshold (RLCT) from training losses and uses it to adjust the
  energy of a Hodgkin‑Huxley‑style dendritic compartment (variables V, m, h,
  n).
- **hybrid_pheromone_infotaxis_m3_s2.py** – maintains pheromone signals on
  discrete surfaces and selects actions by minimizing the expected entropy
  (Infotaxis).

**Mathematical bridge** – Both parents optimise a scalar quantity:
energy (neuron) and information‑theoretic entropy (search).  We fuse them by
letting the pheromone‑derived expected entropy act as a *modulatory factor* on
the neuronal conductances, while the RLCT term continues to correct the raw
energy.  The combined objective is


J = E_neuron(V,m,h,n; g_Na, g_K) – λ·RLCT·loglog(N) + μ·H_expected


where `E_neuron` is the Hodgkin‑Huxley ionic energy, `H_expected` is the
expected entropy from the infotaxis module, and `λ, μ` are scalar weights.
The pheromone signal influences `g_Na` and `g_K`, thereby coupling the two
systems at the level of the governing equations.

The module below implements this hybrid system with three public functions
demonstrating the integrated dynamics.
"""

import math
import random
import sys
import pathlib
import numpy as np
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Parent A – RLCT estimation and neuronal energy
# ----------------------------------------------------------------------
def estimate_rlct_from_losses(train_losses_per_n, n_values):
    """Estimate the Real Log Canonical Threshold (RLCT) from a series of
    training losses `train_losses_per_n` observed at sample sizes `n_values`.

    Returns a float RLCT estimate.
    """
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)
    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if len(losses) != len(ns):
        raise ValueError("train_losses_per_n and n_values must have the same length")
    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)

def sodium_current(V, m, h, g_Na=120.0, E_Na=50.0):
    """Hodgkin‑Huxley sodium current."""
    return g_Na * (m ** 3) * h * (V - E_Na)

def potassium_current(V, n, g_K=36.0, E_K=-77.0):
    """Hodgkin‑Huxley potassium current."""
    return g_K * (n ** 4) * (V - E_K)

def neuronal_energy(V, m, h, n, g_Na=120.0, g_K=36.0):
    """Total ionic energy for a single compartment."""
    Na = sodium_current(V, m, h, g_Na)
    K  = potassium_current(V, n, g_K=g_K)
    return Na + K

# ----------------------------------------------------------------------
# Parent B – Pheromone system and entropy (Infotaxis)
# ----------------------------------------------------------------------
class PheromoneSystem:
    """Tracks pheromone signals per surface and kind, decaying them with a
    half‑life.  The stored value also records the timestamp of the last update.
    """
    def __init__(self):
        # surface_key -> signal_kind -> (value, timestamp)
        self.pheromone_signals = {}

    def _now(self):
        return datetime.now(timezone.utc)

    def calculate_pheromone_signal(self, surface_key, signal_kind,
                                   half_life_seconds):
        """Return the decayed pheromone strength for the given key/kind."""
        if surface_key not in self.pheromone_signals:
            return 0.0
        if signal_kind not in self.pheromone_signals[surface_key]:
            return 0.0
        value, timestamp = self.pheromone_signals[surface_key][signal_kind]
        elapsed = (self._now() - timestamp).total_seconds()
        decay_factor = math.pow(0.5, elapsed / half_life_seconds) if half_life_seconds > 0 else 1.0
        return value * decay_factor

    def update_pheromone_signal(self, surface_key, signal_kind,
                                signal_value, half_life_seconds):
        """Insert or replace a pheromone signal; the half‑life is stored for
        later decay calculations.
        """
        if surface_key not in self.pheromone_signals:
            self.pheromone_signals[surface_key] = {}
        self.pheromone_signals[surface_key][signal_kind] = (signal_value,
                                                            self._now())

def calculate_entropy(probabilities, eps=1e-12):
    """Shannon entropy of a discrete distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    probs = [(p / total) for p in probabilities if p > 0]
    return -sum(p * math.log(max(p, eps)) for p in probs)

def expected_entropy(p_hit, hit_state, miss_state):
    """Expected entropy after a binary observation with hit probability
    `p_hit`.  `hit_state` and `miss_state` are raw probability vectors.
    """
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * calculate_entropy(hit_state) + (1.0 - p_hit) * calculate_entropy(miss_state)

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_conductance_modulation(base_g_Na, base_g_K, pheromone_strength,
                                 alpha=0.3):
    """Modulate sodium and potassium conductances using a pheromone signal.
    Positive pheromone boosts Na, suppresses K; negative does the opposite.
    The modulation is bounded to keep conductances physically plausible.
    """
    # Clamp pheromone_strength to [-1, 1] for stability
    s = max(min(pheromone_strength, 1.0), -1.0)
    g_Na = base_g_Na * (1.0 + alpha * s)
    g_K  = base_g_K * (1.0 - alpha * s)
    # Ensure conductances stay non‑negative
    return max(g_Na, 0.0), max(g_K, 0.0)

def hybrid_rlct_pheromone_energy(V, m, h, n,
                                 train_losses_per_n, n_values,
                                 pheromone_system, surface_key, signal_kind,
                                 half_life_seconds,
                                 lambda_rlct=1.0, mu_entropy=0.5,
                                 base_g_Na=120.0, base_g_K=36.0):
    """
    Compute the hybrid objective:
        J = E_neuron(modulated) – λ·RLCT·loglog(N_max) + μ·H_expected

    The pheromone signal modulates the ionic conductances, while the expected
    entropy is obtained from a synthetic hit/miss model (here we reuse the
    pheromone strength as a proxy for hit probability).
    """
    # 1) RLCT term
    rlct = estimate_rlct_from_losses(train_losses_per_n, n_values)
    loglog_term = math.log(math.log(n_values[-1]))

    # 2) Pheromone‑driven conductance modulation
    pheromone_strength = pheromone_system.calculate_pheromone_signal(
        surface_key, signal_kind, half_life_seconds)
    g_Na_mod, g_K_mod = hybrid_conductance_modulation(
        base_g_Na, base_g_K, pheromone_strength)

    # 3) Neuronal energy with modulated conductances
    energy = neuronal_energy(V, m, h, n, g_Na=g_Na_mod, g_K=g_K_mod)

    # 4) Expected entropy – we treat the pheromone strength (scaled to [0,1])
    #    as the hit probability for a binary observation.
    p_hit = (pheromone_strength + 1.0) / 2.0  # map [-1,1] → [0,1]
    # Simple synthetic states: hit concentrates probability on state 0,
    # miss spreads it uniformly over 5 states.
    hit_state  = [0.9, 0.025, 0.025, 0.025, 0.025]
    miss_state = [0.2, 0.2, 0.2, 0.2, 0.2]
    H_exp = expected_entropy(p_hit, hit_state, miss_state)

    # 5) Combine
    J = energy - lambda_rlct * rlct * loglog_term + mu_entropy * H_exp
    return J

def hybrid_simulation_step(state, rlct_data, pheromone_system,
                           surface_key, signal_kind, half_life_seconds,
                           dt=0.01):
    """
    Perform a single integration step of the hybrid system.

    `state` is a dict containing V, m, h, n.
    `rlct_data` is a tuple (train_losses_per_n, n_values).
    The function updates the neuronal variables using a simple Euler step on
    the membrane potential derivative derived from the ionic currents, while
    also updating the pheromone signal (here we inject a random value to mimic
    environmental feedback).
    Returns the updated state and the hybrid objective value.
    """
    V, m, h, n = state['V'], state['m'], state['h'], state['n']
    train_losses_per_n, n_values = rlct_data

    # Compute hybrid objective (used as a scalar feedback)
    J = hybrid_rlct_pheromone_energy(
        V, m, h, n,
        train_losses_per_n, n_values,
        pheromone_system, surface_key, signal_kind, half_life_seconds)

    # Simple dynamics: dV/dt = - (I_Na + I_K) / C   (C=1 for simplicity)
    I_Na = sodium_current(V, m, h)
    I_K  = potassium_current(V, n)
    dV = - (I_Na + I_K) * dt
    V_new = V + dV

    # Update gating variables with naive first‑order kinetics
    # (These are placeholders; in a full model they'd follow alpha/beta rates)
    m_new = m + dt * (0.1 * (1 - m) - 0.05 * m)
    h_new = h + dt * (0.07 * (1 - h) - 0.03 * h)
    n_new = n + dt * (0.01 * (1 - n) - 0.02 * n)

    # Randomly perturb pheromone to simulate discovery of a resource
    random_signal = random.uniform(0.0, 1.0)
    pheromone_system.update_pheromone_signal(
        surface_key, signal_kind, random_signal, half_life_seconds)

    new_state = {'V': V_new, 'm': m_new, 'h': h_new, 'n': n_new}
    return new_state, J

def run_hybrid_demo():
    """Smoke‑test that runs a short hybrid simulation and prints results."""
    # Initial neuronal state
    state = {'V': 10.0, 'm': 0.5, 'h': 0.5, 'n': 0.5}

    # RLCT data (synthetic)
    train_losses_per_n = [10.0, 20.0, 30.0]
    n_values = [100.0, 200.0, 300.0]
    rlct_data = (train_losses_per_n, n_values)

    # Pheromone system
    pheromone_system = PheromoneSystem()
    surface_key = 'dendrite_surface'
    signal_kind = 'nutrient'
    half_life_seconds = 60.0

    # Run a few steps
    for step in range(5):
        state, J = hybrid_simulation_step(
            state, rlct_data, pheromone_system,
            surface_key, signal_kind, half_life_seconds)
        print(f"Step {step+1}: V={state['V']:.3f}, J={J:.3f}")

if __name__ == "__main__":
    run_hybrid_demo()