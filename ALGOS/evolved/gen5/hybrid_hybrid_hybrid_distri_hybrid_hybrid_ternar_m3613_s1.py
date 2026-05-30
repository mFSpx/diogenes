# DARWIN HAMMER — match 3613, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_distributed_l_hybrid_physarum_netw_m110_s5.py (gen4)
# parent_b: hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m170_s1.py (gen4)
# born: 2026-05-29T23:50:54Z

"""Hybrid Leader‑Physarum & Ternary‑Softmax Scheduler
===================================================

Parent A: ``hybrid_hybrid_distributed_l_hybrid_physarum_netw_m110_s5.py`` – a
simulated‑annealing leader election combined with a Physarum‑inspired flow
network.  Its core scalar is the *temperature* obtained from an exponential
cooling schedule and a broadcast probability.

Parent B: ``hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m170_s1.py`` – a
ternary‑vector encoder whose output is fed to a ternary‑softmax activation
function and subsequently to a learning‑style weight update (here used to
modulate the Physarum conductances).

**Mathematical bridge**

Both parents expose a *scalar* that governs adaptation:

* A’s temperature   :  Tₐ = cooling_temperature(k) · broadcast_probability(phases, phase)
* B’s ternary‑softmax scaling factor :  β = 1 / temperature

We fuse them by letting the joint temperature `temp` scale the ternary‑softmax
output, which in turn multiplicatively weights the Physarum conductance
update.  The same `temp` also serves as the Metropolis temperature for the
leader‑election acceptance probability.  Consequently a single scalar drives
(1) stochastic leader selection, (2) deterministic flux‑based network
adaptation, and (3) ternary‑aware modulation of edge conductances.

The module provides three representative functions:

* ``joint_temperature`` – computes the fused temperature.
* ``ternary_softmax`` – ternary‑softmax activation scaled by the joint temperature.
* ``physarum_update`` – conductance update that incorporates flux, temperature,
  and ternary activation.
* ``leader_election_step`` – Metropolis‑style leader selection using the same
  temperature.

All functions are pure NumPy / std‑lib and can be combined arbitrarily.
"""

from __future__ import annotations

import math
import random
import sys
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Hashable, Mapping, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
Edge = Tuple[Node, Node]

# ----------------------------------------------------------------------
# Parent A utilities (broadcast probability & cooling schedule)
# ----------------------------------------------------------------------
def broadcast_probability(phases: int, phase: int) -> float:
    """Original A: p = 1 / 2^{phases‑phase}, clamped to [0,1]."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Exponential cooling schedule: T = t0 * alpha^{k}."""
    if k < 0:
        raise ValueError("k must be non‑negative")
    if t0 < 0:
        raise ValueError("t0 must be non‑negative")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0,1]")
    return t0 * (alpha ** k)


def joint_temperature(phases: int, phase: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """
    Fuse A’s broadcast probability and cooling temperature.

    temp = cooling_temperature(phase‑1) * broadcast_probability(phases, phase)
    """
    return cooling_temperature(phase - 1, t0, alpha) * broadcast_probability(phases, phase)


# ----------------------------------------------------------------------
# Parent B utilities (ternary vector & softmax)
# ----------------------------------------------------------------------
TERNARY_DIMS = 12  # Fixed dimensionality used by the original parent


def payload_hash(raw_command: str, normalized_intent: str, context: Dict[str, Any]) -> str:
    """Deterministic SHA‑256 of the command envelope."""
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ternary_vector(raw_command: str, normalized_intent: str, context: Dict[str, Any]) -> np.ndarray:
    """
    Produce a ternary vector (values in {‑1,0,1}) from a hashed payload.
    The original implementation used 2‑byte chunks; we preserve that logic.
    """
    payload = payload_hash(raw_command, normalized_intent, context)
    vec = np.zeros((TERNARY_DIMS,), dtype=np.float64)
    # Each 2‑hex‑digit chunk yields a byte (0‑255)
    for i in range(0, min(len(payload), TERNARY_DIMS * 2), 2):
        byte = int(payload[i : i + 2], 16)
        vec[i // 2] = (byte % 4) - 2  # maps 0→‑2, 1→‑1, 2→0, 3→1 → shift to {‑1,0,1}
    # Clamp to {-1,0,1}
    vec = np.clip(vec, -1, 1)
    return vec


def ternary_softmax(vec: np.ndarray, temperature: float) -> np.ndarray:
    """
    Ternary‑softmax activation.

    For each component v_i we compute:
        s_i = exp(v_i / temperature) / Σ_j exp(v_j / temperature)

    The temperature acts as a smoothing factor; lower temperature sharpens
    the distribution.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive for softmax")
    scaled = vec / temperature
    # Numerical stability: subtract max
    shifted = scaled - np.max(scaled)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def physarum_update(
    conductances: np.ndarray,
    fluxes: np.ndarray,
    temperature: float,
    ternary_activation: np.ndarray,
) -> np.ndarray:
    """
    Conductance update that merges Physarum dynamics with ternary modulation.

    Classic Physarum update (simplified):
        g_{new} = g + η * (|q| - g)

    Here η is replaced by the joint temperature and the ternary activation
    acts as an element‑wise scaling factor:

        g_{new} = g + temperature * ternary_activation * (|q| - g)

    Parameters
    ----------
    conductances : np.ndarray
        Current edge conductances (shape N).
    fluxes : np.ndarray
        Current absolute fluxes on edges (shape N).
    temperature : float
        Joint temperature from ``joint_temperature``.
    ternary_activation : np.ndarray
        Output of ``ternary_softmax`` (shape N or broadcastable).

    Returns
    -------
    np.ndarray
        Updated conductances.
    """
    if conductances.shape != fluxes.shape:
        raise ValueError("conductances and fluxes must share shape")
    # Ensure ternary_activation can broadcast
    scaling = temperature * ternary_activation
    delta = np.abs(fluxes) - conductances
    return conductances + scaling * delta


def leader_election_step(
    current_leader: Any,
    candidate: Any,
    fitness_current: float,
    fitness_candidate: float,
    temperature: float,
) -> Any:
    """
    Metropolis‑style leader selection.

    Acceptance probability:
        a = min(1, exp((f_candidate - f_current) / temperature))

    If a random draw ≤ a, the candidate becomes the new leader.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive for Metropolis acceptance")
    delta = fitness_candidate - fitness_current
    acceptance = min(1.0, math.exp(delta / temperature))
    if random.random() <= acceptance:
        return candidate
    return current_leader


# ----------------------------------------------------------------------
# Helper utilities for a minimal Physarum flux model
# ----------------------------------------------------------------------
def random_fluxes(num_edges: int, seed: int = 0) -> np.ndarray:
    """Generate synthetic absolute flux values for demonstration purposes."""
    rng = np.random.default_rng(seed)
    return rng.random(num_edges)  # values in [0,1)


def init_conductances(num_edges: int, scale: float = 0.1, seed: int = 0) -> np.ndarray:
    """Initialize conductances with small positive random values."""
    rng = np.random.default_rng(seed)
    return rng.random(num_edges) * scale + 0.01


# ----------------------------------------------------------------------
# Smoke‑test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration
    phases = 10
    phase = 3
    t0 = 1.0
    alpha = 0.92

    # Joint temperature
    temp = joint_temperature(phases, phase, t0, alpha)
    print(f"Joint temperature (phase {phase}/{phases}): {temp:.6f}")

    # Ternary vector from a dummy command
    raw_cmd = "run simulation"
    intent = "execute"
    ctx = {"user": "alice", "priority": "high"}
    t_vec = ternary_vector(raw_cmd, intent, ctx)
    print("Ternary vector:", t_vec)

    # Ternary‑softmax activation (scaled by temperature)
    t_act = ternary_softmax(t_vec, temperature=temp)
    print("Ternary softmax activation:", t_act)

    # Simple Physarum network with 12 edges (matching TERNARY_DIMS)
    num_edges = TERNARY_DIMS
    g = init_conductances(num_edges, scale=0.05, seed=42)
    q = random_fluxes(num_edges, seed=99)

    print("Initial conductances:", g)
    print("Fluxes:", q)

    # Update conductances using the hybrid rule
    g_new = physarum_update(g, q, temperature=temp, ternary_activation=t_act)
    print("Updated conductances:", g_new)

    # Leader election demo
    current_leader = "node_A"
    candidate = "node_B"
    fitness_current = random.random()
    fitness_candidate = random.random()
    new_leader = leader_election_step(
        current_leader,
        candidate,
        fitness_current,
        fitness_candidate,
        temperature=temp,
    )
    print(
        f"Leader election: current={current_leader} (f={fitness_current:.3f}), "
        f"candidate={candidate} (f={fitness_candidate:.3f}) -> new leader={new_leader}"
    )

    sys.exit(0)