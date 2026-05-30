# DARWIN HAMMER — match 5584, survivor 3
# gen: 6
# parent_a: hybrid_rlct_grokking_dendritic_compartmen_m61_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m599_s0.py (gen5)
# born: 2026-05-30T00:03:11Z

"""Hybrid RLCT-Grokking Dendritic Bandit Router

This module fuses the core of **Parent A** (RLCT/Grokking with the Hodgkin‑Huxley
membrane‑potential dynamics) and **Parent B** (health‑score driven bandit
selection with Hoeffding confidence bounds and ternary routing).

Mathematical bridge
------------------
* The Hodgkin‑Huxley equations provide a *state variable* `V` (membrane
  potential) that evolves under synaptic current `I_syn`.  
* Singular Learning Theory supplies a *free‑energy* `F` that quantifies the
  learning landscape via the Real Log Canonical Threshold (RLCT).  

Both quantities are energies; we map them onto a *health score* `h` for each
endpoint:


h = exp( -α·V²  -  β·F )


where `α,β>0` weight the biophysical and statistical contributions.
The health scores become the context vector for a stochastic‑bandit
algorithm.  Using a Hoeffding bound we construct an Upper Confidence Bound
(UCB) for each endpoint and select the one with the highest bound.  The
chosen endpoint then routes the incoming packet (ternary‑style routing).

The resulting system simultaneously evolves a biophysical state, evaluates
statistical learning complexity, and makes data‑driven routing decisions.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Hodgkin‑Huxley membrane potential & RLCT free energy
# ----------------------------------------------------------------------


def calculate_membrane_potential(
    V: float,
    C_m: float,
    g_L: float,
    E_L: float,
    g_Na: float,
    E_Na: float,
    m: float,
    h: float,
    g_K: float,
    E_K: float,
    n_gate: float,
    I_syn: float,
) -> float:
    """
    One Euler step of the Hodgkin‑Huxley cable model.

    Returns the updated membrane potential after a single time step Δt=1.
    """
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n_gate ** 4) * (V - E_K)
    I_L = g_L * (V - E_L)
    dV_dt = (-I_L - I_Na - I_K + I_syn) / C_m
    return V + dV_dt


def calculate_free_energy(
    n: int,
    L0: float,
    lambda_rlct: float,
    m_param: int = 1,
) -> float:
    """
    Asymptotic free‑energy from Singular Learning Theory:

        F_n = n·L0 + λ·log n + (m−1)·log log n  + o(1)

    Parameters
    ----------
    n : int
        Dataset size (must be > 1).
    L0 : float
        True risk (minimum achievable loss).
    lambda_rlct : float
        Real Log Canonical Threshold.
    m_param : int, optional
        Multiplicity of the singularity (default 1).

    Returns
    -------
    float
        Approximate free energy.
    """
    if n <= 1:
        raise ValueError("n must be greater than 1 for logarithms.")
    term1 = n * L0
    term2 = lambda_rlct * math.log(n)
    term3 = (m_param - 1) * math.log(math.log(n)) if m_param > 1 else 0.0
    return term1 + term2 + term3


# ----------------------------------------------------------------------
# Parent B – Hoeffding‑bound bandit with ternary routing
# ----------------------------------------------------------------------


@dataclass
class Endpoint:
    """Endpoint descriptor used by the bandit router."""
    name: str
    health_score: float = 0.0
    failure_rate: float = 0.0
    recovery_priority: float = 0.0
    observations: int = 0  # number of times selected


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """
    Hoeffding bound for a bounded random variable in [0, r].

    Parameters
    ----------
    r : float
        Range (max – min) of the variable.
    delta : float
        Desired failure probability (0 < delta < 1).
    n : int
        Number of independent observations.

    Returns
    -------
    float
        Upper confidence radius.
    """
    if n <= 0:
        return float("inf")
    return r * math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def health_score_from_physics(V: float, free_energy: float, alpha: float = 0.01, beta: float = 0.001) -> float:
    """
    Map biophysical and statistical energies onto a scalar health score.

    The exponential form guarantees a score in (0, 1].

    Parameters
    ----------
    V : float
        Membrane potential.
    free_energy : float
        Free energy from Singular Learning Theory.
    alpha, beta : float
        Weighting coefficients.

    Returns
    -------
    float
        Health score.
    """
    exponent = -alpha * (V ** 2) - beta * free_energy
    # Clip to avoid underflow for extreme negatives
    exponent = max(exponent, -700)  # np.exp(-700) ~ 5e-305
    return math.exp(exponent)


def select_endpoint(endpoints: List[Endpoint], delta: float = 0.05, r: float = 1.0) -> Endpoint:
    """
    Bandit selection using Hoeffding‑UCB.

    For each endpoint we compute:
        UCB = health_score + HoeffdingBound(r, delta, observations)

    The endpoint with the maximal UCB is returned.
    """
    best_ep = None
    best_ucb = -float("inf")
    for ep in endpoints:
        bound = hoeffding_bound(r, delta, ep.observations)
        ucb = ep.health_score + bound
        if ucb > best_ucb:
            best_ucb = ucb
            best_ep = ep
    return best_ep  # type: ignore


def route_packet(packet: Dict[str, Any], endpoints: List[Endpoint], delta: float = 0.05) -> Dict[str, Any]:
    """
    Perform ternary‑style routing based on bandit selection.

    The packet is annotated with the chosen endpoint name and a simple
    routing flag.
    """
    chosen = select_endpoint(endpoints, delta=delta)
    # Update observation count (simulating a reward‑free pull)
    chosen.observations += 1

    routed = packet.copy()
    routed["routed_to"] = chosen.name
    routed["routing_success"] = True
    return routed


# ----------------------------------------------------------------------
# Hybrid orchestration
# ----------------------------------------------------------------------


def hybrid_step(
    V: float,
    C_m: float,
    g_L: float,
    E_L: float,
    g_Na: float,
    E_Na: float,
    m_gate: float,
    h_gate: float,
    g_K: float,
    E_K: float,
    n_gate: float,
    I_syn: float,
    n_data: int,
    L0: float,
    lambda_rlct: float,
    endpoints: List[Endpoint],
) -> Dict[str, Any]:
    """
    Execute one hybrid iteration:
      1. Update membrane potential.
      2. Compute free energy.
      3. Refresh health scores of all endpoints.
      4. Route a dummy packet using the bandit router.

    Returns a dictionary summarising the step.
    """
    # 1. Biophysical update
    V_new = calculate_membrane_potential(
        V, C_m, g_L, E_L, g_Na, E_Na, m_gate, h_gate, g_K, E_K, n_gate, I_syn
    )

    # 2. Statistical free energy
    F = calculate_free_energy(n_data, L0, lambda_rlct)

    # 3. Update health scores
    for ep in endpoints:
        ep.health_score = health_score_from_physics(V_new, F)

    # 4. Routing
    dummy_packet = {"payload": "example", "timestamp": sys.maxsize}
    routed_packet = route_packet(dummy_packet, endpoints)

    return {
        "V_before": V,
        "V_after": V_new,
        "free_energy": F,
        "selected_endpoint": routed_packet["routed_to"],
        "packet": routed_packet,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Hodgkin‑Huxley baseline parameters (arbitrary but plausible)
    V0 = -65.0  # mV
    C_m = 1.0   # µF/cm²
    g_L = 0.1   # mS/cm²
    E_L = -65.0
    g_Na = 120.0
    E_Na = 50.0
    m_gate = 0.05
    h_gate = 0.6
    g_K = 36.0
    E_K = -77.0
    n_gate = 0.32
    I_syn = 10.0  # external current

    # Singular Learning Theory parameters
    n_data = 10000
    L0 = 0.02
    lambda_rlct = 0.5

    # Create a small pool of endpoints
    endpoint_names = ["alpha", "beta", "gamma"]
    endpoints = [Endpoint(name=n) for n in endpoint_names]

    # Run a few hybrid steps
    state = V0
    for step in range(5):
        result = hybrid_step(
            V=state,
            C_m=C_m,
            g_L=g_L,
            E_L=E_L,
            g_Na=g_Na,
            E_Na=E_Na,
            m_gate=m_gate,
            h_gate=h_gate,
            g_K=g_K,
            E_K=E_K,
            n_gate=n_gate,
            I_syn=I_syn,
            n_data=n_data,
            L0=L0,
            lambda_rlct=lambda_rlct,
            endpoints=endpoints,
        )
        state = result["V_after"]
        print(
            f"Step {step+1}: V={result['V_before']:.2f}->{result['V_after']:.2f}, "
            f"FreeEnergy={result['free_energy']:.2f}, "
            f"Chosen={result['selected_endpoint']}"
        )
    print("Smoke test completed without errors.")