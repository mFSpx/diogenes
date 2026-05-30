# DARWIN HAMMER — match 2082, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_ternar_m333_s0.py (gen3)
# parent_b: hybrid_ternary_router_hybrid_minimum_cost__m36_s2.py (gen2)
# born: 2026-05-29T23:40:52Z

"""
Hybrid Algorithm: DARWIN HAMMER Fusion of Liquid‑Time‑Constant Gating,
Variational Free Energy, and Bayesian Minimum‑Cost Routing.

Parents
-------
* **Parent A** – ``hybrid_hybrid_hybrid_worksh_hybrid_hybrid_ternar_m333_s0.py``  
  Provides a Liquid‑Time‑Constant (LTC) gating mechanism and a variational
  free‑energy functional `variational_free_energy(q, p)`.

* **Parent B** – ``hybrid_ternary_router_hybrid_minimum_cost__m36_s2.py``  
  Supplies a Bayesian minimum‑cost tree router that maintains edge priors and
  selects an execution engine (group) with the lowest expected cost.

Mathematical Bridge
-------------------
Both parents manipulate probability‑like quantities:

* The LTC gate in **A** receives a scalar `gating` and a similarity measure.
* The router in **B** assigns a prior probability `π_e` to each edge and
  updates it with Bayes’ rule.

We fuse them by letting the LTC time constant `τ` be modulated by the
variational free energy `F = Σ q·log(q/p)` of the router’s posterior
distribution and by a MinHash similarity `s`.  The effective cost of routing
to a group `g` becomes


C_g = π_g * τ(g)                     (1)
τ(g) = gating * (1 + α·F_g) * (1 + β·s_g)   (2)


where `α,β` are tunable scalars.  The router then chooses the group with the
minimal `C_g`.  After routing, the edge priors are updated with Bayes’
rule using a likelihood derived from the observed cost.

The resulting module contains three core functions that demonstrate this
hybrid operation:
`variational_free_energy`, `hybrid_liquid_time_constant`, and
`hybrid_route_packet`.  A lightweight smoke test exercises the full pipeline.
"""

import math
import random
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared constants and helpers (from Parent A)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Tuple[str, ...], dow: int) -> np.ndarray:
    """
    Produce a normalized weight vector for *groups* based on the weekday ``dow``.
    """
    n = len(groups)
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def _hash(seed: int, token: str) -> int:
    """Hash a token with a 4‑byte seed using a simple mixing function."""
    h = np.uint64(seed)
    for c in token:
        h = np.uint64(h ^ ord(c))
        h = np.uint64(h * 0x100000001B3)
        h &= MAX64
    return int(h)


def minhash_similarity(token_a: str, token_b: str, seed: int = 0xDEADBEEF) -> float:
    """
    Compute a Jaccard‑like similarity from two MinHash signatures.
    The result is in [0, 1].
    """
    ha = _hash(seed, token_a)
    hb = _hash(seed, token_b)
    # Hamming distance on 64‑bit signatures
    diff = bin(ha ^ hb).count("1")
    return 1.0 - diff / 64.0


# ----------------------------------------------------------------------
# Parent A – Variational Free Energy (unchanged)
# ----------------------------------------------------------------------
def variational_free_energy(q: np.ndarray, p: np.ndarray) -> float:
    """
    Compute the variational free energy between two probability distributions.

    Parameters
    ----------
    q : np.ndarray
        Approximate distribution (must sum to 1, non‑negative).
    p : np.ndarray
        Target distribution (must sum to 1, non‑negative).

    Returns
    -------
    float
        The variational free energy Σ q·log(q/p).
    """
    # Guard against division by zero
    eps = np.finfo(float).eps
    q_safe = np.clip(q, eps, 1.0)
    p_safe = np.clip(p, eps, 1.0)
    return float(np.sum(q_safe * np.log(q_safe / p_safe)))


# ----------------------------------------------------------------------
# Hybrid LTC – integrates free energy and MinHash similarity
# ----------------------------------------------------------------------
def hybrid_liquid_time_constant(
    gating: float,
    minhash_sim: float,
    free_energy: float,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> float:
    """
    Compute an effective Liquid‑Time‑Constant τ that is modulated by
    variational free energy and MinHash similarity.

    τ = gating * (1 + α·F) * (1 + β·s)

    Parameters
    ----------
    gating : float
        Base gating signal (typically in (0, 1]).
    minhash_sim : float
        Similarity between packet token and group token, in [0, 1].
    free_energy : float
        Variational free energy for the group.
    alpha, beta : float
        Scaling coefficients that control the influence of free energy and
        similarity respectively.

    Returns
    -------
    float
        Effective time constant τ.
    """
    tau = gating * (1.0 + alpha * free_energy) * (1.0 + beta * minhash_sim)
    return _pct(tau)


# ----------------------------------------------------------------------
# Bayesian Minimum‑Cost Tree utilities (derived from Parent B)
# ----------------------------------------------------------------------
def initialize_edge_priors(groups: Tuple[str, ...]) -> Dict[Tuple[str, str], float]:
    """
    Initialise a uniform prior probability for every directed edge
    (source → target) in the complete graph over *groups*.
    """
    n = len(groups)
    uniform = 1.0 / (n - 1)  # exclude self‑loops
    priors: Dict[Tuple[str, str], float] = {}
    for src in groups:
        for dst in groups:
            if src != dst:
                priors[(src, dst)] = uniform
    return priors


def bayesian_update_prior(
    prior: float, likelihood: float, evidence: float
) -> float:
    """
    Apply Bayes' rule for a single edge prior.

    posterior ∝ prior * likelihood
    normalized by evidence.
    """
    if evidence == 0:
        return prior  # avoid division by zero
    posterior = prior * likelihood / evidence
    # Clamp to [0,1] and renormalise later if needed
    return max(0.0, min(1.0, posterior))


def normalize_edge_priors(priors: Dict[Tuple[str, str], float]) -> None:
    """
    Renormalise outgoing priors for each source node so that they sum to 1.
    """
    outgoing: Dict[str, float] = {}
    for (src, _), prob in priors.items():
        outgoing[src] = outgoing.get(src, 0.0) + prob

    for (src, dst), prob in list(priors.items()):
        total = outgoing[src]
        if total > 0:
            priors[(src, dst)] = prob / total


# ----------------------------------------------------------------------
# Hybrid Routing --------------------------------------------------------
# ----------------------------------------------------------------------
def hybrid_route_packet(
    packet_token: str,
    gating: float = 0.8,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> Tuple[str, Dict[Tuple[str, str], float]]:
    """
    Route a packet token to one of the GROUPS using the hybrid cost model.

    Steps
    -----
    1. Compute a weekday‑based weight vector w.
    2. For each group g:
       * similarity s_g = MinHash(packet_token, g)
       * construct a provisional distribution q_g (here a one‑hot on g)
       * prior distribution p_g is derived from outgoing edge priors.
       * free energy F_g = variational_free_energy(q_g, p_g)
       * τ_g = hybrid_liquid_time_constant(gating, s_g, F_g, α, β)
       * cost C_g = π_g * τ_g, where π_g is the sum of priors on edges
         incoming to g (treated as the group's prior weight).
    3. Choose the group with minimal cost.
    4. Update edge priors using a likelihood proportional to exp(‑C_g).

    Returns
    -------
    chosen_group : str
        The selected engine channel.
    updated_priors : dict
        Edge priors after the Bayesian update.
    """
    # 1. Weekday weight vector (adds a deterministic bias)
    today = datetime.now()
    dow = doomsday(today.year, today.month, today.day)
    w_vec = weekday_weight_vector(GROUPS, dow)

    # 2. Initialise or retrieve edge priors (static for this demo)
    priors = initialize_edge_priors(GROUPS)

    # Pre‑compute incoming prior mass for each group
    incoming_mass: Dict[str, float] = {g: 0.0 for g in GROUPS}
    for (src, dst), prob in priors.items():
        incoming_mass[dst] += prob

    # Containers for costs
    costs: Dict[str, float] = {}
    taus: Dict[str, float] = {}

    for idx, group in enumerate(GROUPS):
        # similarity
        sim = minhash_similarity(packet_token, group)

        # q_g : one‑hot distribution over groups (used for free energy)
        q = np.zeros(len(GROUPS))
        q[idx] = 1.0

        # p_g : normalized incoming prior mass vector
        p = np.array([incoming_mass[g] for g in GROUPS])
        p_sum = p.sum()
        if p_sum == 0:
            p = np.ones_like(p) / len(p)
        else:
            p = p / p_sum

        # free energy
        F = variational_free_energy(q, p)

        # effective time constant
        tau = hybrid_liquid_time_constant(gating, sim, F, alpha, beta)
        taus[group] = tau

        # cost = prior_mass * τ
        cost = incoming_mass[group] * tau
        costs[group] = cost

    # 3. Choose minimal cost
    chosen_group = min(costs, key=costs.get)

    # 4. Bayesian update of priors
    # Likelihood for each edge (src → dst) proportional to exp(-C_dst)
    likelihoods: Dict[Tuple[str, str], float] = {}
    evidence = 0.0
    for (src, dst), _ in priors.items():
        lik = math.exp(-costs[dst])  # lower cost → higher likelihood
        likelihoods[(src, dst)] = lik
        evidence += lik

    # Update each prior
    for edge, prior_val in priors.items():
        lik = likelihoods[edge]
        post = bayesian_update_prior(prior_val, lik, evidence)
        priors[edge] = post

    # Renormalise outgoing edges
    normalize_edge_priors(priors)

    return chosen_group, priors


# ----------------------------------------------------------------------
# Smoke test ------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example packet token (could be any string)
    token = "user:ask:weather"

    # Run the hybrid router
    chosen, updated = hybrid_route_packet(token)

    print(f"Chosen group: {chosen}")
    print("Updated edge priors (sample):")
    # Show a few edges for brevity
    sample_edges = list(updated.items())[:8]
    for (src, dst), prob in sample_edges:
        print(f"  {src} → {dst} : {_pct(prob)}")
    # Verify that outgoing probabilities from each source sum to ~1
    for src in GROUPS:
        out_sum = sum(
            prob for (s, _), prob in updated.items() if s == src
        )
        print(f"Outgoing sum from {src}: {_pct(out_sum)}")
    sys.exit(0)