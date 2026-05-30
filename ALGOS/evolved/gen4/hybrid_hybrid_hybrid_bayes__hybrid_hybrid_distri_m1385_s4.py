# DARWIN HAMMER — match 1385, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s0.py (gen3)
# parent_b: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s4.py (gen2)
# born: 2026-05-29T23:35:57Z

"""Hybrid Bayesian‑Hoeffding‑Tree‑Ternary Router Algorithm
======================================================

Parents
-------
* **hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s0.py** –  
  Bayesian update of a high‑dimensional “brain‑map” feature vector and
  Structural‑Similarity‑Index (SSIM)‑driven action selection.
* **hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s4.py** –  
  Distributed leader election using simulated‑annealing acceptance of
  Hoeffding‑tree splits, with split quality measured by a tropical
  max‑plus gain.

Mathematical Bridge
-------------------
Both parents decide *whether* a structural change is kept:

* The Bayesian side produces a posterior distribution **p(θ|x)** over the
  feature vector **θ**.  Its Shannon entropy **H(p)** is a measure of
  uncertainty; we treat **H** as a temperature‑like quantity **T**.
* The Hoeffding‑tree side computes a Hoeffding bound **ε** for a candidate
  split and a tropical max‑plus gain **G**.  The energy difference is
  defined as  

  ``ΔE = ε – G``  

  (small bound & large gain ⇒ low energy).

The hybrid acceptance probability is therefore  

``α = exp(‑ΔE / T)``  

which is exactly the simulated‑annealing rule, but with the temperature
derived from the Bayesian posterior.  A split is accepted (and becomes a
distributed leader) if a uniform random draw is ≤ α.

The module below implements this unified system, exposing three core
functions that demonstrate the hybrid operation."""


import math
import random
import sys
import pathlib
import numpy as np
from collections.abc import Mapping, Hashable

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]

# ----------------------------------------------------------------------
# Parent‑A utilities (Bayesian feature handling)
# ----------------------------------------------------------------------
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)


def extract_full_features(text: str) -> dict[str, float]:
    """Generate a synthetic high‑dimensional feature dict."""
    features: dict[str, float] = {}
    features.update({
        "operator_visceral_ratio": random.random(),
        "operator_tech_ratio": random.random(),
        "operator_legal_osint_ratio": random.random(),
    })
    features.update({
        "psyche_forensic_shield_ratio": random.random(),
        "psyche_poetic_entropy": random.random(),
        "psyche_dissociative_index": random.random(),
    })
    features.update({
        "resilience_bureaucratic_weaponization_index": random.random(),
        "resilience_resource_exhaustion_metric": random.random(),
        "resilience_swarm_orchestration_density": random.random(),
    })
    features.update({
        "rainmaker_corporate_grit_tension": random.random(),
        "rainmaker_countdown_density": random.random(),
        "rainmaker_asset_structuring_weight": random.random(),
    })
    features.update({
        "telemetry_agent_symmetry_ratio": random.random(),
        "telemetry_protocol_discipline": random.random(),
        "telemetry_manic_velocity": random.random(),
    })
    return features


def extract_master_vector(text: str) -> np.ndarray:
    """Return a 5‑element vector extracted from the full feature dict."""
    f = extract_full_features(text)
    vec = np.array([
        f.get("operator_visceral_ratio", 0.0),
        f.get("operator_tech_ratio", 0.0),
        f.get("operator_legal_osint_ratio", 0.0),
        f.get("psyche_forensic_shield_ratio", 0.0),
        f.get("psyche_poetic_entropy", 0.0),
    ], dtype=np.float64)
    # Normalise to unit L2 norm to avoid scale issues.
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def ssim_like_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """A lightweight SSIM‑style similarity (range 0‑1)."""
    mu1, mu2 = v1.mean(), v2.mean()
    sigma1, sigma2 = v1.var(), v2.var()
    cov = ((v1 - mu1) * (v2 - mu2)).mean()
    C1, C2 = 1e-6, 1e-6
    numerator = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    return float(numerator / denominator) if denominator != 0 else 0.0


def bayesian_posterior(prior: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """Simple Bayesian update using a Gaussian‑like likelihood derived from SSIM."""
    likelihood = ssim_like_similarity(prior, observation) * np.ones_like(prior)
    unnorm = prior * likelihood
    total = unnorm.sum()
    return unnorm / total if total > 0 else prior


def shannon_entropy(p: np.ndarray) -> float:
    """Compute Shannon entropy (base e) of a probability vector."""
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    return -float(np.sum(p * np.log(p)))


# ----------------------------------------------------------------------
# Parent‑B utilities (Hoeffding‑tree + simulated annealing)
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    """Probability that a node broadcasts in the current phase."""
    if total_phases < 1 or current_phase < 1:
        raise ValueError("phases and phase must be positive")
    # Decay exponentially with the distance to the final phase.
    remaining = max(0, total_phases - current_phase)
    return min(1.0, 1.0 / (2 ** remaining))


def hoeffding_bound(N: int, R: float, delta: float) -> float:
    """
    Hoeffding bound ε = sqrt( (R^2 * ln(1/δ)) / (2N) )
    N – number of observations,
    R – range of the random variable (max‑min),
    δ – desired failure probability.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2.0 * N))


def tropical_max_plus_gain(stats: dict[str, float]) -> float:
    """
    Tropical max‑plus “gain” G = max_i (a_i) + sum_j (b_j)
    where a_i are positive contributions and b_j are penalties.
    For demonstration we treat 'info_gain' as a_i and 'complexity' as b_j.
    """
    a = stats.get("info_gain", 0.0)
    b = stats.get("complexity", 0.0)
    return max(a, 0.0) - b  # max‑plus analogue (larger gain → lower energy)


def simulated_annealing_accept(delta_E: float, temperature: float) -> bool:
    """Accept with probability exp(-ΔE / T)."""
    if temperature <= 0:
        return delta_E <= 0  # deterministic at zero temperature
    prob = math.exp(-delta_E / temperature)
    return random.random() < prob


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_split_acceptance(
    prior_vec: np.ndarray,
    observed_vec: np.ndarray,
    N: int,
    R: float,
    delta: float,
    split_stats: dict[str, float],
) -> bool:
    """
    Decide whether a candidate Hoeffding‑tree split should be kept.

    Steps
    -----
    1. Bayesian posterior over the feature vector → temperature T = H(posterior).
    2. Hoeffding bound ε = hoeffding_bound(N, R, δ).
    3. Tropical gain G = tropical_max_plus_gain(split_stats).
    4. Energy difference ΔE = ε – G.
    5. Accept with probability exp(-ΔE / T).
    """
    # 1. Posterior & temperature
    posterior = bayesian_posterior(prior_vec, observed_vec)
    T = shannon_entropy(posterior) + 1e-9  # avoid division by zero

    # 2. Hoeffding bound
    epsilon = hoeffding_bound(N, R, delta)

    # 3. Tropical gain
    G = tropical_max_plus_gain(split_stats)

    # 4. Energy difference
    delta_E = epsilon - G

    # 5. Acceptance
    return simulated_annealing_accept(delta_E, T)


def hybrid_bayes_tree_update(
    text: str,
    total_phases: int,
    current_phase: int,
    N: int,
    R: float,
    delta: float,
    split_stats: dict[str, float],
) -> dict[str, any]:
    """
    Perform a single hybrid update cycle.

    Returns a dictionary containing:
        * 'posterior' – updated feature distribution,
        * 'temperature' – derived from posterior entropy,
        * 'split_accepted' – boolean decision,
        * 'broadcast_prob' – probability of broadcasting in this phase.
    """
    # Feature extraction
    prior_vec = extract_master_vector(text)

    # Simulated observation (in a real system this would be new data)
    observed_vec = extract_master_vector("observation:" + text)

    # Broadcast probability (leader‑election primitive)
    broadcast_prob = broadcast_probability(total_phases, current_phase)

    # Hybrid acceptance decision
    split_accepted = compute_split_acceptance(
        prior_vec, observed_vec, N, R, delta, split_stats
    )

    # Posterior for downstream use
    posterior = bayesian_posterior(prior_vec, observed_vec)

    return {
        "posterior": posterior,
        "temperature": shannon_entropy(posterior),
        "split_accepted": split_accepted,
        "broadcast_prob": broadcast_prob,
    }


def hybrid_leader_election(
    graph: Graph,
    node: Node,
    total_phases: int,
    current_phase: int,
) -> bool:
    """
    Decide if *node* becomes a leader in the distributed graph.
    The decision uses the broadcast probability from Parent‑B and a
    Bayesian‑derived temperature (here we reuse a dummy prior).
    """
    # Dummy prior – uniform over three abstract states
    dummy_prior = np.array([1 / 3, 1 / 3, 1 / 3])
    dummy_observation = np.array([random.random() for _ in range(3)])
    posterior = bayesian_posterior(dummy_prior, dummy_observation)
    T = shannon_entropy(posterior) + 1e-9

    # Energy for leader election: we treat degree as “complexity”
    degree = len(graph.get(node, []))
    G = tropical_max_plus_gain({"info_gain": 1.0, "complexity": degree})

    # Hoeffding bound with synthetic parameters
    epsilon = hoeffding_bound(N=10 + degree, R=1.0, delta=0.05)

    delta_E = epsilon - G
    accept = simulated_annealing_accept(delta_E, T)

    # Finally modulate by broadcast probability
    return accept and (random.random() < broadcast_probability(total_phases, current_phase))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic graph
    test_graph: Graph = {
        "A": {"B", "C"},
        "B": {"A"},
        "C": {"A"},
    }

    # Parameters for the hybrid update
    params = {
        "text": "sample input for Bayesian feature extraction",
        "total_phases": 5,
        "current_phase": 2,
        "N": 100,
        "R": 1.0,
        "delta": 0.01,
        "split_stats": {"info_gain": 0.12, "complexity": 0.04},
    }

    result = hybrid_bayes_tree_update(**params)
    print("Hybrid update result:")
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.round(4)}")
        else:
            print(f"  {k}: {v}")

    # Leader election test
    is_leader = hybrid_leader_election(test_graph, "A", 5, 2)
    print(f"\nNode 'A' elected leader? {is_leader}")