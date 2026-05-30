# DARWIN HAMMER — match 767, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_hoeffding_tre_m44_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hdc_se_m172_s3.py (gen4)
# born: 2026-05-29T23:30:52Z

"""Hybrid Endpoint‑Health‑Fisher Hyperdimensional Module

Parents:
- **Algorithm A** – Hybrid Endpoint‑SSM‑Hoeffding‑Tropical (health scores,
  Hoeffding‑based switching, tropical max‑plus algebra).
- **Algorithm B** – Hybrid Fisher‑JEPA Hyperdimensional (scalar Fisher score
  → bipolar hypervector, hypervector binding/bundling, similarity).

Mathematical Bridge
-------------------
The scalar health score *h* produced by the tropical engine of Algorithm A is
treated as a *Fisher‑like* scalar.  Both scalars are injected into the same
hyperdimensional space by a deterministic hash‑based mapping
`scalar_to_hv`.  For each endpoint *i* we bind its identifier hypervector
`e_i = symbol_vector(id_i)` with its health hypervector `z_i = scalar_to_hv(h_i)`.
The system‑wide representation is the *bundle* (component‑wise majority) of all
bound vectors:


V = ⊎_{i} bind(e_i, z_i)          (⊎ = bundling / majority vote)


Similarity between an endpoint’s bound vector and the global bundle is a
scalar estimate of that endpoint’s relevance.  Hoeffding’s inequality is then
used on this similarity estimate to decide, with confidence `1‑δ`, whether the
current active endpoint should be switched.

The three core functions below demonstrate the full hybrid pipeline:
1. `hybrid_compute_health_scores` – tropical max‑plus health computation.
2. `hybrid_encode_endpoint` – scalar‑to‑hypervector conversion, binding and
   bundling.
3. `hybrid_maybe_switch` – Hoeffding‑bound decision based on hypervector
   similarity.

All operations rely only on NumPy and the Python standard library.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import hashlib

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Endpoint:
    """Endpoint state used by the hybrid system."""
    id: str                     # unique identifier
    failure_rate: float        # observed failure frequency in [0, 1]
    recovery_priority: float   # morphological recovery priority (≥0)
    health_score: float = 0.0  # computed via tropical algebra (filled later)


# ---------------------------------------------------------------------------
# Tropical health computation (Algorithm A)
# ---------------------------------------------------------------------------

def tropical_health(endpoint: Endpoint) -> float:
    """
    Compute a tropical health score using max‑plus algebra.

    In max‑plus, addition is `max` and multiplication is `+`.  We model the
    health as the tropical sum of the negated failure rate (penalty) and the
    recovery priority (reward):

        h = max( -failure_rate , recovery_priority )

    The result lies in ℝ and is later injected into hyperdimensional space.
    """
    return max(-endpoint.failure_rate, endpoint.recovery_priority)


def hybrid_compute_health_scores(endpoints: List[Endpoint]) -> None:
    """
    Populate the `health_score` field of each endpoint using the tropical rule.
    The function mutates the supplied list in‑place.
    """
    for ep in endpoints:
        ep.health_score = tropical_health(ep)


# ---------------------------------------------------------------------------
# Hyperdimensional primitives (Algorithm B)
# ---------------------------------------------------------------------------

Vector = List[int]   # bipolar hypervector (+1 / -1)


def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    """Generate a random bipolar hypervector."""
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Deterministic hypervector for a symbol using SHA‑256 as seed."""
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Component‑wise binding (multiplication) of two hypervectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension for binding.")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: List[Vector]) -> Vector:
    """
    Component‑wise majority vote (bundling).  Ties are broken randomly.
    Returns a bipolar hypervector.
    """
    if not vectors:
        raise ValueError("No vectors to bundle.")
    arr = np.array(vectors, dtype=np.int8)   # shape (n, dim)
    sums = arr.sum(axis=0)                   # positive => +1, negative => -1
    # Resolve zeros (ties) randomly
    ties = sums == 0
    random_choices = np.random.choice([-1, 1], size=ties.sum())
    sums[ties] = random_choices
    return sums.tolist()


def similarity(a: Vector, b: Vector) -> float:
    """
    Normalized Hamming similarity between two bipolar hypervectors.
    Returns a value in [0, 1] where 1 means identical.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension for similarity.")
    a_arr = np.array(a, dtype=np.int8)
    b_arr = np.array(b, dtype=np.int8)
    matches = (a_arr == b_arr).sum()
    return matches / len(a)


# ---------------------------------------------------------------------------
# Scalar → hypervector mapping (bridge between A and B)
# ---------------------------------------------------------------------------

def scalar_to_hv(value: float, dim: int = 10000) -> Vector:
    """
    Deterministically map a real scalar to a bipolar hypervector.
    The mapping hashes the IEEE‑754 binary representation of the float.
    """
    # Convert float to bytes (big‑endian)
    bytes_repr = np.float64(value).tobytes()
    seed_bytes = hashlib.sha256(bytes_repr).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)


# ---------------------------------------------------------------------------
# Fisher score placeholder (Algorithm B component)
# ---------------------------------------------------------------------------

def fisher_score(theta: float) -> float:
    """
    Placeholder for a Fisher information scalar.
    In practice this would be derived from a model; here we use a simple
    quadratic form to keep the module self‑contained.
    """
    return theta * (1.0 - theta)  # maximal at theta = 0.5


def fisher_to_hv(theta: float, dim: int = 10000) -> Vector:
    """Convert a Fisher score into a bipolar hypervector."""
    f = fisher_score(theta)
    return scalar_to_hv(f, dim)


# ---------------------------------------------------------------------------
# Endpoint encoding & system representation
# ---------------------------------------------------------------------------

def hybrid_encode_endpoint(endpoint: Endpoint, dim: int = 10000) -> Vector:
    """
    Encode an endpoint as a bound hypervector:

        v_i = bind( e_i , z_i )

    where `e_i` is the identifier hypervector and `z_i` is the health hypervector.
    """
    e_i = symbol_vector(endpoint.id, dim)
    z_i = scalar_to_hv(endpoint.health_score, dim)
    return bind(e_i, z_i)


def aggregate_system_vector(endpoints: List[Endpoint], dim: int = 10000) -> Vector:
    """
    Produce the system‑wide hypervector by bundling all endpoint encodings.
    """
    bound_vectors = [hybrid_encode_endpoint(ep, dim) for ep in endpoints]
    return bundle(bound_vectors)


# ---------------------------------------------------------------------------
# Hoeffding bound (Algorithm A) and switching decision
# ---------------------------------------------------------------------------

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """
    Hoeffding bound for a bounded random variable in [0, r].

    Returns ε such that P(|mean - μ| > ε) ≤ δ.
    """
    if r <= 0:
        raise ValueError("Range r must be positive.")
    if not (0 < delta < 1):
        raise ValueError("Delta must be in (0,1).")
    if n <= 0:
        raise ValueError("Number of observations n must be positive.")
    return r * math.sqrt(math.log(2 / delta) / (2 * n))


def hybrid_maybe_switch(
    endpoints: List[Endpoint],
    active_id: str,
    delta: float = 0.05,
    n_obs: int = 30,
    dim: int = 10000,
) -> Tuple[str, bool]:
    """
    Decide whether to switch the active endpoint.

    Parameters
    ----------
    endpoints : List[Endpoint]
        Current list of endpoints (health scores must be up‑to‑date).
    active_id : str
        Identifier of the currently active endpoint.
    delta : float
        Desired failure probability for Hoeffding bound.
    n_obs : int
        Number of observations (treated as sample size for the bound).
    dim : int
        Hypervector dimensionality.

    Returns
    -------
    Tuple[str, bool]
        (new_active_id, switched) where `switched` indicates if a change occurred.
    """
    # System representation
    V = aggregate_system_vector(endpoints, dim)

    # Compute similarity for each endpoint
    sims = {}
    for ep in endpoints:
        v_i = hybrid_encode_endpoint(ep, dim)
        sims[ep.id] = similarity(v_i, V)

    # Identify the best candidate (max similarity)
    best_id = max(sims, key=sims.get)
    best_sim = sims[best_id]

    # Hoeffding bound on similarity (range r = 1 because similarity ∈ [0,1])
    eps = hoeffding_bound(r=1.0, delta=delta, n=n_obs)

    # If the best similarity exceeds the current active similarity by more than ε,
    # we switch.
    active_sim = sims.get(active_id, 0.0)
    if best_sim - active_sim > eps:
        return best_id, True
    else:
        return active_id, False


# ---------------------------------------------------------------------------
# Endpoint statistics update (streaming observation)
# ---------------------------------------------------------------------------

def hybrid_update_endpoint(
    endpoint: Endpoint,
    failure_observed: bool,
    recovery_signal: float,
    alpha: float = 0.1,
) -> None:
    """
    Update endpoint statistics with a new observation.

    * `failure_observed` – boolean indicating a failure event.
    * `recovery_signal` – scalar (≥0) indicating recovery priority boost.
    * `alpha` – exponential moving average factor for smoothing.
    """
    # Update failure rate (EMA of binary failure indicator)
    current = endpoint.failure_rate
    endpoint.failure_rate = (1 - alpha) * current + alpha * (1.0 if failure_observed else 0.0)

    # Update recovery priority (EMA)
    current_r = endpoint.recovery_priority
    endpoint.recovery_priority = (1 - alpha) * current_r + alpha * recovery_signal

    # Re‑compute health score after the update
    endpoint.health_score = tropical_health(endpoint)


# ---------------------------------------------------------------------------
# High‑level hybrid workflow demonstration
# ---------------------------------------------------------------------------

def hybrid_step(
    endpoints: List[Endpoint],
    active_id: str,
    observation: Tuple[str, bool, float],
    delta: float = 0.05,
    n_obs: int = 30,
    dim: int = 10000,
) -> Tuple[str, bool]:
    """
    Perform a single processing step:
    1. Update the observed endpoint statistics.
    2. Re‑compute all health scores.
    3. Possibly switch the active endpoint.

    Parameters
    ----------
    endpoints : List[Endpoint]
        List of all endpoints.
    active_id : str
        Currently active endpoint identifier.
    observation : Tuple[id, failure_flag, recovery_signal]
        New data for one endpoint.
    delta, n_obs, dim : passed to the switching routine.

    Returns
    -------
    Tuple[str, bool]
        Updated active identifier and a flag indicating whether a switch occurred.
    """
    obs_id, failure_flag, recovery_sig = observation

    # Locate the endpoint and update its statistics
    ep_dict = {ep.id: ep for ep in endpoints}
    if obs_id not in ep_dict:
        raise KeyError(f"Endpoint {obs_id} not found.")
    hybrid_update_endpoint(ep_dict[obs_id], failure_flag, recovery_sig)

    # Re‑compute health scores for all endpoints (they may depend on shared state)
    hybrid_compute_health_scores(endpoints)

    # Decide on switching
    new_active, switched = hybrid_maybe_switch(
        endpoints, active_id, delta=delta, n_obs=n_obs, dim=dim
    )
    return new_active, switched


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a small set of endpoints
    eps = [
        Endpoint(id="A", failure_rate=0.02, recovery_priority=0.8),
        Endpoint(id="B", failure_rate=0.10, recovery_priority=0.4),
        Endpoint(id="C", failure_rate=0.05, recovery_priority=0.6),
    ]

    # Initial health computation
    hybrid_compute_health_scores(eps)

    # Choose an initial active endpoint arbitrarily
    active = "A"
    print(f"Initial active endpoint: {active}")

    # Simulate a stream of observations
    observations = [
        ("A", False, 0.05),
        ("B", True, 0.10),
        ("C", False, 0.02),
        ("B", False, 0.07),
        ("A", True, 0.20),
        ("C", False, 0.03),
    ]

    for step, obs in enumerate(observations, 1):
        active, switched = hybrid_step(eps, active, obs, delta=0.05, n_obs=20, dim=4096)
        print(f"Step {step:02d}: obs={obs} -> active={active} (switched={switched})")

    # Demonstrate Fisher‑to‑hypervector conversion
    theta = 0.42
    hv = fisher_to_hv(theta, dim=4096)
    print(f"Fisher hypervector for theta={theta:.2f} has mean value {np.mean(hv):.3f}")

    sys.exit(0)