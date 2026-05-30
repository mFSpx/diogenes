# DARWIN HAMMER — match 53, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s1.py (gen3)
# parent_b: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s2.py (gen2)
# born: 2026-05-29T23:26:33Z

"""Hybrid Bayesian‑SSIM‑Voronoi Engine.

Parents:
* **hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s1** – provides a
  Structural Similarity (SSIM) based likelihood and a simple Bayesian update
  for packet‑to‑brain‑map inference.
* **hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s2** – supplies a
  geometric‑morphological health‑distance score for assigning points to
  endpoints.

Mathematical bridge:
Both parents produce a *score* in the unit interval [0, 1] that can be
interpreted as a likelihood.  The hybrid engine multiplies the SSIM‑derived
likelihood 𝓁_ssim(packet) with the Voronoi‑health‑distance likelihood
𝓁_geo(endpoint, point) and raises each factor to a configurable weight.
The product is a weighted geometric mean that remains in [0, 1] and can be
used directly as a Bayesian posterior (with a uniform prior) or as a
selection criterion for a bandit‑style routing policy.

Unified score for a packet *p* routed to endpoint *e* from spatial point *x*:

    S(p, e, x) = 𝓁_ssim(p)^{w_s} · 𝓁_geo(e, x)^{1‑w_s}

where 𝓁_ssim ∈ [0,1] is the SSIM similarity between the packet payload and a
prototype vector, and 𝓁_geo is the health‑distance score defined in the
Voronoi parent.  The weight w_s ∈ (0,1) balances visual similarity against
geometric‑morphological fitness.

The module implements the core operations, a Bayesian posterior updater,
and a simple assignment routine that selects the endpoint maximising S.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants & Prototype
# ----------------------------------------------------------------------
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

# ----------------------------------------------------------------------
# SSIM (Parent A)
# ----------------------------------------------------------------------
def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index (SSIM) between two equal‑length vectors."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)


def ssim_likelihood(payload: List[float]) -> float:
    """Likelihood derived from SSIM between payload and the prototype vector."""
    # Align dimensions to prototype length
    payload_vec = np.asarray(payload, dtype=np.float64)
    if payload_vec.size < PROTOTYPE_VECTOR.size:
        payload_vec = np.pad(payload_vec, (0, PROTOTYPE_VECTOR.size - payload_vec.size))
    elif payload_vec.size > PROTOTYPE_VECTOR.size:
        payload_vec = payload_vec[: PROTOTYPE_VECTOR.size]
    return compute_ssim(payload_vec.tolist(), PROTOTYPE_VECTOR.tolist(), dynamic_range=1.0)


# ----------------------------------------------------------------------
# Voronoi‑Health‑Distance (Parent B)
# ----------------------------------------------------------------------
@dataclass
class Endpoint:
    """Morphological endpoint used in the Voronoi score."""
    seed: np.ndarray                # coordinate vector (shape (d,))
    reliability: bool               # True = circuit closed
    priority: float                 # recovery priority (higher → more urgent)
    sphericity: float               # compactness index (0‑1)
    flatness: float                 # flatness index (0‑1, 0 = perfectly flat)

    def reliability_factor(self) -> float:
        return 1.0 if self.reliability else 0.0


def health_distance_score(
    endpoint: Endpoint,
    point: np.ndarray,
    max_dist: float,
    weights: Dict[str, float],
) -> float:
    """
    Geometric‑morphological score S_geo(e, p) defined in the Voronoi parent.
    The score is a weighted geometric mean of five factors and lives in [0,1].
    """
    # reliability
    r = endpoint.reliability_factor()
    # normalized distance term
    d = np.linalg.norm(point - endpoint.seed)
    dist_term = max(0.0, 1.0 - d / max_dist) if max_dist > 0 else 0.0
    # priority (already in [0,1] after normalisation by caller)
    p = endpoint.priority
    # sphericity
    sph = endpoint.sphericity
    # inverse flatness (higher flatness → lower contribution)
    flat_inv = 1.0 / endpoint.flatness if endpoint.flatness > 0 else 0.0

    # Apply weights (they sum to 1)
    components = {
        "reliability": (r, weights["reliability"]),
        "distance": (dist_term, weights["distance"]),
        "priority": (p, weights["priority"]),
        "sphericity": (sph, weights["sphericity"]),
        "flatness": (flat_inv, weights["flatness"]),
    }
    # Weighted geometric mean
    log_sum = 0.0
    for _, (val, w) in components.items():
        # Clamp to a tiny epsilon to avoid log(0)
        val = max(val, 1e-12)
        log_sum += w * math.log(val)
    return float(math.exp(log_sum))


def compute_max_distance(endpoints: List[Endpoint], points: List[np.ndarray]) -> float:
    """Maximum Euclidean distance between any point and any endpoint seed."""
    max_d = 0.0
    for p in points:
        for e in endpoints:
            d = np.linalg.norm(p - e.seed)
            if d > max_d:
                max_d = d
    return max_d


# ----------------------------------------------------------------------
# Unified Hybrid Score & Bayesian Update
# ----------------------------------------------------------------------
def unified_score(
    payload: List[float],
    endpoint: Endpoint,
    point: np.ndarray,
    max_dist: float,
    w_ssim: float,
    geo_weights: Dict[str, float],
) -> float:
    """
    Combined likelihood S = L_ssim^{w_ssim} * L_geo^{1‑w_ssim}.
    Both L_ssim and L_geo lie in [0,1]; the result also lies in [0,1].
    """
    l_ssim = ssim_likelihood(payload)
    l_geo = health_distance_score(endpoint, point, max_dist, geo_weights)
    # Clamp to avoid under‑flow
    l_ssim = max(l_ssim, 1e-12)
    l_geo = max(l_geo, 1e-12)
    return float(l_ssim ** w_ssim * l_geo ** (1.0 - w_ssim))


def bayesian_posterior_update(
    priors: Dict[int, float],
    payload: List[float],
    endpoint: Endpoint,
    point: np.ndarray,
    max_dist: float,
    w_ssim: float,
    geo_weights: Dict[str, float],
) -> Dict[int, float]:
    """
    Perform a single Bayesian update for one observation.
    `priors` maps endpoint indices to prior probabilities (must sum to 1).
    Returns a new posterior dictionary (also normalised).
    """
    numerators = {}
    for idx, prior in priors.items():
        # Re‑use the endpoint from the list (caller ensures correspondence)
        post = prior * unified_score(
            payload,
            endpoint,
            point,
            max_dist,
            w_ssim,
            geo_weights,
        )
        numerators[idx] = post
    total = sum(numerators.values())
    if total == 0.0:
        # Avoid division by zero: revert to uniform distribution
        n = len(priors)
        return {k: 1.0 / n for k in priors}
    return {k: v / total for k, v in numerators.items()}


def assign_packets(
    packets: List[Dict[str, List[float]]],
    endpoints: List[Endpoint],
    points: List[np.ndarray],
    w_ssim: float = 0.4,
    geo_weights: Dict[str, float] = None,
) -> List[Tuple[int, int]]:
    """
    Assign each packet to the endpoint that maximises the unified score.
    Returns a list of (packet_index, endpoint_index) tuples.
    """
    if geo_weights is None:
        geo_weights = {
            "reliability": 0.2,
            "distance": 0.2,
            "priority": 0.2,
            "sphericity": 0.2,
            "flatness": 0.2,
        }
    # Normalise priority across endpoints for a fair geometric mean
    max_prio = max(e.priority for e in endpoints) or 1.0
    for e in endpoints:
        e.priority = e.priority / max_prio

    max_dist = compute_max_distance(endpoints, points)
    assignments = []
    for pkt_idx, packet in enumerate(packets):
        payload = packet.get("payload", [])
        best_score = -1.0
        best_e_idx = -1
        for e_idx, endpoint in enumerate(endpoints):
            point = points[pkt_idx]  # assume one‑to‑one mapping of packet↔point
            score = unified_score(payload, endpoint, point, max_dist, w_ssim, geo_weights)
            if score > best_score:
                best_score = score
                best_e_idx = e_idx
        assignments.append((pkt_idx, best_e_idx))
    return assignments


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic scenario
    random.seed(0)
    np.random.seed(0)

    # Endpoints (3 of them) with random seeds and morphology
    endpoints = [
        Endpoint(
            seed=np.random.rand(3),
            reliability=bool(random.getrandbits(1)),
            priority=random.uniform(0.5, 1.0),
            sphericity=random.uniform(0.4, 1.0),
            flatness=random.uniform(0.2, 0.9),
        )
        for _ in range(3)
    ]

    # Points in the same 3‑D space (one per packet)
    points = [np.random.rand(3) for _ in range(5)]

    # Packets with random payloads of varying length
    packets = [
        {"payload": np.random.rand(random.randint(3, 7)).tolist()}
        for _ in range(5)
    ]

    # Perform assignment
    assignments = assign_packets(packets, endpoints, points)

    # Print results
    for pkt_idx, e_idx in assignments:
        print(
            f"Packet {pkt_idx} assigned to Endpoint {e_idx} | "
            f"Payload SSIM={ssim_likelihood(packets[pkt_idx]['payload']):.3f}"
        )
    # Demonstrate a Bayesian posterior update after the first observation
    priors = {i: 1.0 / len(endpoints) for i in range(len(endpoints))}
    post = bayesian_posterior_update(
        priors,
        packets[0]["payload"],
        endpoints[assignments[0][1]],
        points[0],
        compute_max_distance(endpoints, points),
        w_ssim=0.4,
        geo_weights={
            "reliability": 0.2,
            "distance": 0.2,
            "priority": 0.2,
            "sphericity": 0.2,
            "flatness": 0.2,
        },
    )
    print("Posterior after first packet:", {k: f"{v:.3f}" for k, v in post.items()})