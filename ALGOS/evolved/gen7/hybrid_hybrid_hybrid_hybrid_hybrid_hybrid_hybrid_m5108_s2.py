# DARWIN HAMMER — match 5108, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m1191_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1535_s1.py (gen6)
# born: 2026-05-29T23:59:51Z

"""Hybrid Perceptual‑Fisher & Tropical‑SSIM Engine Endpoint Scoring
=================================================================

This module fuses the two parent algorithms:

* **Parent A** – *hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m1191_s1.py*  
  Provides a Fisher‑weighted radial‑basis‑function similarity
  `S_fRBF(a, b) = G_ε(‖a‑b‖₂) · F_σ,μ(‖a‑b‖₂)`.

* **Parent B** – *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1535_s1.py*  
  Supplies a max‑plus (tropical) network, a structural‑similarity (SSIM)
  measure and a health‑score derived from morphology and recovery priority.

**Mathematical Bridge**  
Both parents expose a *scalar similarity* between feature vectors:

* Parent A – the Fisher‑weighted RBF similarity `S_fRBF`.
* Parent B – the SSIM between a tropical‑network output and its input,
  optionally weighted by a confidence scalar.

The hybrid algorithm therefore:

1. Computes `S_fRBF` for a pair of vectors.
2. Feeds the first vector through a tropical network, obtains `T(x)`,
   and evaluates a confidence‑weighted SSIM `S_SSIM`.
3. Multiplies the two similarities to obtain a **joint similarity**.
4. Combines this joint similarity with a morphology‑based health score
   to rank engine endpoints.

The three core functions below illustrate this fused pipeline.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Basic utilities shared by both parent lineages
# ----------------------------------------------------------------------


Vector = List[float]


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance ‖a‑b‖₂."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel 𝔾ε(r) = exp(‑(ε·r)²)."""
    return math.exp(-((epsilon * r) ** 2))


def fisher_score(r: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Simple Fisher‑information‑inspired weight for a Gaussian beam.
    For a Gaussian N(μ,σ²) the Fisher information w.r.t. the mean is
    1/σ²; we embed the distance `r` as a Gaussian likelihood term.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return math.exp(-(((r - mu) / sigma) ** 2))


def fisher_weighted_rbf(a: Vector, b: Vector,
                       epsilon: float = 1.0,
                       mu: float = 0.0,
                       sigma: float = 1.0) -> float:
    """
    Fisher‑weighted RBF similarity S_fRBF(a,b) = G_ε(r)·F_σ,μ(r)
    where r = ‖a‑b‖₂.
    """
    r = euclidean(a, b)
    return gaussian_rbf(r, epsilon) * fisher_score(r, mu, sigma)


# ----------------------------------------------------------------------
# Parent B components – Tropical network and SSIM
# ----------------------------------------------------------------------


class TropicalNetwork:
    """
    Max‑plus (tropical) linear layer.
    For each output dimension i:
        out[i] = max(0, w_i·x + b_i)
    """

    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        if weights.shape[0] != biases.shape[0]:
            raise ValueError("weights rows must match number of biases")
        self.weights = weights
        self.biases = biases

    def evaluate(self, input_vector: Vector) -> np.ndarray:
        """Apply the tropical transformation."""
        x = np.asarray(input_vector, dtype=float)
        out = np.empty(self.weights.shape[0], dtype=float)
        for i in range(self.weights.shape[0]):
            out[i] = max(0.0, float(np.dot(self.weights[i], x) + self.biases[i]))
        return out


def ssim(x: List[float],
         y: List[float],
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """
    Structural Similarity Index (SSIM) between two 1‑D signals.
    Implements the classic formula:
        (2μₓμ_y + C₁)(2σ_xy + C₂) / ((μₓ²+μ_y² + C₁)(σₓ²+σ_y² + C₂))
    """
    if len(x) != len(y):
        raise ValueError("inputs must have the same length")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mu_x = np.mean(x_arr)
    mu_y = np.mean(y_arr)

    sigma_x = np.std(x_arr)
    sigma_y = np.std(y_arr)

    sigma_xy = np.mean((x_arr - mu_x) * (y_arr - mu_y))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def weighted_ssim(input_vec: Vector,
                  network: TropicalNetwork,
                  confidence: float = 1.0) -> float:
    """
    Compute SSIM between the raw input and the tropical network output,
    then weight it by a scalar confidence (derived e.g. from a signal‑to‑noise gap).
    """
    if not (0.0 <= confidence <= 1.0):
        raise ValueError("confidence must lie in [0,1]")
    net_out = network.evaluate(input_vec).tolist()
    # Pad the shorter sequence so SSIM can be evaluated element‑wise.
    min_len = min(len(input_vec), len(net_out))
    s = ssim(input_vec[:min_len], net_out[:min_len])
    return confidence * s


# ----------------------------------------------------------------------
# Health‑score and endpoint modelling (Parent B side)
# ----------------------------------------------------------------------


class Morphology:
    """Simple morphological descriptor."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

    def volume(self) -> float:
        return self.length * self.width * self.height


def health_score(morph: Morphology,
                 failure_rate: float,
                 recovery_priority: float) -> float:
    """
    Composite health score:
        H = (1 - failure_rate) * (volume)^{recovery_priority}
    The volume term captures size‑related robustness; the exponent
    allows the recovery priority to amplify or dampen its effect.
    """
    if not (0.0 <= failure_rate <= 1.0):
        raise ValueError("failure_rate must be in [0,1]")
    vol = morph.volume()
    # Avoid zero or negative volumes for the power operation.
    vol = max(vol, 1e-9)
    return (1.0 - failure_rate) * (vol ** recovery_priority)


class EngineEndpoint:
    """Container for endpoint metadata used in ranking."""
    def __init__(self,
                 engine_id: str,
                 channel: str,
                 residency: str,
                 runtime: str,
                 resource_class: str,
                 always_on: bool,
                 endpoint: str,
                 capabilities: List[str],
                 morphology: Morphology):
        self.engine_id = engine_id
        self.channel = channel
        self.residency = residency
        self.runtime = runtime
        self.resource_class = resource_class
        self.always_on = always_on
        self.endpoint = endpoint
        self.capabilities = capabilities
        self.morphology = morphology


def endpoint_score(endpoint: EngineEndpoint,
                   similarity: float,
                   health: float) -> float:
    """
    Final ranking score for an engine endpoint.
    The joint similarity (from the hybrid A‑B pipeline) is multiplied
    by the health score, yielding a scalar that can be maximised.
    """
    return similarity * health


# ----------------------------------------------------------------------
# Core hybrid functions (demonstration of the fused topology)
# ----------------------------------------------------------------------


def hybrid_similarity(vec_a: Vector,
                      vec_b: Vector,
                      epsilon: float = 1.0,
                      mu: float = 0.0,
                      sigma: float = 1.0,
                      network: TropicalNetwork = None,
                      confidence: float = 1.0) -> float:
    """
    Compute the joint similarity:
        J = S_fRBF(a,b) * (confidence·SSIM(a, T(a)))
    If `network` is None, the SSIM term defaults to 1.0 (i.e. ignored).
    """
    s_frbf = fisher_weighted_rbf(vec_a, vec_b, epsilon, mu, sigma)

    if network is not None:
        s_ssim = weighted_ssim(vec_a, network, confidence)
    else:
        s_ssim = 1.0

    return s_frbf * s_ssim


def rank_endpoints(query_vec: Vector,
                   candidate_vecs: List[Vector],
                   endpoints: List[EngineEndpoint],
                   network: TropicalNetwork,
                   epsilon: float = 1.0,
                   mu: float = 0.0,
                   sigma: float = 1.0,
                   confidence: float = 1.0,
                   failure_rate: float = 0.1,
                   recovery_priority: float = 1.0) -> List[Tuple[EngineEndpoint, float]]:
    """
    Rank a list of engine endpoints against a query vector.
    For each candidate vector we compute the hybrid similarity to the query,
    then combine it with the endpoint's health score.
    Returns a list of (endpoint, combined_score) sorted descendingly.
    """
    if not (len(candidate_vecs) == len(endpoints)):
        raise ValueError("candidate_vecs and endpoints must be same length")

    results = []
    for vec, ep in zip(candidate_vecs, endpoints):
        sim = hybrid_similarity(query_vec, vec,
                                epsilon, mu, sigma,
                                network, confidence)
        health = health_score(ep.morphology, failure_rate, recovery_priority)
        combined = endpoint_score(ep, sim, health)
        results.append((ep, combined))

    results.sort(key=lambda pair: pair[1], reverse=True)
    return results


def perceptual_cluster_hash(vectors: List[Vector], bits: int = 8) -> dict:
    """
    Very lightweight perceptual‑hash clustering.
    For each vector we compute a deterministic integer hash from its
    quantised values, then group vectors whose hashes differ by at most
    `bits` bits. The function returns a dict mapping a representative hash
    to the list of member vectors.
    """
    def simple_hash(vec: Vector) -> int:
        # Quantise to 0/1 per component and pack into an int.
        q = [int(x > 0) for x in vec]
        h = 0
        for bit in q:
            h = (h << 1) | bit
        return h

    clusters = {}
    for v in vectors:
        h = simple_hash(v)
        # Find an existing cluster within Hamming distance <= bits.
        found_key = None
        for key in clusters:
            if bin(key ^ h).count('1') <= bits:
                found_key = key
                break
        if found_key is None:
            clusters[h] = [v]
        else:
            clusters[found_key].append(v)
    return clusters


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create two random 16‑dimensional vectors
    vec1 = [random.uniform(-10, 10) for _ in range(16)]
    vec2 = [random.uniform(-10, 10) for _ in range(16)]

    # Build a tiny tropical network (4 outputs)
    w = np.random.randn(4, 16)
    b = np.random.randn(4)
    trop_net = TropicalNetwork(w, b)

    # Demonstrate core hybrid similarity
    joint_sim = hybrid_similarity(vec1, vec2,
                                  epsilon=0.5,
                                  mu=0.0,
                                  sigma=2.0,
                                  network=trop_net,
                                  confidence=0.85)
    print(f"Joint hybrid similarity: {joint_sim:.6f}")

    # Create mock engine endpoints with simple morphologies
    endpoints = []
    candidates = []
    for i in range(5):
        morph = Morphology(length=random.uniform(0.5, 3.0),
                           width=random.uniform(0.5, 3.0),
                           height=random.uniform(0.5, 3.0),
                           mass=random.uniform(1.0, 10.0))
        ep = EngineEndpoint(engine_id=f"eng{i}",
                            channel="alpha",
                            residency="edge",
                            runtime="python3.11",
                            resource_class="standard",
                            always_on=False,
                            endpoint=f"https://engine{i}.example.com",
                            capabilities=["compute", "store"],
                            morphology=morph)
        endpoints.append(ep)
        # Each candidate vector is a noisy version of vec1
        noisy = [x + random.gauss(0, 0.5) for x in vec1]
        candidates.append(noisy)

    # Rank the endpoints
    ranked = rank_endpoints(query_vec=vec1,
                            candidate_vecs=candidates,
                            endpoints=endpoints,
                            network=trop_net,
                            epsilon=0.7,
                            mu=0.0,
                            sigma=1.5,
                            confidence=0.9,
                            failure_rate=0.07,
                            recovery_priority=1.2)

    print("\nEndpoint ranking (best first):")
    for ep, score in ranked:
        print(f"  {ep.engine_id}: score={score:.6f}")

    # Demonstrate perceptual clustering on the candidate vectors
    clusters = perceptual_cluster_hash(candidates, bits=2)
    print(f"\nNumber of perceptual clusters: {len(clusters)}")
    for h, members in clusters.items():
        print(f"  Hash {h:#06x}: {len(members)} members")