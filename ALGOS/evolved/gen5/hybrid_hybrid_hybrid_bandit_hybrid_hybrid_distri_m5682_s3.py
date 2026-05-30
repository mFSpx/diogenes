# DARWIN HAMMER — match 5682, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_hybrid_m896_s0.py (gen4)
# parent_b: hybrid_hybrid_distributed_l_hybrid_hybrid_fisher_m165_s0.py (gen4)
# born: 2026-05-30T00:04:19Z

import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Iterable

import numpy as np


# --------------------------- Core Mathematical Primitives ---------------------------

def gaussian_beam(x: float, mu: float, sigma: float) -> float:
    """Standard Gaussian kernel (unnormalized)."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z)


def fisher_score(x: float, mu: float, sigma: float, eps: float = 1e-12) -> float:
    """
    Fisher information of a single‑parameter Gaussian w.r.t. its mean.
    For a Gaussian N(mu, sigma²) the Fisher information I(μ)=1/σ².
    Here we compute a data‑driven analogue that respects the local intensity.
    """
    intensity = max(gaussian_beam(x, mu, sigma), eps)
    # derivative of log‑likelihood w.r.t. μ for a single observation
    derivative = (x - mu) / (sigma * sigma)
    return (derivative * derivative) / intensity


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


# --------------------------- Perceptual Hashing Utilities ---------------------------

def compute_phash(values: List[float]) -> int:
    """
    Deterministic 64‑bit perceptual hash.
    Bits are set according to whether each value exceeds the median of the
    (up‑to‑64) first entries.
    """
    if not values:
        return 0
    # Use median for robustness against outliers
    median = np.median(values[:64])
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v > median)
    return bits


def pairwise_similarity(hashes: Dict[int, int]) -> Dict[Tuple[int, int], float]:
    """
    Compute similarity weight w_ij = 1 - d(h_i, h_j)/64 for every unordered pair.
    """
    sim = {}
    nodes = list(hashes.keys())
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            d = hamming_distance(hashes[u], hashes[v])
            w = 1.0 - d / 64.0
            sim[(u, v)] = w
            sim[(v, u)] = w
    return sim


def cluster_by_phash(
    hashes: Dict[int, int],
    max_distance: int = 4,
) -> List[List[int]]:
    """
    Agglomerative single‑link clustering using Hamming distance.
    Guarantees deterministic output by processing nodes in sorted order.
    """
    nodes = sorted(hashes.keys())
    clusters: List[List[int]] = []
    for n in nodes:
        placed = False
        for cluster in clusters:
            # distance to the *representative* (first element) of the cluster
            rep = cluster[0]
            if hamming_distance(hashes[n], hashes[rep]) <= max_distance:
                cluster.append(n)
                placed = True
                break
        if not placed:
            clusters.append([n])
    return clusters


# --------------------------- Temperature‑Dependent Reward ---------------------------

def arrhenius_rate(temp_k: float, params: Dict[str, float]) -> float:
    """
    Classic Arrhenius equation:
        k(T) = A * exp(-E_a / (R * T))
    Here we expose A, E_a and R via the params dict.
    """
    if temp_k <= 0.0:
        raise ValueError("Temperature in Kelvin must be positive")
    A = params.get("A", 1.0)
    E_a = params.get("E_a", 12000.0)      # activation energy (J·mol⁻¹)
    R = params.get("R", 8.314)           # universal gas constant (J·mol⁻¹·K⁻¹)
    return A * math.exp(-E_a / (R * temp_k))


def temperature_dependent_reward(temp_c: float, params: Dict[str, float]) -> float:
    """
    Combine Arrhenius rate with a Fisher‑information‑based temperature
    sensitivity term. The Fisher term uses a Gaussian centred at 0 °C with
    a width that reflects biologically plausible temperature variance.
    """
    temp_k = temp_c + 273.15
    rate = arrhenius_rate(temp_k, params)
    # Fisher score evaluates how informative the current temperature is
    # relative to the biologically optimal temperature (0 °C here).
    fisher = fisher_score(temp_c, mu=0.0, sigma=5.0)
    return rate * fisher


# --------------------------- Hybrid Node Update Logic ---------------------------

class Node:
    """
    Minimal representation of a graph node for the hybrid algorithm.
    """
    __slots__ = ("id", "features", "theta", "temp_c", "hash", "neighbors")

    def __init__(self, node_id: int, features: List[float], theta: float, temp_c: float):
        self.id = node_id
        self.features = features
        self.theta = theta                # e.g. a spatial localisation parameter
        self.temp_c = temp_c
        self.hash = compute_phash(features)
        self.neighbors: List[int] = []    # populated externally


def weighted_fisher_information(
    node: Node,
    nodes: Dict[int, Node],
    sim_weights: Dict[Tuple[int, int], float],
    sigma_fisher: float = 2.0,
) -> float:
    """
    Aggregate Fisher information of `node` with respect to its undecided neighbours,
    weighted by perceptual similarity.
    """
    total = 0.0
    for nb_id in node.neighbors:
        w = sim_weights.get((node.id, nb_id), 0.0)
        if w == 0.0:
            continue
        nb = nodes[nb_id]
        # Fisher score between the two localisation parameters
        f = fisher_score(node.theta, nb.theta, sigma_fisher)
        total += w * f
    return total


def broadcast_probability(
    node: Node,
    phase_step: int,
    undecided: set,
    nodes: Dict[int, Node],
    sim_weights: Dict[Tuple[int, int], float],
) -> float:
    """
    Compute the probability that `node` will broadcast in the current MIS phase.
    The base probability decays exponentially with the phase step.
    It is further attenuated by both perceptual similarity and the
    aggregated Fisher information, encouraging diverse and informative leaders.
    """
    if phase_step < 0:
        raise ValueError("phase_step must be non‑negative")
    p_raw = 1.0 / (2 ** phase_step)

    # similarity to undecided neighbours
    sim_sum = 0.0
    count = 0
    for nb_id in node.neighbors:
        if nb_id in undecided:
            sim_sum += sim_weights.get((node.id, nb_id), 0.0)
            count += 1
    sim_avg = sim_sum / count if count > 0 else 0.0

    # Fisher‑information‑based diversity term
    fisher_agg = weighted_fisher_information(node, nodes, sim_weights)

    # Combine multiplicatively; the Fisher term is placed in the denominator
    # so that high information (i.e. similar localisation) reduces aggressiveness.
    p_mod = p_raw * sim_avg / (1.0 + fisher_agg)

    # Clamp to a valid probability range
    return max(0.0, min(1.0, p_mod))


def hybrid_update(
    hypothesis: Dict,
    evidence: Dict,
    temp_c: float,
    reward_params: Dict[str, float],
) -> Dict:
    """
    Perform a Bayesian update of `hypothesis` using a likelihood derived from
    the temperature‑dependent reward. The likelihood ratio is normalised to
    stay within a numerically stable range.
    """
    raw_likelihood = temperature_dependent_reward(temp_c, reward_params)

    # Normalise to avoid exploding odds; we map the raw value to (0, 2] via tanh.
    likelihood = 1.0 + math.tanh(raw_likelihood)

    return update_hypothesis(hypothesis, evidence, likelihood)


def update_hypothesis(
    hypothesis: Dict,
    evidence: Dict,
    likelihood_ratio: float,
) -> Dict:
    """
    Classic Bayesian update on a Bernoulli hypothesis.
    """
    if likelihood_ratio < 0.0:
        raise ValueError("likelihood_ratio must be non‑negative")
    prior = max(0.0, min(1.0, hypothesis.get("posterior", hypothesis.get("prior", 0.5))))
    if prior in (0.0, 1.0) or likelihood_ratio == 0.0:
        posterior = prior
    else:
        odds = prior / (1.0 - prior)
        new_odds = odds * likelihood_ratio
        posterior = new_odds / (1.0 + new_odds)

    posterior = max(0.0, min(1.0, posterior))
    return {
        "id": hypothesis["id"],
        "prior": prior,
        "posterior": posterior,
        "evidence_ids": hypothesis.get("evidence_ids", []) + [evidence["id"]],
    }


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """
    Exponential decay used to prune stale candidates.
    """
    if t < 0.0 or lam < 0.0 or alpha < 0.0:
        raise ValueError("t, lam and alpha must be non‑negative")
    return math.exp(-lam * (t ** alpha))


# --------------------------- Example Execution ---------------------------

if __name__ == "__main__":
    # Build a tiny synthetic graph
    random.seed(42)
    nodes: Dict[int, Node] = {}
    for i in range(6):
        feats = [random.random() for _ in range(10)]
        theta = random.uniform(-10.0, 10.0)
        temp_c = random.uniform(15.0, 35.0)
        nodes[i] = Node(i, feats, theta, temp_c)

    # Fully connect the graph (for demonstration)
    for i in nodes:
        nodes[i].neighbors = [j for j in nodes if j != i]

    # Compute perceptual hashes and similarity matrix
    hashes = {n.id: n.hash for n in nodes.values()}
    sim_weights = pairwise_similarity(hashes)

    # Simulate one MIS broadcast step
    undecided = set(nodes.keys())
    phase_step = 0
    broadcast_probs = {
        n.id: broadcast_probability(n, phase_step, undecided, nodes, sim_weights)
        for n in nodes.values()
    }

    print("Broadcast probabilities per node:")
    for nid, prob in broadcast_probs.items():
        print(f"  Node {nid}: {prob:.4f}")

    # Perform a Bayesian update using temperature‑dependent reward
    hypothesis = {"id": 1, "prior": 0.5, "posterior": 0.5, "evidence_ids": []}
    evidence = {"action_id": "act_1", "id": 99}
    reward_params = {"A": 1.0, "E_a": 12000.0, "R": 8.314}
    updated = hybrid_update(hypothesis, evidence, temp_c=25.0, reward_params=reward_params)

    print("\nUpdated hypothesis:", updated)